import os
from tqdm import tqdm, trange
import logging
import numpy as np
import sklearn
import time

import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
import torch.nn as nn

from helpers import get_transform_function

from .model import Basic_MLP, build_bert_model, build_resnet_model
from .model import ImageReportModel
from .model import make_mlp
from .utils import MimicID
from .model_utils import CXRImageDataset, CXRImageReportDataset
from .mi_critics import dv_bound_loss, infonce_bound_loss


def build_training_imagereportset(text_token_features, img_dir, img_size: int, 
								  dataset_metadata='../data/training.csv',
								  random_degrees=[-20,20], random_translate=[0.1,0.1]):
	""" Build a image-report dataset for model training 
		with data augmentation on the images on the fly
	"""

	transform = torchvision.transforms.Compose([
		torchvision.transforms.Lambda(lambda img: img.astype(np.int16)),
		torchvision.transforms.ToPILImage(),
		torchvision.transforms.RandomAffine(degrees=random_degrees, translate=random_translate),
		torchvision.transforms.CenterCrop(img_size),
		torchvision.transforms.Lambda(
			lambda img: np.array(img).astype(np.float32)),
		torchvision.transforms.Lambda(
			lambda img: img / max(1e-3, img.max()))
	])
	training_dataset = CXRImageReportDataset(text_token_features=text_token_features,
											 img_dir=img_dir, 
											 dataset_metadata=dataset_metadata, 
											 transform=transform)

	return training_dataset


class ImageTextModelManager:
	""" A manager class that creates and manages the joint image-text model
		with global mutual information criterion 
	"""

	def __init__(self, bert_pretrained_dir, bert_config_name,
				 output_channels, image_model_name):
		self.bert_pretrained_dir = bert_pretrained_dir
		self.bert_config_name = bert_config_name
		self.output_channels = output_channels
		self.image_model_name = image_model_name

		self.text_model, self.bert_config = \
			build_bert_model(bert_pretrained_dir=bert_pretrained_dir,
							 bert_config_name=bert_config_name,
							 output_channels=output_channels)

		self.image_model = build_resnet_model(model_name=image_model_name, 
											  output_channels=output_channels)

		self.model = ImageReportModel(text_model=self.text_model,
									  bert_config=self.bert_config,
									  image_model=self.image_model)

		self.mi_discriminator = make_mlp(1536, [1024, 512])
		self.logger = logging.getLogger(__name__)

	def create_mi_pairs(self, embedding_img, embedding_txt, study_id: list, device):
		""" Concatenate image and text features and 
			in this way create pairs from two distrbutions for MI estimation.
			
			Args:
				study_id: a list of IDs that are unique to radiology reports; 
					a study_id only has one associated report but may have more than one CXR image 
		"""
		batch_size = len(study_id)

		'''
		Concatenate matched/positive pairs
		'''
		mi_input = torch.cat((embedding_img, embedding_txt), 1)

		'''
		Shuffle and concatenate unmatched/negative pairs
		'''
		for gap in range(batch_size-1):
			for i in range(batch_size):
				if i+(gap+1)<batch_size:
					j = i+(gap+1) 
				else:
					j = i+(gap+1) - batch_size
				if study_id[i] != study_id[j]:
					embedding_cat = torch.cat((embedding_img[i], embedding_txt[j]))
					embedding_cat = torch.reshape(embedding_cat, (1, embedding_cat.shape[0]))
					mi_input = torch.cat((mi_input, embedding_cat), 0)

		return mi_input

	def train(self, text_token_features, device, args):
		'''
		Create a logger for logging model training
		'''
		logger = logging.getLogger(__name__)

		'''
		Create an instance of traning data loader
		'''
		print('***** Instantiate a data loader *****')
		dataset = build_training_imagereportset(text_token_features=text_token_features,
												img_dir=args.image_dir,
												img_size=args.img_size,
												dataset_metadata=args.dataset_metadata)
		data_loader = DataLoader(dataset, batch_size=args.batch_size,
								 shuffle=True, num_workers=8,
								 pin_memory=True, drop_last=True)
		print(f'Total number of training image-report pairs: {len(dataset)}')

		'''
		Move models to device
		'''
		self.model = self.model.to(device)
		self.mi_discriminator = self.mi_discriminator.to(device)

		'''
		Define a loss criterion
		'''
		if args.mi_estimator == 'dv':
			mi_critic = dv_bound_loss
		if args.mi_estimator == 'infonce':
			mi_critic = infonce_bound_loss

		'''
		Create three instances of optimizer 
		(one for the image encoder, one for the MI estimator, and one for the text encoder)
		and a learning rate scheduler
		'''
		print('***** Instantiate an optimizer *****')
		img_optimizer = optim.Adam(self.model.image_model.parameters(), lr=args.init_lr)
		mi_optimizer = optim.Adam(self.mi_discriminator.parameters(), lr=args.init_lr)

		# For BERT-like text models, it appears important to use
		# AdamW and warmup linear learning rate schedule 
		# Refer to https://huggingface.co/transformers/training.html
		no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
		param_txt = list(self.model.text_model.named_parameters())
		grouped_parameters_txt = [
			{'params': [p for n, p in param_txt if not any(nd in n for nd in no_decay)], 
			'weight_decay': 0.1},
			{'params': [p for n, p in param_txt if any(nd in n for nd in no_decay)], 
			'weight_decay': 0.0}
			]
		txt_optimizer = AdamW(grouped_parameters_txt, 
							  lr=2e-5,
							  correct_bias=False)
		num_train_steps = int(args.num_train_epochs*len(data_loader))
		scheduler = WarmupLinearSchedule(txt_optimizer, 
										 warmup_steps=0.1*num_train_steps,
										 t_total=num_train_steps)

		'''
		Train the model
		'''
		print('***** Train the model *****')
		self.model.train()
		total_steps = 0
		train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
		for epoch in train_iterator:
			start_time = time.time()
			epoch_loss = 0
			epoch_steps = 0
			epoch_iterator = tqdm(data_loader, desc="Iteration")
			for i, batch in enumerate(epoch_iterator, 0):
				# Parse the batch 
				# Note the txt_ids is the tokenized txt
				img, txt_ids, txt_masks, txt_segments, study_id, img_id = batch
				img = img.to(device, non_blocking=True)
				txt_ids = txt_ids.to(device, non_blocking=True)
				txt_masks = txt_masks.to(device, non_blocking=True)
				txt_segments = txt_segments.to(device, non_blocking=True)

				# Zero out the parameter gradients
				img_optimizer.zero_grad()
				txt_optimizer.zero_grad()
				mi_optimizer.zero_grad()

				# Forward + backward + optimize
				inputs = {'img': img,
						  'txt_ids': txt_ids,
						  'txt_masks': txt_masks,
						  'txt_segments': txt_segments} 	
				outputs = self.model(**inputs)
				embedding_img, embedding_txt, logits_img, logits_txt = outputs
				mi_input = self.create_mi_pairs(embedding_img, embedding_txt, 
												study_id, device)
				mi_output = self.mi_discriminator(mi_input)

				loss = mi_critic(mi_output, args.batch_size, device)

				loss.backward()
				mi_optimizer.step()
				img_optimizer.step()
				txt_optimizer.step()
				scheduler.step()

				# Record training statistics
				epoch_loss += loss.item()

				total_steps += 1
				epoch_steps += 1
				if total_steps % 5000 == 0:
					projected_epoch_loss = len(data_loader)*epoch_loss/epoch_steps/args.batch_size
					logger.info(f"  Projected epoch {epoch+1} loss = {projected_epoch_loss:.5f}")

			image_model_file_path= self.model.save_image_model(args.save_dir)
			checkpoint_path = self.model.save_pretrained(args.save_dir, epoch=epoch + 1)
			interval = time.time() - start_time

			print(f'Epoch {epoch+1} finished! Epoch loss: {epoch_loss:.5f}')
			print(f'Epoch checkpoint saved in {checkpoint_path}')

			logger.info(f"  Epoch {epoch+1} loss = {epoch_loss:.5f}")
			logger.info(f"  Epoch {epoch+1} took {interval:.3f} s")
			logger.info(f"  Epoch {epoch+1} checkpoint saved in {checkpoint_path}")
			logger.info(f"  Image model saved in {image_model_file_path}")


		return
	
class ExplainableImageModelManager:
	""" A manager class that creates image classifier with input from image embeddings
			and heatmap generation with Grad-CAM
			and metrics for classifier and heatmap generation
	"""

	def __init__(self, args, using_pre_trained):
				
		self.args = args
		self.using_pre_trained = using_pre_trained

		self.image_classifier_model = Basic_MLP(768,[512,256,128])

		#NOTE: Load pre_trained image model from MI training
		output_model_file = os.path.join(args.save_dir, 'pytorch_MI_image_model.bin')
		
		self.pre_trained_img_model = build_resnet_model(model_name=args.image_model_name, checkpoint_path=output_model_file,
													output_channels=args.output_channels)
		
		data_loaders = self.construct_data_loader()
		self.test_data_loader = data_loaders[0]
		self.validate_data_loader =  data_loaders[1]
		
		# self.classifier_explanation = classifier_explanation_name
		# self.classifier_metric_name = classifier_metric_name
		# self.classifier_explanation_metric_name = classifier_explanation_metric_name

	def construct_data_loader(self):

		'''
		Create an instance of traning data loader
		'''
		args = self.args
		dataset = CXRImageDataset(img_dir=args.image_dir, 
									dataset_metadata=args.dataset_metadata, 
									disease=args.disease_label,
									transform=get_transform_function(args.img_size))
		
		#NOTE: separate training and validate dataset/dataloader here, might need to split with balanced label classes
		train_size = int(0.5 * len(dataset))
		valid_size = len(dataset) - train_size

		test_ds, valid_ds = torch.utils.data.random_split(dataset, [train_size, valid_size])
		test_data_loader = DataLoader(test_ds, batch_size=8,
								 shuffle=True, num_workers=8,
								 pin_memory=True, drop_last=True)
		
		validate_data_loader = DataLoader(test_ds, batch_size=8,
								 shuffle=True, num_workers=8,
								 pin_memory=True, drop_last=True)
		
		return test_data_loader, validate_data_loader

	def train(self, device):
		
		args = self.args

		logger = logging.getLogger(__name__)
		logger.info(f"ExplainableImageModelManager training start, args = {args}")

		#TODO: add code to load classifier from pre_trained model saved in file

		if(self.using_pre_trained):
			output_model_file = os.path.join(args.save_dir, 'pytorch_image_classifier_model.bin')
			# output_model_file = os.path.join(args.save_dir, 
            #                                  'pytorch_image_classifier_model_epoch'+str(args.num_train_epochs)+'.bin')
			self.image_classifier_model = self.image_classifier_model.from_pretrained(output_model_file)

			return
		else:

			'''
			Train the model
			'''		
			
			
			self.image_classifier_model = self.image_classifier_model.to(device)
			# self.pre_trained_img_model = self.pre_trained_img_model.to(device)

			'''
			Define Loss function and optimizer
			'''

			criterion = torch.nn.BCELoss().to(device) 
			optimizer = torch.optim.Adam(self.image_classifier_model.parameters(), lr=args.init_lr)

			total_batch = len(self.test_data_loader)

			print('total batch of test_data_loader: ' + str(total_batch))
			start_time = time.time()

			for epoch in range(args.num_train_epochs):
				
				avg_cost = 0
				print('[Start Epoch: {:>4}]'.format(epoch + 1))
				start_time_epoch = time.time()

				for image, label in self.test_data_loader:
				
					output_image = self.pre_trained_img_model.forward(image)
					image_embeddings=output_image[1]
					image_embeddings= image_embeddings.to(device)
					
					label = label.unsqueeze(1).to(device)

					optimizer.zero_grad()
					expectedLabel = self.image_classifier_model(image_embeddings)
					
					loss = criterion( expectedLabel, label.float())
					
					loss.backward()
					
					optimizer.step()

					avg_cost += loss.item() / total_batch

				print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
				logger.info(f"  Epoch {epoch+1} loss = {avg_cost:.5f}")
				interval_epoch = time.time() - start_time_epoch
				logger.info(f"  Epoch {epoch+1} took {interval_epoch:.3f} s")
				

			checkpoint_path = self.image_classifier_model.save_pretrained(args.save_dir)
			interval = time.time() - start_time

			print(f"Total  Epoch {epoch+1} took {interval:.3f} s")
			logger.info(f"  Epoch {epoch+1} checkpoint saved in {checkpoint_path}")

	def validate(self,device):	
		logger = logging.getLogger(__name__)

		count =0
		total_batch = len(self.validate_data_loader)

		showLog=True
		
		# self.image_classifier_model = self.image_classifier_model.to(device)

		for image, label in self.validate_data_loader:
			
				output_image = self.pre_trained_img_model.forward(image)
				image_embeddings=output_image[1]
				image_embeddings= image_embeddings.to(device)

				expectedLabel = self.image_classifier_model(image_embeddings)
				expectedLabel = expectedLabel.cpu().detach().numpy() #torch.flatten(expectedLabel).cpu().detach().numpy()

				if(showLog == True):
					print('Size of label, expectedLabel')
					print(label)
					print(expectedLabel)
					print(torch.sum(expectedLabel == label).item())
					
					showLog = False

				count = count + torch.sum(expectedLabel == label).item()
		
		accuracy = count / total_batch

		logger.info(f"ExplainableImageModelManager validate with accuracy = {accuracy}")

		return accuracy

	def generate_heatmap():
		
		'''
		TODO
		'''

	def generate_accuracy_metric():
		'''
		TODO: calculate accuracy metric for classifier from validate dataset
		'''

	def generate_IoU_metric():
		'''
		TODO: calculate IoU metric for heatmap from validate dataset with bounding box
		'''
