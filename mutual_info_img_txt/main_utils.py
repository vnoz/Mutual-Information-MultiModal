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

			checkpoint_path = self.model.save_pretrained(args.save_dir, epoch=epoch + 1)
			interval = time.time() - start_time

			print(f'Epoch {epoch+1} finished! Epoch loss: {epoch_loss:.5f}')
			print(f'Epoch checkpoint saved in {checkpoint_path}')

			logger.info(f"  Epoch {epoch+1} loss = {epoch_loss:.5f}")
			logger.info(f"  Epoch {epoch+1} took {interval:.3f} s")
			logger.info(f"  Epoch {epoch+1} checkpoint saved in {checkpoint_path}")

		return
	
class ExplainableImageModelManager:
	""" A manager class that creates image classifier with input from image embeddings
			and heatmap generation with Grad-CAM
			and metrics for classifier and heatmap generation
	"""

	# def __init__(self):
				#   ,classifier_explanation_name, classifier_metric_name, classifier_explanation_metric_name):
		# self.training_data_set= training_data_set
		# self.validate_data_set = validate_data_set
		# self.output_channels = output_channels
		# self.image_classifier_name = image_classifier_name
		# self.image_classifier_model = Basic_MLP(768,[512,256])
		# self.classifier_explanation = classifier_explanation_name
		# self.classifier_metric_name = classifier_metric_name
		# self.classifier_explanation_metric_name = classifier_explanation_metric_name

	def train(self, device, args):
		'''
		Train the model
		'''

		
		#NOTE: Load pre_trained image model from MI training
		save_dir = os.path.join(args.save_dir,
                                 f'{args.mi_estimator}_total_epochs{args.num_train_epochs}')
		output_model_file = os.path.join(save_dir, 
													'pytorch_model_epoch'+str(args.num_train_epochs)+'.bin')

		self.pre_trained_img_model = build_resnet_model(model_name=args.image_model_name, checkpoint_path=output_model_file,
													output_channels=args.output_channels)
		
		'''
		Create an instance of traning data loader
		'''

		random_degrees = [-20,20]
		random_translate = [0.1,0.1]
		img_size = args.img_size

		transform = transforms.Compose([
		torchvision.transforms.Lambda(lambda img: img.astype(np.int16)),
		torchvision.transforms.ToPILImage(),
		torchvision.transforms.RandomAffine(degrees=random_degrees, translate=random_translate),
		torchvision.transforms.CenterCrop(img_size),
		torchvision.transforms.Lambda(
			lambda img: np.array(img).astype(np.float32)),
		torchvision.transforms.Lambda(
			lambda img: img / max(1e-3, img.max()))
		])


		dataset = CXRImageDataset(img_dir=args.image_dir, 
									dataset_metadata=args.dataset_metadata, 
									disease='PleuralEffusion',
									transform=transform)
		
		data_loader = DataLoader(dataset, batch_size=8,
								 shuffle=True, num_workers=8,
								 pin_memory=True, drop_last=True)

		'''
		Define Loss function and optimizer
		'''
		image_classifier_model = Basic_MLP(768,[512,256,128]).to(device)
		
		criterion = torch.nn.BCELoss().to(device)    # Softmax is internally computed.
		optimizer = torch.optim.Adam(image_classifier_model.parameters(), lr=args.init_lr)

		total_batch = len(data_loader)

		start_time = time.time()

		for epoch in range(args.num_train_epochs):
    		
			avg_cost = 0

			for image, label in data_loader:
			
				output_image = self.pre_trained_img_model.forward(image)
				image_embeddings=output_image[1]
				image_embeddings= image_embeddings.to(device)
				
				label = label.to(device)

				optimizer.zero_grad()
				expectedLabel = image_classifier_model(image_embeddings)
				# print('***label: ' + str(label.item())+', size: ' )
				print(label.size())
				# print('expectedLabel: '+str(expectedLabel.item()) +', size: ')
				print(expectedLabel.size())
				loss = criterion( expectedLabel, label)
				
				loss.backward()
				
				optimizer.step()

				# avg_cost += cost / total_batch

			# print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
		
		checkpoint_path = self.image_classifier_model.save_pretrained(args.save_dir, epoch=epoch + 1)
		interval = time.time() - start_time

		print(f"  Epoch {epoch+1} took {interval:.3f} s")
		print(f'Epoch checkpoint saved in {checkpoint_path}')

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
