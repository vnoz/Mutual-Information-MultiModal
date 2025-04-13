import os
import cv2
from tqdm import tqdm, trange
import logging
import numpy as np
import time

import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
import torch.nn as nn
from matplotlib import pyplot as plt

from helpers import get_transform_function

from .model import Basic_MLP, BasicBlock, ResNet256_6_2_1, build_bert_model, build_resnet_model
from .model import ImageReportModel
from .model import make_mlp
from .utils import MimicID, Plot_Training
from .model_utils import CXRImageDataset, CXRImageDiseaseDataset, CXRImageReportDataset, generate_GradCAM_image
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


class MultiModalManager:
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

		
		# '''
		# Shuffle and concatenate unmatched/negative pairs
		# '''
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
								 shuffle=True, num_workers=args.data_loader_workers,
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
		training_loss = []

		dataset_defaultValue = False
		for epoch in train_iterator:
			start_time = time.time()
			epoch_loss = 0
			epoch_steps = 0
			epoch_iterator = tqdm(data_loader, desc="Iteration")
			for i, batch in enumerate(epoch_iterator, 0):
				# Parse the batch 
				# Note the txt_ids is the tokenized txt
				img, txt_ids, txt_masks, txt_segments, study_id, img_id = batch

				#Note: set default value for dataset
				if(dataset_defaultValue == False):
					# print('img before send to device')
					# print(img[0])
					dataset.set_default(img[0].clone().detach(),txt_ids[0].clone().detach(),txt_masks[0].clone().detach(),txt_segments[0].clone().detach(), study_id[0])
					dataset_defaultValue = True

				img = img.to(device, non_blocking=True)
				txt_ids = txt_ids.to(device, non_blocking=True)
				txt_masks = txt_masks.to(device, non_blocking=True)
				txt_segments = txt_segments.to(device, non_blocking=True)
				# print('img after send to device')
				# print(img[0])
				
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
				# if total_steps % 5000 == 0:
				# 	projected_epoch_loss = len(data_loader)*epoch_loss/epoch_steps/args.batch_size
				# 	logger.info(f"  Projected epoch {epoch+1} loss = {projected_epoch_loss:.5f}")

			training_loss.append(epoch_loss)
			image_model_file_path= self.model.save_image_model(args.save_directory)
			text_model_file_path= self.model.save_text_model(args.save_directory)

			checkpoint_path = self.model.save_pretrained(args.save_directory, epoch=epoch + 1)
			interval = time.time() - start_time

			print(f'Epoch {epoch+1} finished! Epoch loss: {epoch_loss:.5f}')
			print(f'Epoch checkpoint saved in {checkpoint_path}')

			logger.info(f"  Epoch {epoch+1} loss = {epoch_loss:.5f}")
			logger.info(f"  Epoch {epoch+1} took {interval:.3f} s")
			logger.info(f"  Epoch {epoch+1} checkpoint saved in {checkpoint_path}")
			logger.info(f"  Image model saved in {image_model_file_path}")
			logger.info(f"  Text model saved in {text_model_file_path}")
		
		#Note: plot loss and accuracy and save to file

		plt.xlabel('Epochs')
		plt.ylabel('Value for Loss')
		
		plt.plot(training_loss,label="train loss")
		
		plt.legend()
		plt.show()
		plt.savefig(os.path.join(args.save_directory, 'mutual_information_training.png'))

		return


class ClassifierModelManager:
	""" A manager class that creates image classifier with input from image embeddings
			and metrics for classifier and heatmap generation
	"""

	def __init__(self, args, pre_trained_img_model,mlp_hidden_layers):
				
		self.args = args
		
		self.image_classifier_model = Basic_MLP(768,mlp_hidden_layers)

		self.pre_trained_img_model = pre_trained_img_model
		self.disease_label = args.disease_label
		data_loaders = self.construct_data_loader(args.disease_label)
		self.train_data_loader = data_loaders[0]
		self.validate_data_loader =  data_loaders[1]
		#self.class_weight = data_loaders[2]

	def construct_data_loader(self, label):

		'''
		Create an instance of traning data loader
		'''
		args = self.args
		dataset = CXRImageDiseaseDataset(img_dir=args.image_dir, 
									dataset_metadata=args.dataset_metadata, 
									disease=label,
									disease_stats=args.dataset_disease_stats,
									transform=get_transform_function(args.img_size))
		
		#NOTE: separate training and validate dataset/dataloader here, might need to split with balanced label classes
		train_size = int(0.95 * len(dataset))
		valid_size = len(dataset) - train_size

		train_ds, valid_ds = torch.utils.data.random_split(dataset, [train_size, valid_size])
		train_data_loader = DataLoader(train_ds, batch_size=args.batch_size,
								 shuffle=True, num_workers=args.data_loader_workers,
								 pin_memory=True, drop_last=True)
		
		validate_data_loader = DataLoader(valid_ds, batch_size=args.batch_size,
								 shuffle=True, num_workers=args.data_loader_workers,
								 pin_memory=True, drop_last=True)
		

		return train_data_loader, validate_data_loader 

	def train(self, device):
		
		args = self.args

		logger = logging.getLogger(__name__)
		logger.info(f"ExplainableImageModelManager training start, disease label= {self.disease_label}, args = {args}")


		'''
		Train the model
		'''		
		
		self.image_classifier_model = self.image_classifier_model.to(device)
		self.pre_trained_img_model = self.pre_trained_img_model.to(device)

		'''
		Define Loss function and optimizer
		'''
		
		# class_weights =  torch.tensor([21198/14232], dtype=torch.float32) 
		
		criterion = torch.nn.BCELoss().to(device) #nn.BCELoss(weight=class_weights).to(device)  #

		if(args.optimizer =='Adam'):
			optimizer = torch.optim.Adam(self.image_classifier_model.parameters(), lr=args.init_lr)
		else:
			optimizer = torch.optim.SGD(self.image_classifier_model.parameters(), lr=args.init_lr, weight_decay = 1e-08, momentum=0.0009, nesterov = True)

		#optimizer = torch.optim.Adam(self.image_classifier_model.parameters(), lr=args.init_lr, weight_decay = 1e-08, amsgrad=True)
		#optimizer = torch.optim.SGD(self.image_classifier_model.parameters(), lr=args.init_lr, weight_decay = 1e-08, momentum=0.0009, nesterov = True)
		#optimizer = torch.optim.SGD(self.image_classifier_model.parameters(), lr=args.init_lr,  momentum=0.009)
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

		total_batch = len(self.train_data_loader)

		logger.info(f"total batch of train_data_loader:, total_batch = {total_batch}")
		
		start_time = time.time()

		training_epoch_loss=[]
		validation_epoch_loss=[]
		validation_epoch_accuracy=[]
		max_accuracy = 0

		show_log_threshold = 2

		for epoch in range(args.num_train_epochs_classifier):
			self.image_classifier_model.train()
			step_loss=[]
			
			print('[Start Epoch: {:>4}]'.format(epoch + 1))
			start_time_epoch = time.time()

			#Note: Epoch training start
			training_epoch_iterator = tqdm(self.train_data_loader)
			#training_epoch_iterator = tqdm(self.train_data_loader, desc="***Training: train_data_loader Iteration")
			for batch_id, batch in enumerate(training_epoch_iterator, 0):

				image, label = batch
				image= image.to(device)
				output_image = self.pre_trained_img_model.forward(image)
				image_embeddings=output_image[1]
				image_embeddings= image_embeddings.to(device)
			

				label=label.to(torch.float32)
				label = label.to(device)

				optimizer.zero_grad()


				expectedLabel = self.image_classifier_model(image_embeddings)
				# if(batch_id< show_log_threshold):
				# 	print(f'Training BatchId: {batch_id}')
				# 	print(f"Image: {image.shape}")

				# 	print(f"Image embeddings: {image_embeddings.shape}")
				# 	print(image_embeddings)

				# 	print(f'Labels: {label}')
					
				# 	print(f'PredictedLabel: {expectedLabel}')

				expectedLabel= expectedLabel.to(torch.float32)
				expectedLabel= torch.flatten(expectedLabel)

				loss = criterion( expectedLabel, label)

				# if(batch_id< show_log_threshold):
					
				# 	print(f"loss={loss}")
					
				step_loss.append(loss.item())

				loss.backward()
				
				optimizer.step()
				

			scheduler.step()

			interval_epoch = time.time() - start_time_epoch
			
			training_epoch_loss.append(np.array(step_loss).mean())
			#Note: training completed for the current epoch


			#Note: perform accuracy calculation for Validation
			validate_data_iterators=tqdm(self.validate_data_loader, desc='Validation Accuracy calculation Iterations')
	
			total_validation = 	len(self.validate_data_loader)*args.batch_size
			val_count=0
			positive_count=0
			negative_count=0
			tp_count=0
			tn_count=0
			fp_count=0
			fn_count=0
			val_step_loss=[]
			for batch_id, batch in enumerate(validate_data_iterators, 0):
				image, label = batch
				image= image.to(device)
				output_image = self.pre_trained_img_model.forward(image)
				image_embeddings=output_image[1]
				image_embeddings= image_embeddings.to(device)

				output_result = self.image_classifier_model(image_embeddings)
				predictedLabel = torch.flatten(output_result).cpu().detach().numpy().round()

				output_result= output_result.to(torch.float32)
				output_result= torch.flatten(output_result)

				label=label.to(torch.float32)
				label = label.to(device)

				val_loss = criterion(output_result, label)

				label = label.cpu().detach().numpy()

				val_count = val_count + np.sum(predictedLabel == label).item()
				positive_count = positive_count + np.sum(label == 1).item()
				negative_count = negative_count + np.sum(label == 0).item()
				
				
				tp_count = tp_count +  np.sum(np.logical_and(predictedLabel == 1, label == 1))
				tn_count = tn_count + np.sum(np.logical_and(predictedLabel == 0, label == 0))  

				fp_count = fp_count + np.sum(np.logical_and(predictedLabel == 0, label == 1))
				fn_count = fn_count + np.sum(np.logical_and(predictedLabel == 1, label == 0))

				
				val_step_loss.append(val_loss.item())

				if(batch_id< show_log_threshold):
					print(f'Validation BatchId: {batch_id}')
					print(f'Labels: {label}')
					
					print(f'Classifier outputs: {output_result}')
					
					print(f'PredictedLabel: {predictedLabel}')
					
					print(f'tp_count: {tp_count}, tn_count: {tn_count}, fp_count: {fp_count}, fn_count: {fn_count}')
					
					print('--------------------')
		
			val_accuracy = val_count / total_validation
			validation_epoch_accuracy.append(val_accuracy)
			validation_epoch_loss.append(np.array(val_step_loss).mean())

			if(val_accuracy > max_accuracy):
				max_accuracy = val_accuracy
				
				checkpoint_path = self.image_classifier_model.save_pretrained(args.save_directory, self.disease_label+'_epoch'+str(epoch + 1))
				print(f'Save model for Epoch {epoch + 1} with accuracy {val_accuracy}: {checkpoint_path}')


			if((epoch + 1)%20 == 0):
				title = 'Training stats for disease ' + self.disease_label
				loss_output_file= os.path.join(args.save_directory, 'image_classifier_training_loss_'+self.disease_label+'_epoch'+str(epoch+1)+'.png')
				Plot_Training(xlabel='Epochs',ylabel='Value for Loss', title=title,data=[training_epoch_loss,validation_epoch_loss],dataLabel=['Training loss','Validation loss'],out_imgage_file=loss_output_file)
			

				accuracy_title = 'Training stats for disease ' + self.disease_label+'\n batch_size= ' + str(args.batch_size)+', batch = ' + str(total_batch)+'\n accuracy mean= '+ str("{:.5f}".format(np.array(validation_epoch_accuracy).mean()))
				accuracy_output_file= os.path.join(args.save_directory, 'image_classifier_training_accuracy_'+self.disease_label+'_epoch'+str(epoch+1)+'.png')
				Plot_Training(xlabel='Epochs',ylabel='Value for Accuracy', title=accuracy_title,data=[validation_epoch_accuracy],dataLabel=['validation accuracy'],out_imgage_file=accuracy_output_file)
	
		

			#Note: accuracy calculation completed in Training dataset and Validation dataset for current epoch


			#Note: logging for current epoch: start
			logger.info(f"Label: {self.disease_label},  Epoch {epoch+1} took {interval_epoch:.3f} s, training loss = {np.array(step_loss).mean():.5f}, validation_loss = {np.array(val_step_loss).mean():.5f}, validation accuracy={val_accuracy:.5f}")
			logger.info(f"  Total validation samples = {total_validation}, total positive={positive_count}, total negative={negative_count} ,tp_count={tp_count}, tn_count={tn_count}, fp_count={fp_count}, fn_count={fn_count}")
			print('[Epoch: {:>4}], time= {:.3f}, cost = {:>.9}, learning_rate = {:>.9}, validate accuracy = {:>.9}'.format(epoch + 1,interval_epoch, np.array(step_loss).mean(),np.array(scheduler.get_last_lr()).mean(),val_accuracy))
			
			print('Label='+self.disease_label+', Validation samples =  '+str(total_validation)+', total positive='+str(positive_count) +', total negative='+str(negative_count)+', tp_count='+str(tp_count)+', tn_count='+ str(tn_count)+',fp_count='+ str(fp_count)+', fn_count='+str(fn_count))

			#Note: logging for current epoch: end
			
		
		#Note: save img_classifier_model to file and add logs
		
		interval = time.time() - start_time


		print(f"Total  Epoch {epoch+1} took {interval:.3f} s")
		print('checkpoint_path: ' + str(checkpoint_path))

		logger.info('training loss:')
		logger.info(training_epoch_loss)

		logger.info('validation loss:')
		logger.info(validation_epoch_loss)

		logger.info('epoch training accuracy:')
		logger.info(validation_epoch_accuracy)
		logger.info(f"Training for {epoch+1} Epochs checkpoint saved in {checkpoint_path}")


		#Note: plot loss and accuracy and save to file

		title = 'Training stats for disease ' + self.disease_label
		loss_output_file= os.path.join(args.save_directory, 'image_classifier_training_loss_'+self.disease_label+'.png')
		Plot_Training(xlabel='Epochs',ylabel='Value for Loss', title=title,data=[training_epoch_loss,validation_epoch_loss],dataLabel=['Training loss','Validation loss'],out_imgage_file=loss_output_file)
	

		accuracy_title = 'Training stats for disease ' + self.disease_label+'\n batch_size= ' + str(args.batch_size)+', batch = ' + str(total_batch)+ ', time= '  + str("{:.2f}".format(interval))+'\n accuracy mean= '+ str("{:.5f}".format(np.array(validation_epoch_accuracy).mean()))
		accuracy_output_file= os.path.join(args.save_directory, 'image_classifier_training_accuracy_'+self.disease_label+'.png')
		Plot_Training(xlabel='Epochs',ylabel='Value for Accuracy', title=accuracy_title,data=[validation_epoch_accuracy],dataLabel=['validation accuracy'],out_imgage_file=accuracy_output_file)
	
		return self.image_classifier_model


class UniModalManager:
	""" A manager class that creates and manages the auto encoder model
		for image input only
	"""

	def __init__(self, output_channels, image_model_name):
		self.encoder = build_resnet_model(model_name=image_model_name, 
											  output_channels=output_channels)
		self.decoder = None
		self.logger = logging.getLogger(__name__)
	
	def train(self, device, args):
		'''
		Create a logger for logging model training
		'''
		logger = logging.getLogger(__name__)

	
		'''
		Create an instance of traning data loader
		'''
		print('***** Instantiate a data loader *****')
		dataset = CXRImageDataset(img_dir=args.image_dir,
								img_size=args.img_size,
								dataset_metadata=args.dataset_metadata)
		data_loader = DataLoader(dataset, batch_size=args.batch_size,
								 shuffle=True, num_workers=args.data_loader_workers,
								 pin_memory=True, drop_last=True)
		
		print(f'Total number of training images {len(dataset)}')

		'''
		Move encoder and decoder to device
		'''
		self.encoder = self.encoder.to(device)
		self.decoder = self.decoder.to(device)