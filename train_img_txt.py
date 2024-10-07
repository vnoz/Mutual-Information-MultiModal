import os
import argparse
import logging
import json
import numpy as np
import cv2

import torch
from pytorch_transformers import BertTokenizer

from helpers import construct_training_parameters
from mutual_info_img_txt import model_utils
from mutual_info_img_txt.main_utils import ExplainableImageModelManager, ImageTextModelManager
from mutual_info_img_txt.model import build_resnet_model


def train_image_text():

    args = construct_training_parameters()

    print(f"Initial args: {args}")

    '''
    Check cuda
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert torch.cuda.is_available(), "No GPU/CUDA is detected!"

    '''
    Create a sub-directory under save_dir
    '''
    args.save_dir = os.path.join(args.save_dir,
                                 f'{args.mi_estimator}_total_epochs{args.num_train_epochs}')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    '''
    Configure the log file
    '''
    log_path = os.path.join(args.save_dir, 'training_MI.log')
    logging.basicConfig(filename=log_path, level=logging.INFO, filemode='w',
                                        format='%(asctime)s - %(name)s %(message)s',
                                        datefmt='%m-%d %H:%M')

    logger = logging.getLogger(__name__)
    logger.info(f"args: {args}")

    print(f"Updated args: {args}")

    '''
    Tokenize text
    '''
    if not os.path.exists(args.bert_pretrained_dir):
        os.makedirs(args.bert_pretrained_dir)
    
    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrained_dir)
    text_token_features = model_utils.load_and_cache_examples(args, tokenizer)

    '''
    Initialize a joint image-text model manager
    '''
    model_manager = ImageTextModelManager(bert_pretrained_dir=args.bert_pretrained_dir,
                                          bert_config_name=args.bert_config_name,
                                          output_channels=args.output_channels,
                                          image_model_name=args.image_model_name)

    '''
    Train the joint model
    '''

    print(f"Start training for ImageTextModelManager")

    model_manager.train(text_token_features=text_token_features,
                        device=device,
                        args=args)
    print(f"Finish training for ImageTextModelManager")

#train_image_text()



def train_image_classifier():
    
    args =  construct_training_parameters()
    args.save_dir = os.path.join(args.save_dir,
                                 f'{args.mi_estimator}_total_epochs{args.num_train_epochs}')
    
    print(f"Train_image_classifier args: {args}")

    log_path = os.path.join(args.save_dir, 'training_classifier.log')
    logging.basicConfig(filename=log_path, level=logging.INFO, filemode='w',
                                        format='%(asctime)s - %(name)s %(message)s',
                                        datefmt='%m-%d %H:%M')

    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_manager = ExplainableImageModelManager( args=args, using_pre_trained=False)

    model_manager.train(device=device)

    accuracy = model_manager.validate(device=device)

    print('Accuracy for downstream image classifier: ' + str(accuracy))
    
train_image_classifier()