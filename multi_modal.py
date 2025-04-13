import os
import logging
from mutual_info_img_txt.utils import PrintModel
import torch 
from pytorch_transformers import BertTokenizer



from helpers import construct_training_parameters
from mutual_info_img_txt import model_utils
from mutual_info_img_txt.main_utils import ClassifierModelManager, MultiModalManager
from mutual_info_img_txt.model import build_resnet_model


def train_mutual_information(args, device):

    '''
    Create a sub-directory under save_dir
    '''
    
    if not os.path.exists(args.save_directory):
        os.makedirs(args.save_directory)

    '''
    Configure the log file
    '''
    log_path = os.path.join(args.save_directory, 'training_MI.log')
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
    print('tokens features: '+ str(len(text_token_features)))
    
    '''
    Initialize a joint image-text model manager
    '''
    model_manager = MultiModalManager(bert_pretrained_dir=args.bert_pretrained_dir,
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

    return model_manager.image_model

def train_image_classifier(pre_trained_img_model, mlp_hidden_layers, args, device):
    

    log_path = os.path.join(args.save_directory, 'training_classifier_'+args.disease_label+'.log')
    logging.basicConfig(filename=log_path, level=logging.INFO, filemode='w',
                                        format='%(asctime)s - %(name)s %(message)s',
                                        datefmt='%m-%d %H:%M')

    logger = logging.getLogger(__name__)
    

    model_manager = ClassifierModelManager( args,pre_trained_img_model, mlp_hidden_layers)

    print(f'Classifier Image Model initialise: ')
    PrintModel(model=model_manager.image_classifier_model)
    model_manager.train(device=device)
    return model_manager



