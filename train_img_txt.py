import os
import logging
import torch
from pytorch_transformers import BertTokenizer

from helpers import construct_training_parameters
from mutual_info_img_txt import model_utils
from mutual_info_img_txt.main_utils import ExplainableImageModelManager, ImageTextModelManager
from mutual_info_img_txt.model import build_resnet_model

args =  construct_training_parameters()
args.save_directory = os.path.join(args.save_directory,
                                 f'{args.mi_estimator}_total_epochs{args.num_train_epochs}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_image_text():

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

    return model_manager.image_model

#mi_image_model= train_image_text()


def train_image_classifier(pre_trained_img_model, using_pre_trained_classifier=True):
    
    
    print(f"Train_image_classifier args: {args}")

    log_path = os.path.join(args.save_directory, 'training_classifier.log')
    logging.basicConfig(filename=log_path, level=logging.INFO, filemode='w',
                                        format='%(asctime)s - %(name)s %(message)s',
                                        datefmt='%m-%d %H:%M')

    logger = logging.getLogger(__name__)
    

    model_manager = ExplainableImageModelManager( args=args, pre_trained_img_model= pre_trained_img_model, using_pre_trained_classifier=using_pre_trained_classifier)

    model_manager.train(device=device)
    return model_manager

using_pre_trained_image_text_model = True
using_pre_trained_classifier= False

image_classifider_model_manager: ExplainableImageModelManager # type: ignore

if(using_pre_trained_image_text_model == True):
    output_model_file = os.path.join(args.save_directory, 'pytorch_MI_image_model.bin')
		
    pre_trained_img_model = build_resnet_model(model_name=args.image_model_name, checkpoint_path=output_model_file,
													output_channels=args.output_channels)
		
    image_classifider_model_manager =  train_image_classifier(pre_trained_img_model = pre_trained_img_model, using_pre_trained_classifier=using_pre_trained_classifier)
else:
    mi_image_model= train_image_text()
    image_classifider_model_manager =  train_image_classifier(pre_trained_img_model = mi_image_model, using_pre_trained_classifier=using_pre_trained_classifier)

accuracy= image_classifider_model_manager.validate(device=device, batch_size=args.batch_size)
print(' accuracy = {:>.9}'.format(accuracy))
