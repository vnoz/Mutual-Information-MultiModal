import os
import argparse
import numpy as np
import torchvision
import torchvision.transforms as transforms

def construct_dataset_parameters():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser()

    #MIMIC settings to download dataset
    parser.add_argument('--download_user', type=str,
                        default='tuanle',
                        help='The user to download MIMIC dataset')

    parser.add_argument('--download_password', type=str,
                        default='A1thebest',
                        help='The password to download MIMIC dataset')


    #Amount of samples for full dataset, training and testing
    parser.add_argument('--total_amount', type=str,
                        default=10000,
                        help='Total amount of samples to download from MIMIC dataset')

    parser.add_argument('--amount_for_training', type=str,
                        default=9000,
                        help='Total amount of samples for training')


    parser.add_argument('--amount_for_testing', type=str,
                        default=1000,
                        help='Total amount of samples for testing')

    #Location for Full dataset 
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(current_dir, 'full_data_set/'),
                        help='The parent data directory')

    parser.add_argument('--image_storage_dir', type=str,
                        default=os.path.join(current_dir, 'full_data_set/images/'),
                        help='The image data directory')

    parser.add_argument('--text_storage_dir', type=str,
                        default=os.path.join(current_dir, 'full_data_set/text/'),
                        help='The text data directory')

    #Location for training dataset 
    parser.add_argument('--training_data_dir', type=str,
                        default=os.path.join(current_dir, 'testing_data/'),
                        help='The parent data directory')

    parser.add_argument('--training_image_dir', type=str,
                        default=os.path.join(current_dir, 'testing_data/images/'),
                        help='The training image data directory')
    parser.add_argument('--training_text_dir', type=str,
                        default=os.path.join(current_dir, 'testing_data/text/'),
                        help='The training text data directory')
    parser.add_argument('--training_dataset_metadata', type=str,
                        default=os.path.join(current_dir, 'testing_data/training_text_label_negbio.csv'),
                        help='The metadata for the model training ')

    #Location for testing dataset 
    parser.add_argument('--testing_data_dir', type=str,
                        default=os.path.join(current_dir, 'testing_data/'),
                        help='The parent directory for testing data')
    parser.add_argument('--testing_image_dir', type=str,
                        default=os.path.join(current_dir, 'testing_data/images/'),
                        help='The testing image data directory')
    parser.add_argument('--testing_text_dir', type=str,
                        default=os.path.join(current_dir, 'testing_data/text/'),
                        help='The testing text data directory')
    parser.add_argument('--testing_dataset_metadata', type=str,
                        default=os.path.join(current_dir, 'testing_data/testing_text_label_negbio.csv'),
                        help='The metadata for the model testing ')

    return parser.parse_args()

def construct_training_parameters():

    current_dir = os.path.dirname(os.path.abspath(__file__))
   
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir', type=str,
                        default=os.path.join(current_dir, 'training_data/images/'),
                        help='The image data directory')
    parser.add_argument('--text_data_dir', type=str,
                        default=os.path.join(current_dir, 'training_data/text/'),
                        help='The text data directory')
    parser.add_argument('--bert_pretrained_dir', type=str,
                        default=os.path.join(current_dir, 'bert_pretrain_all_notes_150000'),
                        help='The directory that contains a pretrained BERT model')
    parser.add_argument('--bert_config_name',
                        default='bert_config.json', help='Bert model config file')
    parser.add_argument('--save_dir', type=str,
                        default=os.path.join(current_dir, 'save_dir'))
    parser.add_argument('--dataset_metadata', type=str,
                        default=os.path.join(current_dir, 'training_data/training_text_label_negbio.csv'),
                        help='The metadata for the model training ')

    parser.add_argument('--batch_size', default=8, type=int,
                        help='Mini-batch size')
    parser.add_argument('--num_train_epochs', default=100, type=int,
                        help='Number of training epochs')
    parser.add_argument('--mi_estimator', type=str,
                        default='infonce',
                        help='Mutual information estimator (variational bound): dv or infonce')
    parser.add_argument('--init_lr', default=5e-4, type=float,
                        help='Intial learning rate')

    parser.add_argument('--max_seq_length', default=320, type=int,
                        help='Maximum sequence length for the BERT model')
    parser.add_argument('--img_size', default=256, type=int,
                        help='The size of the input image')
    parser.add_argument('--output_channels', default=1, type=int,
                        help='The number of ouput channels for the classifier')
    parser.add_argument('--image_model_name', default='resnet256_6_2_1', type=str,
                        help='Neural network architecture to be used for image model')
    
    parser.add_argument('--disease_label', default='Pleural Effusion', type=str,
                        help='Disease lable for downstream classifier')
    

    return parser.parse_args()

def get_transform_function(img_size):
    random_degrees = [-20,20]
    random_translate = [0.1,0.1]
    img_size = img_size

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
    return transform