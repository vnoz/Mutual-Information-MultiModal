from multi_modal import train_image_classifier, train_mutual_information
from mutual_info_img_txt.model import build_resnet_model
import torch 
import os
from helpers import construct_training_parameters
from uni_modal import train_auto_encoder


args =  construct_training_parameters()
args.save_directory = os.path.join(args.save_directory,
                                 f'{args.mi_estimator}_total_epochs{args.num_train_epochs}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


critics=['dv','infonce']
training_epochs= [5,10,20]

def train_MI_models():

    #Settings for Mutual Information
    #Note: around 3hrs for training 1 epoch for MI with batch_size=32 in 200k image-report pairs, 
    #        so time in total might be 2*(5+10+20)*2*3/24 = 17 days to complete, so might exclude batch_sizes varied steps and choose batch_size=128 to reduce running time to 8.5 days

    #critics=['dv','infonce']
    #training_epochs= [5,10,20] #Note: around 3hrs for training 1 epoch for MI with batch_size=32 in 200k image-report pairs
    #batch_sizes=[32,128]
    for critic in critics:
        for epoch in training_epochs:
            train_mutual_information(critic=critic, training_epoch=epoch, batch_size=128)

def train_VAE_models():
    #Settings for Variational Auto Encoder 
    critics=['ELBO','KL']
    for critic in critics:
        train_auto_encoder(critic=critic)

def train_Classifier():
    #Classifier settings
    diseases=['Pneumonia', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Edema','Pleural Effusion']
    training_epochs_classifier=100
    mlp_layers=[[[256,128,64],[256,128,64,32,16],[512,256,256,128,64,64,32]]]
    optimizers=['Adam','SGD']
    learning_rates=[1e-4,5e-4]

    #Load image model from MI or AutoEncoder
    for critic in critics:
        for epoch in training_epochs:
            sub_folder = critic+'_epoch'+epoch
            output_model_file = os.path.join(args.save_directory,sub_folder, 'pytorch_MI_image_model.bin')
            
            mi_image_model = build_resnet_model(model_name=args.image_model_name, checkpoint_path=output_model_file,
                                                            output_channels=args.output_channels)

            for label in diseases: 
                for hidden_layer in mlp_layers:
                    for optimizer in optimizers:
                        for learning_rate in learning_rates:
                            train_image_classifier(pre_trained_img_model = mi_image_model, label=label, training_epoch=training_epochs_classifier, mlp_hidden_layers=hidden_layer, optimizer=optimizer, learning_rate=learning_rate)


