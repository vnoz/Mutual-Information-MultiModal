import datetime
from multi_modal import train_image_classifier, train_mutual_information
from mutual_info_img_txt.autoencoder_model import ResNetAE, ResNetEncoder
from mutual_info_img_txt.model import build_resnet_model
from mutual_info_img_txt.utils import PrintModel
import torch 
import os
from helpers import construct_training_parameters
from uni_modal import train_auto_encoder


args =  construct_training_parameters()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


critics=['dv']     #['dv','infonce']
training_epochs=[20]   #[5,10,20]
batch_sizes=[64]      #[32,64,128]

def train_MI_models():

    #Settings for Mutual Information
    #Note: around 3hrs for training 1 epoch for MI with batch_size=32 in 200k image-report pairs, 
    #        so time in total might be 2*(5+10+20)*2*3/24 = 17 days to complete, so might exclude batch_sizes varied steps and choose batch_size=128 to reduce running time to 8.5 days

    for critic in critics:
        for epoch in training_epochs:
            for batch_size in batch_sizes:
                args.mi_estimator = critic
                args.batch_size=batch_size
                args.num_train_epochs= epoch

                args.save_directory = os.path.join(args.save_directory,
                                    f'mm_{args.mi_estimator}_epoch{args.num_train_epochs}')
                train_mutual_information(args=args, device=device)

def train_AE_models():
    args.save_directory = os.path.join(args.save_directory,
                                    f'um_ae_epoch{args.num_train_epochs}')
    print(f'train_AE_models and save in folder {args.save_directory}')
    if not os.path.exists(args.save_directory):
        print(f'create new folder for {args.save_directory}')
        os.makedirs(args.save_directory)

    train_auto_encoder(args=args, device=device)

def train_Classifier(isMultiModal):
    #Classifier settings
    diseases=['Cardiomegaly', 'Pneumonia']  #['Cardiomegaly'] #['Pleural Effusion']        #['Pneumonia', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Edema','Pleural Effusion']
    training_epochs_classifier=100
    mlp_layers=[[1024,512,256,128,64,32,16]]   #[[[256,128,64],[256,128,64,32,16],[512,256,256,128,64,64,32]]]
    optimizers=['Adam']  #['Adam','SGD']
    learning_rates= [5e-4]  #[1e-4,5e-4]

    if(isMultiModal == True):
    #Load image model from MI or AutoEncoder
        for critic in critics:
            for epoch in training_epochs:
                for batch_size in batch_sizes:

                    args.mi_estimator = critic
                    args.batch_size=batch_size
                    args.num_train_epochs= epoch

                    args.save_directory = os.path.join(args.save_directory,
                                        f'{args.mi_estimator}_epoch{args.num_train_epochs}')
                

                    print(f'Args for Loading pre-trained MI Imag Model: {args}')  
                    output_model_file = os.path.join(args.save_directory, 'pytorch_MI_image_model.bin')
                    
                    mi_image_model = build_resnet_model(model_name=args.image_model_name, checkpoint_path=output_model_file,
                                                                    output_channels=args.output_channels)
                    
                    print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}: MI Image model loaded from {output_model_file}')
                    print(mi_image_model)
                    
                    PrintModel(model=mi_image_model)

                    for label in diseases: 
                        for hidden_layer in mlp_layers:
                            for optimizer in optimizers:
                                for learning_rate in learning_rates:
                                    args.init_lr = learning_rate
                                    args.num_train_epochs_classifier = training_epochs_classifier
                                    args.disease_label = label
                                    args.optimizer = optimizer
                                    
                                    print(f"Args for Classifier training: hidden layers={hidden_layer}, args= {args}")
                                    
                                    train_image_classifier(pre_trained_img_model = mi_image_model, isMultiModal=isMultiModal,mlp_hidden_layers=hidden_layer, args=args, device=device)
    else:
        args.save_directory = os.path.join(args.save_directory,
                                    f'um_ae_epoch{args.num_train_epochs}')
                
        print(f'Args for Loading pre-trained AutoEncoder Model: {args}')  
        encoder_model_file = os.path.join(args.save_directory, 'autoencoder_path_20.bin')
        
        autoencoder_model = ResNetAE(input_shape=(256, 256, 1),
                 n_ResidualBlock=2,
                 n_levels=6,
                 z_dim=192,
                 bottleneck_dim=192*4,
                 bUseMultiResSkips=True)
        
        print(f'Default CTOR Encoder Model:')
        PrintModel(model=autoencoder_model)

        state_dict = torch.load(encoder_model_file, map_location='cpu')
        autoencoder_model.load_state_dict(state_dict=state_dict)
        
        
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")} AutoEncoder Model loaded from file: {encoder_model_file}')
        
        PrintModel(model=autoencoder_model)

        for label in diseases: 
            for hidden_layer in mlp_layers:
                for optimizer in optimizers:
                    for learning_rate in learning_rates:
                        args.init_lr = learning_rate
                        args.num_train_epochs_classifier = training_epochs_classifier
                        args.disease_label = label
                        args.optimizer = optimizer
                        
                        print(f"Args for Classifier training: hidden layers={hidden_layer}, args= {args}")
                        
                        train_image_classifier(pre_trained_img_model = autoencoder_model, isMultiModal= isMultiModal,mlp_hidden_layers=hidden_layer, args=args, device=device)

train_Classifier(isMultiModal=False)
#train_MI_models()
#train_AE_models()

