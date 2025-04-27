import math
import numpy as np
from mutual_info_img_txt.main_utils import Plot_Training
from mutual_info_img_txt.model import Basic_MLP, build_resnet_model
from mutual_info_img_txt.model_utils import CXRImageDiseaseDataset
from mutual_info_img_txt.utils import PrintModel
import torch
import os
from helpers import construct_training_parameters, get_transform_function
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


def z_scored_transform(source_tensor):
	
	scaler = StandardScaler()
	scaler.fit(source_tensor)
	result_transform = scaler.transform(source_tensor)
	return result_transform

def mean_intra_class_distance(items):
	intra_class_tensor = pairwise_distances(items, metric='euclidean',n_jobs=10)
	total_items = items.shape[0]*items.shape[1]
	divide_factor = total_items * (total_items-1)
	return np.sum(intra_class_tensor) *2 / divide_factor

def mean_inter_class_distance(source,dest):
	inter_class_tensor = pairwise_distances(source,dest,metric='euclidean',n_jobs=10)
	total_items_source = source.shape[0]*source.shape[1]
	total_items_dest = dest.shape[0]*dest.shape[1]
	divide_factor = total_items_source * total_items_dest
	return np.sum(inter_class_tensor) / divide_factor


def gdv_calculation(positive_embeddings,negative_embeddings):
    print('gdv_calculation start')
    positive_class_transform = z_scored_transform(source_tensor=positive_embeddings)
    negative_class_transform = z_scored_transform(source_tensor=negative_embeddings)

    positive_intra_class = mean_intra_class_distance(positive_class_transform)
    negative_intra_class = mean_intra_class_distance(negative_class_transform)

    inter_class = mean_inter_class_distance(source=positive_class_transform, dest=negative_class_transform)

    dimension_invariance= 1/math.sqrt(len(positive_embeddings)+len(negative_embeddings))
    generalised_discrimination_value = dimension_invariance*((positive_intra_class + negative_intra_class)/2 - inter_class)
    return generalised_discrimination_value





args =  construct_training_parameters()

device = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")


critics=['dv']     #['dv','infonce']
training_epochs=[20]   #[5,10,20]

def validate_MultiModal(mlp_layer,disease):	
   
    #Note: run function parsing_csv_meta_data_for_label_stats in dataset_populate.py to generate disease_stats.csv for testing_data before construct DataLoader
    dataset = CXRImageDiseaseDataset(img_dir=args.image_dir, 
									dataset_metadata='testing_data/testing_label_negbio.csv', 
									disease=disease,
									disease_stats='testing_data/disease_stats.csv',
									transform=get_transform_function(args.img_size))
	
    dataLoader = DataLoader(dataset, batch_size=args.batch_size,
								 shuffle=True, num_workers=args.data_loader_workers,
								 pin_memory=True, drop_last=True)

   

    for critic in critics:
        for epoch in training_epochs:
            args.mi_estimator = critic
            args.num_train_epochs= epoch

            args.save_directory = os.path.join(args.save_directory,
                                f'{args.mi_estimator}_epoch{args.num_train_epochs}')
        
            print(f'Args for Loading pre-trained MI Image Model: {args}')  
            output_model_file = os.path.join(args.save_directory, 'pytorch_MI_image_model.bin')
            
            mi_image_model = build_resnet_model(model_name=args.image_model_name, checkpoint_path=output_model_file,
                                                            output_channels=args.output_channels)	
            print(f'Load MI model from file: {output_model_file}')
            PrintModel(mi_image_model)
            
            #classifier_model = Basic_MLP(input_dim=768, hidden_dims= mlp_layer)
            output_classifier_file = os.path.join(args.save_directory,'pytorch_image_classifier_Pneumonia_epoch7.bin')
            classifier_model = Basic_MLP.load_from_pretrained(input_dim=768, hidden_dims= mlp_layer, pretrained_model_path= output_classifier_file)
            print(f'Load Classifier model from file: {output_classifier_file}')
            PrintModel(classifier_model)

            mi_image_model.to(device)
            classifier_model.to(device)

            test_data_iterators=tqdm(dataLoader, desc='Accuracy calculation Iterations in Test Dataset')
            
            total_test = len(dataLoader)*args.batch_size
            test_count=0

            positive_embeddings=[]
            negative_embeddings=[]

            for batch_id, batch in enumerate(test_data_iterators, 0):
                image, label = batch
                image= image.to(device)
                output_image = mi_image_model.forward(image)
                image_embeddings=output_image[1]
                
                #Note: group embeddings to label positive or negative 
                embeddings= image_embeddings.cpu().detach().numpy()
                labels= label.cpu().detach().numpy()
                for i in range(len(labels)):
                    if(label[i] == 1):
                        positive_embeddings.append(embeddings[i])
                    else:
                        negative_embeddings.append(embeddings[i])

                # if(batch_id < 5):
                #     print(f'embeddings: {len(embeddings)}')
                #     print(f'labels: {labels}')
                #     print(f'positive_embeddings: {len(positive_embeddings)}')
                #     print(f'negative_embeddings: {len(negative_embeddings)}')

                image_embeddings= image_embeddings.to(device)

                output_result = classifier_model(image_embeddings)
                predictedLabel = torch.flatten(output_result).cpu().detach().numpy().round()

                label = label.cpu().detach().numpy()

                test_count = test_count + np.sum(predictedLabel == label).item()

            accuracy = test_count / total_test

            print(f'Accuracy: {accuracy}')

            print(f'positive_embeddings: {len(positive_embeddings)}')
            print(f'negative_embeddings: {len(negative_embeddings)}')
            
            separability = gdv_calculation(positive_embeddings,negative_embeddings)

           
            print(f'Separability: {separability}')
            
            return accuracy, separability

# mlp_layer=[1024,512,256,128,64,32,16]
# disease='Pneumonia'
# accuracy, separability = validate_MultiModal(mlp_layer=mlp_layer, disease=disease)

# disease='Cardiomegaly'

title = 'Training stats for UniModal '
loss_output_file= os.path.join(args.save_directory, 'um_ae_epoch20','autoencoder_training_loss_epoch20.png')
training_epoch_loss= [5.33208, 2.95880, 2.53993, 2.33715, 2.22067, 2.17062, 2.13688, 2.08275, 2.05269, 2.02723, 2.01354, 1.99350, 1.99693, 1.97989, 1.96648, 1.94977, 1.92462, 1.92096, 1.90587, 1.88781 ]
#validation_epoch_loss=[0.5171646946354916, 0.5133194366568014, 0.5084891734938872, 0.511075593138996, 0.5186354957128826, 0.5134522695290414, 0.5100842596668946, 0.5170770726705852, 0.5135867925066697, 0.5203410096858677, 0.511287551961447, 0.515552996020568, 0.5039283005814803, 0.5058009608795768, 0.5082380050107053, 0.5032256896558561, 0.5110381112286919, 0.5024153574516899, 0.5085309900735554, 0.5104486048221588, 0.5086603643078553, 0.5009645819664001, 0.5135763107161773, 0.5063072592020035, 0.5108719750454551, 0.5097643513428537, 0.5054828736342882, 0.5006989813164661, 0.5060766519684541, 0.508492114512544, 0.5130916175089384, 0.49983010088142593, 0.4999388895536724, 0.5083453804254532, 0.5053121925968873, 0.5057004815653751, 0.5105348205880115, 0.5063045675817289, 0.5117278899017134, 0.5020675761135001, 0.5012340757407641, 0.5024471769207403, 0.505302010398162, 0.512509338949856, 0.5075595214178688, 0.5115708144087541, 0.5098714287343779, 0.5096206076835331, 0.5097879422338385, 0.5076601034716556, 0.5141130999514931, 0.5123301971899835, 0.50803200586846, 0.5122623404389933, 0.49856186069940267, 0.5166235926904177, 0.5090262301658329, 0.513514076408587, 0.5081978738307953, 0.5128207340052253, 0.5027274160008681, 0.5085740034517489, 0.5029723706998324, 0.5069229344004079, 0.5045629096658606, 0.5142150419323068, 0.5034713156913456, 0.5080432750676808, 0.5079107802165183, 0.5052422402720702, 0.5044430996242323, 0.5060419612809232, 0.5161459736133877, 0.5107580470411401, 0.5044760837366706, 0.5091029099727932, 0.5097235794130125, 0.5075653532617971, 0.5060611351540214, 0.5188472451348054, 0.5106968879699707, 0.5064111320595992, 0.5030726437505922, 0.5059284552147514, 0.513903757459239, 0.5001846183287469, 0.5114736313882627, 0.49882908792872177, 0.49200070531744705, 0.5121662875539378, 0.500540587462877, 0.5071800672694257, 0.5042103207425067, 0.5158453677829943, 0.512466582812761, 0.5111852178448125, 0.5051352718943044, 0.4964552796200702, 0.502977969615083, 0.5045459780253863]
Plot_Training(xlabel='Epochs',ylabel='Value for Loss', title=title,data=[training_epoch_loss],dataLabel=['Training loss'],out_imgage_file=loss_output_file)


# accuracy_title = 'Training stats for disease ' + disease+'\n batch_size= ' + str(args.batch_size)
# accuracy_output_file= os.path.join(args.save_directory, 'test_image_classifier_training_accuracy_'+disease+'.png')
# validation_epoch_accuracy=[0.759046052631579, 0.7701480263157895, 0.7685032894736842, 0.7602796052631579, 0.7545230263157895, 0.764391447368421, 0.7602796052631579, 0.7680921052631579, 0.7598684210526315, 0.765625, 0.7689144736842105, 0.7652138157894737, 0.7676809210526315, 0.766858552631579, 0.7606907894736842, 0.7648026315789473, 0.7648026315789473, 0.7648026315789473, 0.766858552631579, 0.7598684210526315, 0.7664473684210527, 0.7713815789473685, 0.7689144736842105, 0.7631578947368421, 0.7639802631578947, 0.7623355263157895, 0.772203947368421, 0.772203947368421, 0.7648026315789473, 0.7709703947368421, 0.7553453947368421, 0.7763157894736842, 0.759046052631579, 0.7619243421052632, 0.7742598684210527, 0.7631578947368421, 0.7631578947368421, 0.7631578947368421, 0.7652138157894737, 0.765625, 0.7660361842105263, 0.7648026315789473, 0.7693256578947368, 0.7594572368421053, 0.766858552631579, 0.7611019736842105, 0.7693256578947368, 0.7680921052631579, 0.7689144736842105, 0.7623355263157895, 0.7697368421052632, 0.7623355263157895, 0.7619243421052632, 0.7615131578947368, 0.7693256578947368, 0.7602796052631579, 0.765625, 0.764391447368421, 0.7652138157894737, 0.7676809210526315, 0.7693256578947368, 0.7635690789473685, 0.7730263157894737, 0.7623355263157895, 0.7693256578947368, 0.7648026315789473, 0.7623355263157895, 0.7676809210526315, 0.7672697368421053, 0.7553453947368421, 0.7594572368421053, 0.7606907894736842, 0.764391447368421, 0.7689144736842105, 0.7611019736842105, 0.7693256578947368, 0.7631578947368421, 0.7697368421052632, 0.7713815789473685, 0.7586348684210527, 0.7680921052631579, 0.7615131578947368, 0.7672697368421053, 0.7619243421052632, 0.7615131578947368, 0.7672697368421053, 0.7606907894736842, 0.7709703947368421, 0.7705592105263158, 0.7561677631578947, 0.7705592105263158, 0.7639802631578947, 0.7660361842105263, 0.7648026315789473, 0.7664473684210527, 0.7685032894736842, 0.7631578947368421, 0.7734375, 0.7697368421052632, 0.7705592105263158]
# Plot_Training(xlabel='Epochs',ylabel='Value for Accuracy', title=accuracy_title,data=[validation_epoch_accuracy],dataLabel=['validation accuracy'],out_imgage_file=accuracy_output_file)

