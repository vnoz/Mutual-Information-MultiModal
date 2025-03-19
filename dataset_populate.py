import os
import argparse
import logging
import json
import requests
import csv
import sys
import gzip
from pathlib import Path
from os import listdir
from os.path import isfile, join
import datetime
import random
import numpy as np
from sklearn.metrics import confusion_matrix

from helpers import construct_dataset_parameters

import cv2
import numpy as np
from pytorch_transformers import BertTokenizer
import torch
from mutual_info_img_txt import model_utils
from helpers import construct_training_parameters

args = construct_dataset_parameters()

#print(f"Dataset_populate: args: {args}")


def get_filename_url(base, file, save_location):
    wget_cmd = 'wget -r -o -N -c -np -nH --cut-dirs 10 --user '+ args.download_user + ' --password '+ args.download_password + ' '
    host_url=  os.path.join('https://physionet.org/files/',base)
    return wget_cmd + host_url + file + ' -P ' + save_location 

def process_file(base, filename):
    url = get_filename_url(base, filename, args.data_dir)
    execute_command(url)

    with open(os.path.join(args.data_dir,filename),"rt") as f:
        findings_content=[]
        start_getting_content=False
        
        content_without_findings_keyword=[]
        new_line_for_findings_content = False

        for line in f:
            if('FINDINGS:' in line.strip()):
                findings_content.append(line.strip())
                start_getting_content = True
                continue
            elif('IMPRESSION:' in line.strip() and start_getting_content==True):
                start_getting_content = False
                break

            if(start_getting_content == True and line.strip() != ''):
                findings_content.append(line.strip())
            
            if(line.strip() == ''):
                new_line_for_findings_content = True
                content_without_findings_keyword = []
                
            elif(new_line_for_findings_content == True and 'FINDINGS:' not in line.strip() and 'IMPRESSION:' not in line.strip()):
                content_without_findings_keyword.append(line.strip())

        if(len(findings_content)==0 and len(content_without_findings_keyword) > 0):
            findings_content = content_without_findings_keyword
        
        print(findings_content)

def get_filename_new_location_url(base,file, new_filename):
    wget_cmd = 'wget -r -o -N -c -np -nH --cut-dirs 10 --user '+ args.download_user + ' --password '+ args.download_password + ' '
    host_url=  os.path.join('https://physionet.org/files/',base)
    return wget_cmd + host_url + file + ' -O '+new_filename

def execute_command(cmd):
    os.system(cmd)

#study_dictionary={}
#image_file_dictionary={}

 # Download mimic-cxr-2.0.0-metadata.csv.gz from MIMIC-CXR JPG for all files metadata
meta_filename = 'mimic-cxr-2.0.0-metadata.csv.gz'   #metadata for mapping between image and associate text file
label_filename = 'mimic-cxr-2.0.0-negbio.csv.gz'  #mapping of study and labels from 14 diseases

def create_data_folder():
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    if not os.path.exists(args.image_storage_dir):
        os.makedirs(args.image_storage_dir)
    if not os.path.exists(args.text_storage_dir):
        os.makedirs(args.text_storage_dir)

    sub_folders = ['p10', 'p11','p12', 'p13','p14', 'p15','p16', 'p17','p18', 'p19']
    for id, subfolder in enumerate(sub_folders):
        img_sub_folder = os.path.join(args.image_storage_dir,subfolder)
        txt_sub_folder = os.path.join(args.text_storage_dir,subfolder)
        
        if not os.path.exists(img_sub_folder):
            os.makedirs(img_sub_folder)
        if not os.path.exists(txt_sub_folder):
            os.makedirs(txt_sub_folder)

    if not os.path.exists(args.training_data_dir):
        os.makedirs(args.training_data_dir)
    if not os.path.exists(args.training_image_dir):
        os.makedirs(args.training_image_dir)
    if not os.path.exists(args.training_text_dir):
        os.makedirs(args.training_text_dir)

    if not os.path.exists(args.testing_data_dir):
        os.makedirs(args.testing_data_dir)
    if not os.path.exists(args.testing_image_dir):
        os.makedirs(args.testing_image_dir)
    if not os.path.exists(args.testing_text_dir):
        os.makedirs(args.testing_text_dir)
create_data_folder()
def download_full_dataset(subfolder,download_from_mimic=True):

    create_data_folder()

    jpg_cxr_base_url = 'mimic-cxr-jpg/2.1.0/'
    mimic_cxr_base_url = 'mimic-cxr/2.1.0/'

    if not os.path.exists(os.path.join(args.data_dir,meta_filename)):
        metadata_file_url = get_filename_url(jpg_cxr_base_url,meta_filename,args.data_dir)
        print('Start downloading meta data file' + metadata_file_url)
        execute_command(metadata_file_url)

    if not os.path.exists(os.path.join(args.data_dir,label_filename)):
        label_file_url = get_filename_url(jpg_cxr_base_url,label_filename,args.data_dir)
        print('Start downloading label file' + label_file_url)
        execute_command(label_file_url)


    # Read content of meta data file, and loop through each item 
    #     to download images from MIMIC-CXR-JPG and free-text report from MIMIC-CXR
   
    count = 0

    contents_list=[]
    study_list=[]
    img_path_list=[]
    subject_id_list=[]
    
    previous_study_id=''

    # Download cxr-jpg file and create associate free-text file list
    print('Download free-text report, parse the content for findings and impression and download CXR JPEG image in subfolder: ' + subfolder)
    print('start time: ' + str(datetime.datetime.now()))
    #with open(os.path.join(args.data_dir,'mimic-cxr-2.0.0-metadata.csv'), "rt") as f:
    new_meta_file_for_subfolder = os.path.join(args.text_storage_dir,subfolder,meta_filename)
    if(os.path.isfile(new_meta_file_for_subfolder) == False):
        copy_cmd = 'cp ' + os.path.join(args.data_dir,meta_filename) + ' ' + new_meta_file_for_subfolder
               
        execute_command(copy_cmd)
    showLog=False
    with gzip.open(new_meta_file_for_subfolder, "rt") as f:
            
            for line in f:
                #ignore first line for column labels
                if count == 0:
                    count = count+1
                    continue

                split_items = line.split(',')

                dicom_id= split_items[0]
                subject_id = split_items[1]
                study_id = split_items[2]
                view_position = split_items[4]
                if((view_position == 'PA' or view_position == 'AP' ) and subfolder =='p'+subject_id[:2] 
                        and study_id not in study_list and study_id != previous_study_id):
                    previous_study_id = study_id
                    src_img_filename = os.path.join('files',subfolder,'p'+subject_id,'s'+ study_id, dicom_id+'.jpg')
                    new_img_filename_without_extension = 'p'+subject_id+'_'+'s'+ study_id+'_'+dicom_id
                    
                    text_filename = os.path.join('files',subfolder,'p'+subject_id,'s'+ study_id+'.txt')
                    
                    # Downloading text file in MIMIC-CXR
                    current_file_sub_folder = os.path.join(args.text_storage_dir,subfolder)

                    # check file exist before start downloading and parsing
                    
                    if(os.path.isfile(os.path.join(args.text_storage_dir,subfolder,'s'+ study_id+'.txt'))):
                        if(showLog == False):
                            print('file already existed: ' + study_id)
                            showLog = True

                        continue

                    print('continue to download from previous checking at time: ' + str(datetime.datetime.now()) + ', study_id=' + study_id)
                    
                    text_file_url = get_filename_url(mimic_cxr_base_url,text_filename,current_file_sub_folder)
                    
                    execute_command(text_file_url)

                    # Finish Downloading text file in MIMIC-CXR

                    # Parsing text file and construct content from FINDINGS keyword in free-text report
                    
                    findings_content=[]
                    has_findings_keyword=False

                    impression_content=[]
                    has_impression_keyword=False

                    second_last_paragraph_before_new_line=[]
                    last_paragraph_before_new_line=[]

                    new_line_for_findings_content = False

                    extracted_content=[]

                    with open(os.path.join(current_file_sub_folder,'s'+ study_id+'.txt'),"rt") as f:
                        # NOTE: if content has Findings keyword, then return the text between Findings and Impression keywords, 
                        #           otherwise return the text after the last empty line break
                        for line in f:
                            line_content= line.strip()
                            if('FINDINGS:' in line_content):
                                if(line_content != 'FINDINGS:' and line_content.startswith('FINDINGS:')):
                                    findings_content.append(line_content[line_content.index('FINDINGS:')+9:].strip())
                                    
                                has_findings_keyword = True
                                continue
                            elif('IMPRESSION:' in line_content):
                                if(line_content != 'IMPRESSION:' and line_content.startswith('IMPRESSION:')):
                                    impression_content.append(line_content[line_content.index('IMPRESSION:')+11:].strip())
                                
                                has_impression_keyword = True
                                continue
                        
                          
                            if(line_content != ''):   
                                if(line_content[0].istitle and ':' in line_content):
                                    hyphen_index = line_content.index(':') 
                                    line_content = line_content[hyphen_index+1:].strip()

                                if(has_findings_keyword == True):
                                    findings_content.append(line_content)
                                elif(has_impression_keyword == True):
                                    impression_content.append(line_content)
                                elif(new_line_for_findings_content == True):
                                    last_paragraph_before_new_line.append(line_content)
                            elif(line_content == ''):
                                new_line_for_findings_content = True
                                if(has_impression_keyword == False):
                                    second_last_paragraph_before_new_line = last_paragraph_before_new_line
                                    last_paragraph_before_new_line = []

                        if(len(findings_content)> 0):
                            extracted_content = findings_content
                        elif(len(last_paragraph_before_new_line) > 0):
                            extracted_content = last_paragraph_before_new_line
                        else: 
                            extracted_content = second_last_paragraph_before_new_line

                        if(len(impression_content)> 0):
                            extracted_content +=impression_content

                    if(len(extracted_content) > 0):
                        contents_list.append(' '.join(map(str,extracted_content)))
                        
                        study_list.append(study_id)

                        img_path_list.append(new_img_filename_without_extension)
                        subject_id_list.append(subject_id)

                        # Only download image file when Text file has Findings_Content
                        if(download_from_mimic == True):
                            dest_img_filename_local_path = os.path.join(args.image_storage_dir,'p'+subject_id[:2],new_img_filename_without_extension+'.jpg')  
                    
                            download_img_url = get_filename_new_location_url(jpg_cxr_base_url,src_img_filename,dest_img_filename_local_path)
                            execute_command(download_img_url)

                        if(count %10 == 0):
                            data_fileName = 'data_subfolder_'+subfolder+'.tsv'
                            print('Write pairs of studyID and Findings to file: ' + data_fileName + ', attempt='+str(count/10))
                            with open(os.path.join(args.text_storage_dir,data_fileName), 'a', encoding='utf8', newline='') as tsv_file:
                                tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
                                
                                for i in range(len(contents_list)):
                                    tsv_writer.writerow([count - 10 +i,subject_id_list[i], study_list[i],img_path_list[i],contents_list[i]])

                            study_list=[]
                            contents_list=[]
                            img_path_list = []
                            subject_id_list = []

                        count = count+1

                    # Finish Parsing text file
                   
                    # if(count > imgAmount):
                    #     break
    
    print('complete time: ' + str(datetime.datetime.now()))
   
#Note: main execution entrance
# if __name__ == '__main__':
#     subfolder = sys.argv[2]   
   
#     download_full_dataset(subfolder,download_from_mimic=True) 
#download_full_dataset(20,subfolder='p10',download_from_mimic=True)


def populate_subset_dataset():
    current_study_count=0
    training_contents_list={}
    testing_contents_list={}
    training_study_dictionary = {}
    testing_study_dictionary = {}
    

    study_list=[]
    sourceFile = 'all_data.tsv'
    
    
    with open(os.path.join(args.training_text_dir,sourceFile), "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", lineterminator='\n')
            
            for line in reader:
                study_list.append(line[2])
                
            print('all_data total studies: ' + str(len(study_list)))


    # Open mimic-cxr2.0.0-metadata.csv.gz to construct study_dictionary
   
    with gzip.open(os.path.join(args.data_dir,meta_filename), "rt") as f:   
            count = 0
            for line in f:
                #ignore first line for column labels
                if count == 0:
                    count = count+1
                    continue

                split_items = line.split(',')

                dicom_id= split_items[0]
                subject_id = split_items[1]
                study_id = split_items[2]
                
                img_filename = os.path.join('p'+subject_id[:2],'p'+subject_id+'_s'+ study_id+'_'+dicom_id+'.jpg')
                if(study_id == '51513702'):
                    print(img_filename)

                if(os.path.isfile(os.path.join(args.image_storage_dir, img_filename)) and study_id in study_list):
                    count = count + 1
                    if(random.randrange(0,20) == 10):
                        testing_study_dictionary[study_id] = img_filename
                    else:
                        training_study_dictionary[study_id] = img_filename

                # if(count > amount):
                #     break
            print('Total file already downloaded: ' + str(count))
            print('testing_study_dictionary: ' + str(len(testing_study_dictionary)))
            print('training_study_dictionary: ' + str(len(training_study_dictionary)))

    # Move file from full dataset to training dataset folder
    with open(os.path.join(args.text_storage_dir,'all_data.tsv'), "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", lineterminator='\n')
        
        for line in reader:
           
            text = line[-1]
            # labels = line[1]
            study_id = line[2]
            
            if(text != ''):
                current_study_count=current_study_count+1
                
                if(study_id in testing_study_dictionary):
                    testing_contents_list[study_id] = text
                elif(study_id in training_study_dictionary):
                    training_contents_list[study_id]=text

    
    # Write FINDINGS in free text reports of training images to training_data.tsv
    training_data_file = os.path.join(args.training_text_dir,'training_data.tsv')
    testing_data_file = os.path.join(args.testing_text_dir,'testing_data.tsv')
    
   
    with open(training_data_file, 'w', encoding='utf8', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        
        i=0
       
        for study_id in training_contents_list:
            tsv_writer.writerow([i,0, study_id,training_study_dictionary.get(study_id,''),training_contents_list[study_id]])
            i=i+1

        print('Write to training_data.tsv, line count='+ str(i))

    with open(testing_data_file, 'w', encoding='utf8', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        
        i=0
       
        for study_id in testing_contents_list:
            tsv_writer.writerow([i,0, study_id,testing_study_dictionary.get(study_id,''),testing_contents_list[study_id]])
            i=i+1

        print('Write to testing_data.tsv, line count='+ str(i))

    # reset contents dictionary before creating another dictionary/list for label data
    training_contents_list={}
    testing_contents_list={}

    # Read label_filename and find study_id to add into args.training_dataset_metadata
    training_label_report_lines=[]
    testing_label_report_lines=[]
    line_count = 0
    with gzip.open(os.path.join(args.data_dir,label_filename), "rt") as f:
            for line in f:
                # if(line_count > amount):
                #     break
                if (line_count == 0):
                    new_line=[]
                    new_line.append('mimic_id')
                    new_line.extend(line.strip('\n').split(',')[2:])
                    training_label_report_lines.append(new_line)
                    testing_label_report_lines.append(new_line)
                    line_count = line_count + 1

                else:
                    current_study_id = line.split(',')[1]
                    #if (training_contents_list.get(current_study_id,'') != ''):
                   
                    if (current_study_id in training_study_dictionary):  
                        new_line=[]
                        new_line.append(training_study_dictionary.get(current_study_id,'') )
                        new_line.extend(line.strip('\n').split(',')[2:])
                        training_label_report_lines.append(new_line)                
                        line_count = line_count + 1
                       
                    elif (current_study_id in testing_study_dictionary): 
                        new_line=[]
                        new_line.append(testing_study_dictionary.get(current_study_id,'') )
                        new_line.extend(line.strip('\n').split(',')[2:])
                        testing_label_report_lines.append(new_line)                
                        line_count = line_count + 1
                        

    # reset study dictionary after use
    testing_study_dictionary = {}
    training_study_dictionary = {}

    #note: write to file training_label_negbio.csv for training and testing
    with open(args.training_dataset_labeldata, 'w') as csv_file_train:
        tsv_writer = csv.writer(csv_file_train)
        tsv_writer.writerows(training_label_report_lines)

    print('Write to training_label_net_bio.tsv, line count='+ str(len(training_label_report_lines)))
    
    with open(args.testing_dataset_labeldata, 'w') as csv_file_test:
        tsv_writer_test = csv.writer(csv_file_test)
        tsv_writer_test.writerows(testing_label_report_lines)

    print('Write to testing_label_net_bio.tsv, line count='+ str(len(testing_label_report_lines)))

    parsing_csv_meta_data_for_label_stats(args.training_dataset_labeldata, args.training_data_dir)

def parsing_csv_meta_data_for_label_stats(metadata, data_dir):
    result = {}
    labels=[]
    print(metadata)
    with open(metadata, 'rt') as csvfile:
        csvreader = csv.reader(csvfile, lineterminator='\n')
        line_count=0
        total_disease_count=0
        for row in csvreader:
            if(line_count==0):
                labels=row[1:]
                for label in labels:
                    result[label]=[]
                    result[label+'_negative']=[]
                
                print('labels:')
                print(labels)  
            
            line_count =line_count + 1
            mimic_id = row[0]
            for idx,x in enumerate(row[1:]):
                hasValue = False
                if(x == '1.0'):
                    result[labels[idx]].append(mimic_id.split('_')[1][1:])
                    hasValue = True
                elif (x == '0.0'):
                    result[labels[idx]+'_negative'].append(mimic_id.split('_')[1][1:])
                    hasValue = True
                
                if(hasValue == True and idx == len(labels)-1):
                    total_disease_count = total_disease_count + 1


    print('lines count')
    print(line_count)
    print('total_disease_count')
    print(total_disease_count)
    # print('result dict')
    # print(result)

    with open(os.path.join(data_dir,'disease_stats.csv'), 'w') as tsv_file:
        tsv_writer = csv.writer(tsv_file)
        for label in labels:
            line=[]
            line.append(label)
            line.append(len(result[label]))
            line.append(result[label])
            tsv_writer.writerow(line)

            line=[]
            line.append(label+'_negative')
            line.append(len(result[label+'_negative']))
            line.append(result[label+'_negative'])

            tsv_writer.writerow(line)

def fix_missing_studies_for_training_label_negbio_from_download():
    study_list=[]
    sourceFile = 'all_data.tsv'
    missing_studies=[]
    
    with open(os.path.join(args.training_text_dir,sourceFile), "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", lineterminator='\n')
            
            for line in reader:
                study_id = line[2]
                
                study_list.append(study_id)
                
            print('all_data total studies: ' + str(len(study_list)))

    print(args.training_dataset_labeldata)

    with open(args.training_dataset_labeldata, 'rt') as f:
        
    # with open(args.training_dataset_labeldata, "r", encoding="utf-8") as f:
    #     reader = csv.reader(f, delimiter="\t", lineterminator='\n')
        line_count = -1
        for line in f:
            line_count = line_count+1
            if line_count == 0:
                continue

            split_items = line.split(',')

            file_id= (split_items[0]).split('/')[1]
            study_id = file_id.split('_')[1][1:]
            
            if(line_count %10000 ==1):
                print(line)
                print(file_id)
                print(study_id)
          
            # 
            
            if(study_id not in study_list):
                print(study_id)
                missing_studies.append(study_id)
               
            # if(study_id == '59969148'):
            #     print(line[-1])
        print('total lines: ' + str(line_count))
    print('total mismatched studies: '+ str(len(missing_studies)))

def fix_missing_studies_for_all_data_from_download():
    #read full_data_set\all_data.tsv and store study_id in dictionary
    # read all tsv file for subfolders from p0 to p10 to find out any missing study_id and add into full_data_set\add_data.tsv
    study_list=[]
    contents_list=[]

    missing_studies=[]

    subfolder = 'p19'
    filename='data_subfolder_'+subfolder+ '.tsv'

    sourceFile = 'all_data.tsv'
    with open(os.path.join(args.text_storage_dir,sourceFile), "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", lineterminator='\n')
        
        for line in reader:
            study_id = line[2]
            
            study_list.append(study_id)
            
        # print('all_data_github total studies: ' + str(len(study_list)))

    with open(os.path.join(args.text_storage_dir,filename), "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", lineterminator='\n')
        
        for line in reader:
            study_id = line[2]
            if(study_id not in study_list):
                print(study_id)
                missing_studies.append(study_id)
                contents_list.append(line[-1])

            # if(study_id == '59969148'):
            #     print(line[-1])

    print('total missing studies in '+ subfolder +': '+ str(len(missing_studies)))

    with open(os.path.join(args.text_storage_dir,sourceFile), 'a', encoding='utf8', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        
        for i in range(len(contents_list)):
            tsv_writer.writerow([i,-1, missing_studies[i],subfolder,contents_list[i]])


def populate_all_data_from_subfolder_download():
    study_list=[]
    contents_list=[]

    subfolder = 'p19'
    filename='data_subfolder_'+subfolder+ '.tsv'

    sourceFile = 'all_data.tsv'
    file_count=0
    print(subfolder)
    with open(os.path.join(args.text_storage_dir,filename), "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", lineterminator='\n')
        
        for line in reader:
            file_count = file_count+1
            study_list.append(line[2])
            contents_list.append(line[-1])

            if(file_count % 5000 == 0):
                with open(os.path.join(args.text_storage_dir,sourceFile), 'a', encoding='utf8', newline='') as tsv_file:
                    tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
                    shift_index = int(file_count/5000) -1
                    print(shift_index)
                    for i in range(len(contents_list)):
                        tsv_writer.writerow([i + 1+ shift_index* 5000,-1, study_list[i],subfolder,contents_list[i]])
                
                study_list = []
                contents_list=[]
        print('File count: ' + str(file_count))
        #Note: write last batch of study_list and content_list to all_data
        with open(os.path.join(args.text_storage_dir,sourceFile), 'a', encoding='utf8', newline='') as tsv_file:
            tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
            shift_index = int(file_count/5000)
            for i in range(len(contents_list)):
                tsv_writer.writerow([i + 1+ shift_index* 5000,-1, study_list[i],subfolder,contents_list[i]])

#populate_all_data_from_subfolder_download()

#fix_missing_studies_for_training_label_negbio_from_download()

#fix_missing_studies_for_all_data_from_download()

#populate_subset_dataset()

def test_image_loader(studyId, imgId):
    args_training =  construct_training_parameters()
    print('test_image_loader')
    try:
        png_path = os.path.join(args_training.image_dir,imgId)
    
        # print('CXRImageReportDataset getItem: '+png_path)
        img = cv2.imread(png_path, cv2.IMREAD_ANYDEPTH)
        
        if(img is not None):
            
            img = np.expand_dims(img, axis=0)
        else:    
            print('Default image for study_id=' + str(studyId)+', img_id='+str(imgId))
            
            
    except Exception as e: 

        print('Inner Exception loading image for study_id ' + str(studyId)+', img_id='+str(imgId))
        print(repr(e))
#test_image_loader(52242635, 'p15/p15456778_s52242635_2eea1657-c608033f-5ca7f42b-f028fde4-afbae706.jpg')    

def test_dataloader():
    print('test_loader start')
   
    args_training =  construct_training_parameters()
    
    print('Get Text token features') 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bert_pretrained_dir = os.path.join(current_dir, 'bert_pretrain_all_notes_150000')

    if not os.path.exists(bert_pretrained_dir):
        os.makedirs(bert_pretrained_dir)

    tokenizer = BertTokenizer.from_pretrained(bert_pretrained_dir)
    text_token_features = model_utils.load_and_cache_examples(args_training, tokenizer)
    print('tokens features: '+ str(len(text_token_features)))


    all_txt_tokens = {f.report_id: f.input_ids for f in text_token_features}
    all_txt_masks = {f.report_id: f.input_mask for f in text_token_features}
    all_txt_segments = {f.report_id: f.segment_ids for f in text_token_features}

    broken_image_list=[]

    #Note: read training_data.tsv and check all images
    training_data_file = os.path.join(args.training_text_dir,'training_data.tsv')

    count=0
    print('Open file: ' + training_data_file)

    with open(training_data_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", lineterminator='\n')
        
        for line in reader:
            count = count+ 1
            studyId = line[2]
            imgId = line[3]

            if(count % 5000 == 0):
                print('line count='+ str(count))
                print(studyId)
                print('----------')
            try:

                txt = all_txt_tokens[studyId]

                if(txt is not None):
                    txt = torch.tensor(txt, dtype=torch.long)
                else:
                    print('Default token for study_id=' + str(studyId))
                    if(studyId not in broken_image_list):
                        broken_image_list.append(studyId)
                
                txt_masks = all_txt_masks[studyId]
                if(txt_masks is not None):
                    txt_masks = torch.tensor(txt_masks, dtype=torch.long)
                else:
                    print('Default token masks for study_id=' + str(studyId))
                    if(studyId not in broken_image_list):
                        broken_image_list.append(studyId)

                txt_segments = all_txt_segments[studyId]
                if(txt_segments is not None):
                    txt_segments = torch.tensor(txt_segments, dtype=torch.long)
                else:
                    print('Default token segments for study_id=' + str(studyId))
                    if(studyId not in broken_image_list):
                        broken_image_list.append(studyId)

                img = None
                try:
                    png_path = os.path.join(args_training.image_dir,imgId)
                
                    # print('CXRImageReportDataset getItem: '+png_path)
                    img = cv2.imread(png_path, cv2.IMREAD_ANYDEPTH)
                    
                    if(img is not None):
                        
                        img = np.expand_dims(img, axis=0)
                    else:    
                        print('Default image for study_id=' + str(studyId)+', img_id='+str(imgId))
                        if(studyId not in broken_image_list):
                            broken_image_list.append(studyId)
                        
                except Exception as e: 

                    print('Inner Exception loading image for study_id ' + str(studyId)+', img_id='+str(imgId))
                    print(repr(e))
                    if(studyId not in broken_image_list):
                        broken_image_list.append(studyId)

            except Exception as e: 
                print('Outer Exception loading image for study_id ' + str(studyId)+', img_id='+str(imgId))
                print(repr(e))
                if(studyId not in broken_image_list):
                        broken_image_list.append(studyId)
    print('total image file count='+str(count))
    print('broken_image_list length='+ str(len(broken_image_list)))
    print(broken_image_list)
#test_dataloader()

#test_dataloader('59902543','p18/p18257244_s59902543_e354faf6-4a010a95-e3afeed8-f65d7894-1a87dc24.jpg')
def test_numpy_array():
    print('test_numpy_array')
    label=[1,1,1,1,0,0,0,0]
    predictedLabel=[1,1,1,1,0,1,0,1]
    count= np.sum(np.logical_and(predictedLabel == 1.0, label == 1.0))
    print(count)

    val_count=np.sum(predictedLabel == label)
    print('val_count= '+ str(val_count))

    # cm = confusion_matrix(label, predictedLabel)

    # #rows = gt, col = pred

    # #compute tp, tp_and_fn and tp_and_fp w.r.t all classes
    # tp_and_fn = cm.sum(1)
    # tp_and_fp = cm.sum(0)
    # tp = cm.diagonal()

    # precision = tp / tp_and_fp
    # recall = tp / tp_and_fn
    # print('cm= '+ str(cm))
    # print('tp= ' + str(tp))
    # print('precision= ' + str(precision))
    # print('recall= '+str(recall))

test_numpy_array()
