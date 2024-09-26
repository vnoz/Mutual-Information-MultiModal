import os
import argparse
import logging
import json
import requests
import csv
import sys
import gzip
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str,
                    default=os.path.join(current_dir, 'full_data_set/'),
                    help='The parent data directory')

parser.add_argument('--image_storage_dir', type=str,
                    default=os.path.join(current_dir, 'full_data_set/images/'),
                    help='The image data directory')

parser.add_argument('--text_storage_dir', type=str,
                    default=os.path.join(current_dir, 'full_data_set/text/'),
                    help='The text data directory')

parser.add_argument('--download_user', type=str,
                    default='tuanle',
                    help='The user to download MIMIC dataset')

parser.add_argument('--download_password', type=str,
                    default='A1thebest',
                    help='The password to download MIMIC dataset')

parser.add_argument('--total_amount', type=str,
                    default=2000,
                    help='Total amount of samples to download from MIMIC dataset')

parser.add_argument('--amount_for_training', type=str,
                    default=1000,
                    help='Total amount of samples for training')


parser.add_argument('--amount_for_testing', type=str,
                    default=100,
                    help='Total amount of samples for testing')

parser.add_argument('--training_data_dir', type=str,
                    default=os.path.join(current_dir, 'example_data/'),
                    help='The parent data directory')

parser.add_argument('--training_image_dir', type=str,
                    default=os.path.join(current_dir, 'example_data/images/'),
                    help='The training image data directory')
parser.add_argument('--training_text_dir', type=str,
                    default=os.path.join(current_dir, 'example_data/text/'),
                    help='The training text data directory')
parser.add_argument('--training_dataset_metadata', type=str,
                    default=os.path.join(current_dir, 'example_data/training_text_label_negbio.csv'),
                    help='The metadata for the model training ')


args = parser.parse_args()

print(f"Initial args: {args}")


def get_filename_url(base, file, save_location):
    wget_cmd = 'wget -r -N -c -np -nH --cut-dirs 10 --user '+ args.download_user + ' --password '+ args.download_password + ' '
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
    wget_cmd = 'wget -r -N -c -np -nH --cut-dirs 10 --user '+ args.download_user + ' --password '+ args.download_password + ' '
    host_url=  os.path.join('https://physionet.org/files/',base)
    return wget_cmd + host_url + file + ' -O '+new_filename

def execute_command(cmd):
    os.system(cmd)

study_dictionary={}
image_file_dictionary={}
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

    if not os.path.exists(args.training_data_dir):
        os.makedirs(args.training_data_dir)
    if not os.path.exists(args.training_image_dir):
        os.makedirs(args.training_image_dir)
    if not os.path.exists(args.training_text_dir):
        os.makedirs(args.training_text_dir)

def download_full_dataset(imgAmount):

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
    text_files = []
    
    count = 0

    # Download cxr-jpg file and create associate free-text file list
    with gzip.open(os.path.join(args.data_dir,meta_filename), "rt") as f:
            
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
                if(view_position == 'PA'):

                    img_filename = os.path.join('files','p'+subject_id[:2],'p'+subject_id,'s'+ study_id, dicom_id+'.jpg')
                    new_img_filename_without_extension = 'p'+subject_id+'_'+'s'+ study_id+'_'+dicom_id
                    
                    new_img_filename_full_path = os.path.join(args.image_storage_dir,new_img_filename_without_extension+'.jpg')  
                    study_dictionary[study_id] =  new_img_filename_full_path
                    image_file_dictionary[study_id] = new_img_filename_without_extension
                    img_url = get_filename_new_location_url(jpg_cxr_base_url,img_filename,new_img_filename_full_path)
                    execute_command(img_url)
                    
                    text_filename = os.path.join('files','p'+subject_id[:2],'p'+subject_id,'s'+ study_id+'.txt')
                    text_files.append(text_filename)
                    
                    count = count+1

                    if(count > imgAmount):
                        break
    
    # Downloading text files in MIMIC-CXR
    for i in range(len(text_files)):
        text_file_url = get_filename_url(mimic_cxr_base_url,text_files[i],args.text_storage_dir)
        execute_command(text_file_url)


    contents_list=[]
    study_list=[]

    # Parse text file and construct content from FINDINGS keyword in free-text report
    for i in range(len(text_files)):
        findings_content=[]
        start_getting_content=False
        
        single_text_file =text_files[i].split('/')[-1]

        content_without_findings_keyword=[]
        new_line_for_findings_content = False

        # NOTE: if content has Findings keyword, then return the text between Findings and Impression keywords, 
        #           otherwise return the text after the last empty line break
        with open(os.path.join(args.text_storage_dir,single_text_file),"rt") as f:
            for line in f:
                line_content= line.strip()
                if('FINDINGS:' in line_content):
                    if(line_content != 'FINDINGS:' and line_content.startswith('FINDINGS:')):
                        findings_content.append(line_content[line_content.index('FINDINGS:')+9:].strip())
                        
                    start_getting_content = True
                    continue
                elif('IMPRESSION:' in line_content and start_getting_content==True):
                    start_getting_content = False
                    break
            
                if(start_getting_content == True and line_content != ''):
                    findings_content.append(line_content)
                
                if(line_content == ''):
                    new_line_for_findings_content = True
                    content_without_findings_keyword = []
                    
                elif(new_line_for_findings_content == True and 'FINDINGS:' not in line_content and 'IMPRESSION:' not in line_content):
                    content_without_findings_keyword.append(line_content)
            
            if(len(findings_content)==0 and len(content_without_findings_keyword) > 0):
                findings_content = content_without_findings_keyword
        
        contents_list.append(''.join(map(str,findings_content)))
        study_list.append( Path(text_files[i]).stem)
    
    assert len(study_list) == len(contents_list), "Mismatch number of studies and free-text reports"

    # Write to example_data\text\all_data.tsv for pairs of studyID and Findings in free-text report for Mutual Information training

    with open(os.path.join(args.text_storage_dir,'all_data.tsv'), 'w', encoding='utf8', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        
        for i in range(len(contents_list)):
            tsv_writer.writerow([i,0, study_list[i][1:],'a',contents_list[i]])
 
download_full_dataset(args.total_amount)

def populate_training_and_testing_dataset(amount_for_training, amount_for_testing):
    
    create_data_folder()

    current_study_count=0
    contents_list={}

    # Move file from full dataset to training dataset folder
    with open(os.path.join(args.text_storage_dir,'all_data.tsv'), "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", lineterminator='\n')
        
        for line in reader:
            text = line[-1]
            # labels = line[1]
            study_id = line[2]
            image_file = study_dictionary.get(study_id,'')
            if(text != '' and image_file != ''):
                current_study_count=current_study_count+1
                # copy image file to args.training_image_dir folder example_data/images
                copy_cmd = 'cp ' + image_file + ' ' + args.training_image_dir
               
                execute_command(copy_cmd)
                # append FINDINGs content to the list and write to file training_data.tsv in args.training_text_data_dir
                contents_list[study_id]=text
            if(current_study_count >= amount_for_training):
                break
    
    # Write FINDINGS in free text reports of training images to training_data.tsv
    training_data_file = os.path.join(args.training_text_dir,'all_data.tsv')
    
    with open(training_data_file, 'w', encoding='utf8', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        
        i=0
        for study_id in contents_list:
            tsv_writer.writerow([i,0, study_id,'a',contents_list[study_id]])
            i=i+1

    # Read label_filename and find study_id to add into args.training_dataset_metadata
    label_report_lines=[]
    line_count = 0
    with gzip.open(os.path.join(args.data_dir,label_filename), "rt") as f:
            for line in f:
                if(line_count > amount_for_training):
                    break
                if (line_count == 0):
                    new_line=[]
                    new_line.append('mimic_id')
                    new_line.extend(line.strip('\n').split(',')[2:])
                    label_report_lines.append(new_line)
                    line_count = line_count + 1

                else:
                    current_study_id = line.split(',')[1]
                    if (contents_list.get(current_study_id,'') != ''):
                        image_file = image_file_dictionary.get(current_study_id)
                        new_line=[]
                        new_line.append(image_file)
                        new_line.extend(line.strip('\n').split(',')[2:])
                        label_report_lines.append(new_line)                
                        line_count = line_count + 1

    with open(args.training_dataset_metadata, 'w') as tsv_file:
        tsv_writer = csv.writer(tsv_file)
        tsv_writer.writerows(label_report_lines)
 
populate_training_and_testing_dataset(args.amount_for_training, args.amount_for_testing)
