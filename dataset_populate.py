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
                    default=os.path.join(current_dir, 'data_storage/'),
                    help='The parent data directory')

parser.add_argument('--image_dir', type=str,
                    default=os.path.join(current_dir, 'data_storage/images/'),
                    help='The image data directory')

parser.add_argument('--text_data_dir', type=str,
                    default=os.path.join(current_dir, 'data_storage/text/'),
                    help='The text data directory')

parser.add_argument('--download_user', type=str,
                    default='tuanle',
                    help='The user to download MIMIC dataset')

parser.add_argument('--download_password', type=str,
                    default='A1thebest',
                    help='The password to download MIMIC dataset')

parser.add_argument('--total_amount', type=str,
                    default=10000,
                    help='Total amount of samples to download from MIMIC dataset')

parser.add_argument('--amount_for_training', type=str,
                    default=1000,
                    help='Total amount of samples for training')


parser.add_argument('--amount_for_testing', type=str,
                    default=100,
                    help='Total amount of samples for testing')



args = parser.parse_args()

print(f"Initial args: {args}")


def get_filename_url(base, file, save_location):
    wget_cmd = 'wget -r -N -c -np -nH --cut-dirs 10 --user '+ args.download_user + ' --password '+ args.download_password + ' '
    host_url=  os.path.join('https://physionet.org/files/',base)
    return wget_cmd + host_url + file + ' -P ' + save_location 

def process_file(base, filename):
    url = get_filename_url(base, filename, args.data_dir)
    wget_download(url)

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

def wget_download(cmd):
    os.system(cmd)

def download_full_dataset(imgAmount):

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    if not os.path.exists(args.image_dir):
        os.makedirs(args.image_dir)
    if not os.path.exists(args.text_data_dir):
        os.makedirs(args.text_data_dir)

    # Download mimic-cxr-2.0.0-metadata.csv.gz from MIMIC-CXR JPG for all files metadata
    filenames = 'mimic-cxr-2.0.0-metadata.csv.gz'
    jpg_cxr_base_url = 'mimic-cxr-jpg/2.1.0/'
    mimic_cxr_base_url = 'mimic-cxr/2.1.0/'

    if not os.path.exists(os.path.join(args.data_dir,filenames)):
        filenames_url = get_filename_url(jpg_cxr_base_url,filenames,args.data_dir)
        print('Start downloading meta data file' + filenames_url)
        wget_download(filenames_url)

    # Read content of file list, and loop through each item to get free-text report from MIMIC-CXR
    text_files = []
    
    count = 0

    # print('Open gzip file '+ filenames)
    # print('Downloading images from MIMIC-CXR-JPG')
    with gzip.open(os.path.join(args.data_dir,filenames), "rt") as f:
            
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
                    new_img_filename = os.path.join(args.image_dir,'p'+subject_id+'_'+'s'+ study_id+'_'+dicom_id+'.jpg')
                    img_url = get_filename_new_location_url(jpg_cxr_base_url,img_filename,new_img_filename)
                    wget_download(img_url)
                    
                    text_filename = os.path.join('files','p'+subject_id[:2],'p'+subject_id,'s'+ study_id+'.txt')
                    text_files.append(text_filename)
                    
                    count = count+1

                    if(count > imgAmount):
                        break
    
    # print('Downloading text files in MIMIC-CXR')
    for i in range(len(text_files)):
        text_file_url = get_filename_url(mimic_cxr_base_url,text_files[i],args.text_data_dir)
        wget_download(text_file_url)


    contents_list=[]
    study_list=[]

    for i in range(len(text_files)):
        findings_content=[]
        start_getting_content=False
        
        single_text_file =text_files[i].split('/')[-1]

        content_without_findings_keyword=[]
        new_line_for_findings_content = False

        # NOTE: if content has Findings keyword, then return the text between Findings and Impression keywords, 
        #           otherwise return the text after the last empty line break
        with open(os.path.join(args.text_data_dir,single_text_file),"rt") as f:
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

    # print('Start adding study_id and findings in all_data.tsv')
    with open(os.path.join(args.text_data_dir,'all_data.tsv'), 'w', encoding='utf8', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        
        for i in range(len(contents_list)):
            tsv_writer.writerow([i,0, study_list[i][1:],'a',contents_list[i]])
 
# download_full_dataset(args.total_amount)

def populate_training_and_testing_dataset(amount_for_training, amount_for_testing):
    #TODO: read data_storage/text/all_data.tsv, select amount of studies having Findings content in data_storage folder, and assign to training and testing
    print('Total amount for training: '+ str(amount_for_training)+', testing: ' + str(amount_for_testing))
    

populate_training_and_testing_dataset(args.amount_for_training, args.amount_for_testing)
