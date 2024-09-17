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

parser.add_argument('--image_dir', type=str,
                    default=os.path.join(current_dir, 'example_data/images/'),
                    help='The image data directory')
parser.add_argument('--text_data_dir', type=str,
                    default=os.path.join(current_dir, 'example_data/text/'),
                    help='The text data directory')
parser.add_argument('--data_dir', type=str,
                    default=os.path.join(current_dir, 'example_data/'),
                    help='The parent data directory')


args = parser.parse_args()

print(f"Initial args: {args}")


def get_filename_url(base, file, save_location):
    wget_cmd = 'wget -r -N -c -np -nH --cut-dirs 10 --user tuanle --password A1thebest '
    host_url=  os.path.join('https://physionet.org/files/',base)
    return wget_cmd + host_url + file + ' -P ' + save_location 


def get_filename_new_location_url(base,file, new_filename):
    wget_cmd = 'wget -r -N -c -np -nH --cut-dirs 10 --user tuanle --password A1thebest '
    host_url=  os.path.join('https://physionet.org/files/',base)
    return wget_cmd + host_url + file + ' -O '+new_filename

def wget_download(cmd):
    os.system(cmd)

def populate_dataset(imgAmount):

    # Download mimic-cxr-2.0.0-metadata.csv.gz from MIMIC-CXR JPG for all files metadata
    filenames = 'mimic-cxr-2.0.0-metadata.csv.gz'
    jpg_cxr_base_url = 'mimic-cxr-jpg/2.1.0/'
    mimic_cxr_base_url = 'mimic-cxr/2.1.0/'

    if not os.path.exists(os.path.join(args.data_dir,filenames)):
        filenames_url = get_filename_url(jpg_cxr_base_url,filenames,args.data_dir)

        wget_download(filenames_url)

    # Read content of file list, and loop through each item to get free-text report from MIMIC-CXR
    text_files = []
    
    count = 0

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

    for i in range(len(text_files)):
        text_file_url = get_filename_url(mimic_cxr_base_url,text_files[i],args.text_data_dir)
        wget_download(text_file_url)


    contents_list=[]
    study_list=[]

    for i in range(len(text_files)):
        findings_content=[]
        start_getting_content=False
        
        single_text_file =text_files[i].split('/')[-1]

        with open(os.path.join(args.text_data_dir,single_text_file),"rt") as f:
            for line in f:
                if(line.strip() == 'FINDINGS:'):
                    start_getting_content = True
                    continue
                if(line.strip() == 'IMPRESSION:' and start_getting_content==True):
                    start_getting_content = False
                    break
                if(start_getting_content == True and line.strip() != ''):
                    findings_content.append(line.strip())
        
        contents_list.append(''.join(map(str,findings_content)))
        study_list.append( Path(text_files[i]).stem)
    
    assert len(study_list) == len(contents_list), "Mismatch number of studies and free-text reports"

    # Write to example_data\text\all_data.tsv for pairs of studyID and Findings in free-text report for Mutual Information training

    with open(os.path.join(args.text_data_dir,'all_data.tsv'), 'w', encoding='utf8', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        
        for i in range(len(contents_list)):
            tsv_writer.writerow([i,0, study_list[i][1:],'a',contents_list[i]])


    
populate_dataset(3)
    

