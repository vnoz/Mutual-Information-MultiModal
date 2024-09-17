import os
import argparse
import logging
import json
import requests
import csv
import sys

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

def download(url, output_dir):
    # open in binary mode
    # user, password = 'tuanle', '@A1thebest'
    with open(output_dir+'text.txt', "wb") as file:
    #     # get request
        # response = requests.get(url,auth=(user, password))
        response = requests.get(url,data={'user': 'tuanle', 'password': 'A1thebest'}, verify=False)
    #     # write to file
        file.write(response.content)

def get_filenames_list(fileList, save_location):
    wget_cmd = 'wget -r -N -c -np -nH --cut-dirs 10 --user tuanle --password A1thebest '
    host_url=  'https://physionet.org/files/mimic-cxr-jpg/2.1.0/'
    return wget_cmd + host_url + fileList + ' -P ' + save_location

def wget_download(cmd):
    os.system(cmd)

def populate_dataset():
    # Download IMAGE_FILENAMES from MIMIC-CXR JPG for all files
    filenames = 'IMAGE_FILENAMES'
    
    filenames_url = get_filenames_list(filenames,args.data_dir)

    wget_download(filenames_url)

    # Read content of IMAGE_FILENAMES for file list, and loop through each item to get free-text report from MIMIC-CXR
    lines = []
    with open(args.data_dir+filenames, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)

    print(lines)

    # TODO: Write to example_data\text\all_data.tsv for pairs of studyID and Findings in free-text report for Mutual Information training
    # download_top10_cmd = 'head -n 10 IMAGE_FILENAMES | wget -r -N -c -np -nH --cut-dirs=1 --user tuanle --password A1thebest -i - --base=https://physionet.org/files/mimic-cxr-jpg/2.1.0/ -P '+ args.image_dir

populate_dataset()
    

