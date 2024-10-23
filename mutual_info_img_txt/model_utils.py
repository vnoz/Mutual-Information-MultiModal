import os
import logging
import csv
import sys
import numpy as np
from math import floor, ceil
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd
import cv2
import random

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from .utils import MimicID

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# Convert pulmonary edema severity (0-3) to one-hot encoding
def convert_to_onehot(severity):
    if severity == 0:
        return [1,0,0,0]
    elif severity == 1:
        return [0,1,0,0]
    elif severity == 2:
        return [0,0,1,0]
    elif severity == 3:
        return [0,0,0,1]
    else:
        raise Exception("No other possibilities of ordinal labels are possible")


class CXRImageReportDataset(torchvision.datasets.VisionDataset):
    """A CXR iamge-report dataset class that loads png images and 
    tokenized report text given a metadata file 
    and return image, text batches. 

    Args:
        img_dir (string): Root directory for the CXR images.
        dataset_metadata (string): File path of the metadata 
            that will be used to contstruct this dataset. 
            This metadata file should contain data IDs that are used to
            load images and labels associated with data IDs.
        data_key (string): The name of the column that has image IDs.
        transform (callable, optional): A function/tranform that takes in an image 
            and returns a transfprmed version.
    """
    
    def __init__(self, text_token_features, img_dir, dataset_metadata, 
                 data_key='mimic_id', transform=None, cache_images=False):
        super(CXRImageReportDataset, self).__init__(root=None, transform=transform)
        self.all_txt_tokens = {f.report_id: f.input_ids for f in text_token_features}
        self.all_txt_masks = {f.report_id: f.input_mask for f in text_token_features}
        self.all_txt_segments = {f.report_id: f.segment_ids for f in text_token_features}

        self.dataset_metadata = pd.read_csv(dataset_metadata)
        self.dataset_metadata['study_id'] = \
            self.dataset_metadata.apply(lambda row: \
                MimicID.get_study_id(row.mimic_id), axis=1)

        self.img_dir = img_dir
        self.data_key = data_key
        self.transform = transform
        self.image_ids = self.dataset_metadata[data_key]
        self.cache_images = cache_images
        if self.cache_images:
            self.cache_img_set() 
        else:
            self.images = None

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id, study_id = self.dataset_metadata.loc[idx, \
            [self.data_key, 'study_id']]

        txt = self.all_txt_tokens[study_id]
        txt = torch.tensor(txt, dtype=torch.long)

        txt_masks = self.all_txt_masks[study_id]
        txt_masks = torch.tensor(txt_masks, dtype=torch.long)

        txt_segments = self.all_txt_segments[study_id]
        txt_segments = torch.tensor(txt_segments, dtype=torch.long)

        if self.cache_images:
            img = self.images[str(idx)]
        else:
            png_path = os.path.join(self.img_dir, f'{img_id}.jpg')
            img = cv2.imread(png_path, cv2.IMREAD_ANYDEPTH)

        if self.transform is not None:
            img = self.transform(img)

        img = np.expand_dims(img, axis=0)

        return img, txt, txt_masks, txt_segments, study_id, img_id

    def cache_img_set(self):
        for idx in range(self.__len__()):
            img_id = self.dataset_metadata.loc[idx, self.data_key]
            png_path = os.path.join(self.data_dir, f'{img_id}.jpg')
            img = cv2.imread(png_path, cv2.IMREAD_ANYDEPTH)
            if idx == 0:
                self.images = {}
            self.images[str(idx)] = img

class CXRImageDataset(torchvision.datasets.VisionDataset):
    def __init__(self, img_dir, dataset_metadata, disease, disease_stats,
                 data_key='mimic_id', transform=None, cache_images=False):
        super(CXRImageDataset, self).__init__(root=None, transform=transform)
        
        filtered_df = pd.DataFrame(columns=[data_key, disease])
      
        filtered_row=[]
        with open(disease_stats, 'rt') as disease_csvfile:
            disease_csvreader = csv.reader(disease_csvfile, lineterminator='\n')
            for row in disease_csvreader:
                if(row[0] == disease):
                    filtered_row = row
                    break

        total_positive_study_for_disease = int(filtered_row[1])
        total_positive_study_ids_for_disease = filtered_row[2]

        total_negative_study_for_disease = total_positive_study_for_disease

        print('total_positive_study_for_disease: ' + str(total_positive_study_for_disease))
        print('total_negative_study_for_disease: ' + str(total_negative_study_for_disease))
                        

        with open(dataset_metadata, 'rt') as csvfile:
            csvreader = csv.reader(csvfile, lineterminator='\n')
            line_count=0
            total_positive_disease_count=0
            total_negative_disease_count=0
            total_disease_count = 0
            for row in csvreader:
                if(line_count==0):
                    print(row)
                else:
                    mimic_id = row[0]
                    study_id = mimic_id.split('_')[1][1:]
                    

                    if(study_id in total_positive_study_ids_for_disease):
                        if(total_positive_disease_count < total_positive_study_for_disease):
                            filtered_df.loc[total_disease_count]=[mimic_id,1]
                            total_disease_count = total_disease_count +1
                            total_positive_disease_count = total_positive_disease_count +1
                    elif(study_id not in total_positive_study_ids_for_disease):
                        if(total_negative_disease_count < total_negative_study_for_disease):
                            filtered_df.loc[total_disease_count]=[mimic_id,0]
                            total_disease_count = total_disease_count +1
                            total_negative_disease_count = total_negative_disease_count +1
                    

                    if(total_disease_count == total_positive_study_for_disease + total_negative_study_for_disease):
                      
                        # print('filtered_df')
                        # print(filtered_df)
                        break

                line_count =line_count + 1


        self.dataset_metadata = filtered_df 
        self.dataset_metadata['study_id'] = \
            self.dataset_metadata.apply(lambda row: \
                MimicID.get_study_id(row.mimic_id), axis=1)

        self.disease_label = disease

        self.img_dir = img_dir
        self.data_key = data_key
        self.transform = transform
        self.image_ids = self.dataset_metadata[data_key]
        self.cache_images = cache_images
        if self.cache_images:
            self.cache_img_set() 
        else:
            self.images = None

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id, study_id = self.dataset_metadata.loc[idx, \
            [self.data_key, 'study_id']]
        if self.dataset_metadata.loc[idx,self.disease_label] > 0:
            labelDisease = 1
        else:
            labelDisease=0

        if self.cache_images:
            img = self.images[str(idx)]
        else:
            jpg_path = os.path.join(self.img_dir, f'{img_id}.jpg')
            img = cv2.imread(jpg_path, cv2.IMREAD_ANYDEPTH)

        if self.transform is not None:
            img = self.transform(img)

        img = np.expand_dims(img, axis=0)

        return img, labelDisease

# adapted from
# https://towardsdatascience.com/https-medium-com-chaturangarajapakshe-text-classification-with-transformer-models-d370944b50ca
def load_and_cache_examples(args, tokenizer):
    logger = logging.getLogger(__name__)

    '''
    Load text features if they have been pre-processed;
    otherwise pre-process the raw text and save the features
    '''
    processor = ClassificationDataProcessor()
    num_labels = len(processor.get_labels())

    cached_features_file = os.path.join(
        args.text_data_dir,
        f"cachedfeatures_train_seqlen-{args.max_seq_length}")
    if os.path.exists(cached_features_file):
        logger.info(f"Loading features from cached file {cached_features_file}")
        print(f"Loading features from cached file {cached_features_file}")
        features = torch.load(cached_features_file)
    else:
        logger.info(f"Creating features from dataset file at {args.text_data_dir}")
        print(f"Creating features from dataset file at {args.text_data_dir}")
        label_list = processor.get_labels()
        examples = processor.get_all_examples(args.text_data_dir)
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer)
        logger.info(f"Saving features into cached file {cached_features_file}")
        print(f"Saving features into cached file {cached_features_file}")
        torch.save(features, cached_features_file)

    return features


class InputFeatures(object):
    """A single set of features of text data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, report_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.report_id = report_id


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, report_id, guid, text_a, text_b=None, labels=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) [string]. The labels of the example. This should be
            specified for train and dev examples, but not for test examples.
            report_id: id of the report like 4345466 without s or txt extension
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels
        self.report_id = report_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class ClassificationDataProcessor(DataProcessor):
    """Processor for multi class classification dataset. 
    Assume reading from a multiclass file
    so the label will be in 0-3 format
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
    
    def get_all_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "all_data.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]
    
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[-1]
            labels = line[1]
            report_id = line[2]
            examples.append(
                InputExample(
                    report_id=report_id, guid=guid, 
                    text_a=text_a, text_b=None, labels=labels))
        return examples


def convert_example_to_feature(example_row):
    """ 
    returns example_row
    """
    example, label_map, max_seq_length, tokenizer = example_row

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    #if output_mode == "classification":
    label_id = label_map[example.labels]
    #elif output_mode == "regression":
    #    label_id = float(example.label)
    #else:
    #    raise KeyError(output_mode)

    return InputFeatures(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         label_id=label_id, 
                         report_id=example.report_id)


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """
    converts examples to features
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    label_map['-1'] = -1 # To handle '-1' label (i.e. unlabeled data)
    examples_for_processing = [(example, label_map, max_seq_length, tokenizer) \
        for example in examples]
    process_count = cpu_count() - 1
    with Pool(process_count) as p:
            features = list(tqdm(p.imap(convert_example_to_feature, 
                                 examples_for_processing), 
                            total=len(examples)))
    return features


def generate_GradCAM_image(model, device, input_image, location_path):
    #Note: input_image is the np array for image from cv2.read in dataload.getItem function
    input_tensor = input_image.to(device)# Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    targets = [ClassifierOutputTarget(281)]
    model = model.to(device).eval()
    rgb_img = input_image
    # Construct the CAM object once, and then re-use it on many images.
    with GradCAM(model=model, target_layers=targets) as cam:
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        # You can also get the model outputs without having to redo inference
        model_outputs = cam.outputs

        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    cam_output_path = os.path.join(location_path, f'Grad_Cam.jpg')
    cv2.imwrite(cam_output_path, cam_image)