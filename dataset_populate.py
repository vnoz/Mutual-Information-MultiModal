import os
import argparse
import logging
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)

parser = argparse.ArgumentParser()

parser.add_argument('--image_dir', type=str,
                    default=os.path.join(current_dir, 'example_data/images/'),
                    help='The image data directory')
parser.add_argument('--text_data_dir', type=str,
                    default=os.path.join(current_dir, 'example_data/text/'),
                    help='The text data directory')


args = parser.parse_args()

print(f"Initial args: {args}")