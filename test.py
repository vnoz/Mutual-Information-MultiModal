import os
import cv2 # type: ignore
from mutual_info_img_txt.model import Basic_MLP
from mutual_info_img_txt.model_utils import generate_GradCAM_image
import torch  # type: ignore
from helpers import get_transform_function

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad # type: ignore
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget # type: ignore
from pytorch_grad_cam.utils.image import show_cam_on_image # type: ignore

img_path = os.path.join('full_data_set/images/p10', 'p10000032_s50414267_02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print('Start generate heatmap for image: ' + str(img_path))

img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
transform_fn = get_transform_function(256)
img = transform_fn(img)

output_model_file = os.path.join('save_dir/dv_total_epochs10', 'pytorch_image_classifier_Edema.bin')

image_classifier_model = Basic_MLP.load_from_pretrained(output_model_file)

generate_GradCAM_image(image_classifier_model, device=device, input_image = img,location_path='save_dir/heatmap')

print('Finish generate heatmap for image: ' + str(img_path))
