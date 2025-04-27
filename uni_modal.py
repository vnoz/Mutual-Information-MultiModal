
from mutual_info_img_txt.main_utils import UniModalManager


def train_auto_encoder(args, device):
    #Note: setup parameters for Autoencoder
    print('Initialise UniModalManager')
    model_manager = UniModalManager(
                                        output_channels=args.output_channels,
                                        image_model_name=args.image_model_name)
    print('Unimodal training start')
    model_manager.train(device=device, args=args)
    
