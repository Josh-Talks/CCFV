import torch
import numpy as np
from domain_gap.utils import load_h5
from utils.sliding_window_sampling import ms_sliding_window_sampling
from pytorch3dunet.unet3d.model import UNet2D

data_path = "/g/kreshuk/talks/data/epfl/resized_pixels/test.h5"
images = load_h5(data_path, 'raw', roi=[[0,10]])
labels = load_h5(data_path, 'labels', roi=[[0,10]])
images = np.expand_dims(images, axis=(0,1))
labels = np.expand_dims(labels, axis=(0,1))

images = torch.from_numpy(images)
labels = torch.from_numpy(labels)

model_config = {
    'name': 'UNet2D',
    'in_channels': 1,
    'out_channels': 1,
    'layer_order': 'bcr',
    'f_maps': 32,
    'final_sigmoid': True,
    'feature_return': False,
    'is_segmentation': True,
}

model = UNet2D(**model_config)
model.load_state_dict(
    torch.load(
        "/g/kreshuk/talks/segmentation_ModelSelection/experiments/EPFL/BatchNorm/E_model4/best_checkpoint.pytorch"
    )['model_state_dict']
)

ccfv_config = {
    "layers": [
        'encoders.2.basic_module.SingleConv2.ReLU', 
        'decoders.0.basic_module.SingleConv2.ReLU', 
        'decoders.1.basic_module.SingleConv2.ReLU'
    ],
    'sample_num': {
        "encoders.2.basic_module.SingleConv2.ReLU": 100,
        "decoders.0.basic_module.SingleConv2.ReLU": 200,
        "decoders.1.basic_module.SingleConv2.ReLU": 400
    },
    "num_classes": 1,

}

feature_dict = ms_sliding_window_sampling(
    ccfv_config["layers"],
    ccfv_config['sample_num'],
    images,
    labels,
    roi_size=[1, 256, 256],
    sw_batch_size=1,
    predictor=model,
    mode='gaussian',
)
