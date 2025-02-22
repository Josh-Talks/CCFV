from CCFV.utils.sliding_window_sampling import sampling_2d_images
import torch
import torch.nn as nn
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from pytorch3dunet.datasets.dsb import TIF_txt_Dataset
from pytorch3dunet.unet3d.model import UNet2D
# import dataloader
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from CCFV.utils.ccfv import cal_variety, cal_w_distance

transformer_config = {
    "raw": [
        #{"name": "CropToFixed", "size": [256,256]},
        {"name": "PercentileNormalizer"},
        {"name": "FixedClipping", "max_value": 2},
        {"name": "ToTensor", "expand_dims": True},
    ],
    "label": [
        #{"name": "CropToFixed", "size": [256,256]},
        {"name": "Relabel"},
        {"name": "BlobsToMask", "append_label": False},
        {"name": "ToTensor", "expand_dims": True},
    ]
}

model_config = {
    "name": "UNet2D",
    "in_channels": 1,
    "out_channels": 1,
    "layer_order": "bcr",
    "f_maps": [32, 64, 128],
    "final_sigmoid": True,
    "feature_return": False,
    "is_segmentation": True,
    "feature_perturbation": None,
}

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

model = UNet2D(**model_config)
model.load_state_dict(
    torch.load(
        "/g/kreshuk/talks/segmentation_ModelSelection/experiments/BBBC039/BatchNorm/BC_model4/best_checkpoint.pytorch"
    )['model_state_dict']
)

dataset = TIF_txt_Dataset(
    image_dir="/g/kreshuk/talks/data/BBBC039/images",
    mask_dir="/g/kreshuk/talks/data/BBBC039/instance_annotations/instance_labels",
    phase='train',
    transformer_config=transformer_config,
    filenames_path="/g/kreshuk/talks/data/BBBC039/test.txt",
    global_norm=True,
    percentiles=[5,98]
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
layers = [
    'encoders.2.basic_module.SingleConv2.ReLU', 
    'decoders.0.basic_module.SingleConv2.ReLU', 
    'decoders.1.basic_module.SingleConv2.ReLU'
]

def evaluate(config, test_loader, model):
    model.eval()
    layers = config["layers"]
    class_feature_dict = {
        layer: {j: [] for j in range(config["num_classes"]+1)} for layer in layers
    }
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    global_feat_dict = {layer: [] for layer in layers}
    with torch.no_grad():
        for idx, (img, label) in tqdm(enumerate(test_loader), desc="Batch"):
            #print("evaluating index:{}".format(idx))
            # 2d image remove empty depth dimension
            img, label = img.to(device), label.to(device)
            img = torch.squeeze(img, dim=-3)
            label = torch.squeeze(label, dim=-3)
            feature_dict = sampling_2d_images(
                config["layers"],
                sample_num=config['sample_num'],
                inputs=img,
                labels=label,
                predictor=model,
            )
            for layer in class_feature_dict.keys():
                global_feat = np.concatenate(
                    [feat for feat in feature_dict[layer].values()], axis=0)
                global_feat = np.mean(global_feat, axis=0)
                global_feat_dict[layer].append(global_feat)
                for lb in class_feature_dict[layer].keys():
                    class_feature_dict[layer][lb].append(
                        feature_dict[layer][lb])

    ccfv = 0
    for layer in class_feature_dict.keys():
        w_distance = 0.0
        global_feature = np.array(global_feat_dict[layer])
        var_f = cal_variety(global_feature)

        for lb in class_feature_dict[layer].keys():
            if lb == 0:
                continue
            length = len(class_feature_dict[layer][lb])
            for i in range(length):
                for j in range(i+1, length):
                    w_distance += cal_w_distance(
                        class_feature_dict[layer][lb][i], class_feature_dict[layer][lb][j])
            w_distance += w_distance / (length * length / 2) / config["num_classes"]
        ccfv += np.log(var_f / w_distance)

    print("ccfv:", ccfv)
    return ccfv


if __name__ == '__main__':
    ccfv = evaluate(ccfv_config, dataloader, model)
