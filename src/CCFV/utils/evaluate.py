import torch
from tqdm import tqdm
import numpy as np
from CCFV.utils.sliding_window_sampling import sampling_2d_images
from CCFV.utils.ccfv import cal_variety, cal_w_distance


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