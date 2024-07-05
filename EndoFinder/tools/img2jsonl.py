from torchvision import transforms
import os
from torchvision.datasets.folder import is_image_file
import json
import torch
from PIL import Image
import enum
import os.path
from pathlib import Path
import pandas as pd

from timm.models.layers import trunc_normal_

import sys
# Make sure EndoFinder is in PYTHONPATH.

base_path = str(Path(__file__).resolve().parent.parent.parent)
if base_path not in sys.path:
    sys.path.append(base_path)

from EndoFinder.models import models_vit
from EndoFinder.util.pos_embed import interpolate_pos_embed

from torchvision.models import resnet50, vgg19, densenet121
from timm import create_model as creat
import torch.nn as nn
from torch.nn import functional as F

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
)
small_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])

class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x)

class LoadModelSetting(enum.Enum):
    vit_large_patch16 = enum.auto()
    vit_base_patch16 = enum.auto()
    vit_huge_patch14 = enum.auto()

    
    def get_model(self, model_path):
        config = self._get_config(self)
        model = models_vit.__dict__[config]()

        if config == "vit_large_patch16":

            checkpoint = torch.load(model_path, map_location='cpu')
            print("Load pre-trained checkpoint from: %s" % model_path)
            checkpoint_model = checkpoint['model']
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            # interpolate position embedding
            interpolate_pos_embed(model, checkpoint_model)

            # load pre-trained model
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(msg)

            return model

        if config == "vit_base_patch16":

            checkpoint = torch.load(model_path, map_location='cpu')
            print("Load pre-trained checkpoint from: %s" % model_path)
            checkpoint_model = checkpoint['model']
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            # interpolate position embedding
            interpolate_pos_embed(model, checkpoint_model)

            # load pre-trained model
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(msg)

            return model

        if config == "vit_huge_patch14":

            checkpoint = torch.load(model_path, map_location='cpu')
            print("Load pre-trained checkpoint from: %s" % model_path)
            checkpoint_model = checkpoint['model']
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            # interpolate position embedding
            interpolate_pos_embed(model, checkpoint_model)

            # load pre-trained model
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(msg)

            return model

    def _get_config(self, value):
        return {
            self.vit_large_patch16: "vit_large_patch16",
            self.vit_base_patch16: "vit_base_patch16",
            self.vit_huge_patch14: "vit_huge_patch14",
        }[value]    

def readxlsx(path):
    df = pd.read_excel(path)
    img_path_list, ground_truths = df.iloc[:, 2], df.iloc[:, 4] 
    return img_path_list, ground_truths

def get_image_paths(path):
    filenames = [f"{path}/{file}" for file in os.listdir(path)]
    return sorted([fn for fn in filenames if is_image_file(fn)])

def img2embeddings(version, model, dst_path, image_path, jsonl_path):

    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    files = get_image_paths(image_path)
    data = []
    idx = 0
    for path in files:
        idx += 1
        if(idx%100==0):
            print(f"{idx}/{len(files)}")
        json_item = {}
        json_item["type"] = "polyp"
        json_item["model"] = version
        json_item['folder'] = image_path
        json_item['path'] = path
        img = Image.open(path).convert('RGB')
        # batch = small_224(img).unsqueeze(0).cuda()
        # model.cuda()
        batch = small_224(img).unsqueeze(0).to("cuda:3")
        model.to("cuda:3")

        cls_token = model(batch)[0, :]
        embedding = cls_token
        json_item['embedding'] = embedding.detach().cpu().numpy().tolist()
        data.append(json_item)

    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

