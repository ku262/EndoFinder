# EndoFinder: Online Image Retrieval for Explainable Colorectal Polyp Diagnosis

recently accepted to [MICCAI 2024](https://link.springer.com/chapter/10.1007/978-3-031-72117-5_24).

This work employs self-supervised contrastive learning combined with a polyp-aware masked reconstruction task to learn a universal representation of polyps for image retrieval.

<div align="center">
  <img width="80%" alt="EndoFinder diagram" src="img/EndoFinder.png">
</div>

______

## Preparation

- ### Pretrained models

  We provide trained models from our original experiments to allow others to reproduce our evaluation results [EndoFinder.pth](https://huggingface.co/KU626/EndoFinder/blob/main/pretrained_models/EndoFinder.pth).

- ### Dataset

  We provide Polyp-Twin [here](https://huggingface.co/KU626/EndoFinder/blob/main/PolypTwin.zip)
______

## Installation
- ### Option 1: Install dependencies using Conda

  Install and activate conda, then create a conda environment for EndoFinder as follows:

  ```bash
  # Create conda environment
  conda create --name EndoFinder -c pytorch -c conda-forge \
    pytorch torchvision cudatoolkit=11.3 \
    "pytorch-lightning>=1.5,<1.6" lightning-bolts \
    faiss python-magic pandas numpy

  # Activate environment
  conda activate EndoFinder

  # Install Classy Vision and AugLy from PIP:
  python -m pip install classy_vision augly
  ```

- ### Option 2: Install dependencies using PIP

  ```bash
  # Create environment
  python3 -m virtualenv ./venv

  # Activate environment
  source ./venv/bin/activate

  # Install dependencies in this environment
  python -m pip install -r ./requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
  ```
______

## Inference using EndoFinder models

This section describes how to use pretrained EndoFinder models for inference.

- ### Preprocessing

  We recommend preprocessing images for inference either resizing the small edge to 224 or resizing the image to a square tensor.

  ```python
  from torchvision import transforms

  normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
  )
  small_224 = transforms.Compose([
      transforms.Resize(224),
      transforms.ToTensor(),
      normalize,
  ])

  ```

- ### Inference 

  ```python
  import torch
  from EndoFinder.tools.img2jsonl import LoadModelSetting
  from PIL import Image

  model = LoadModelSetting.get_model(LoadModelSetting['vit_large_patch16'], "/path/to/EndoFinder.pth", use_hash=False)
  img = Image.open("/path/to/image.png").convert('RGB')
  batch = small_224(img).unsqueeze(0)
  embedding = model(batch)[0, :]
  ```

- ### Load model weight files

  To load model weight files, first construct the `Model` object, then load the weights using the standard `torch.load` and `load_state_dict` methods.

  ```python
  import torch
  from EndoFinder.models import models_vit

  model = models_vit.__dict__["vit_large_patch16"](use_hash=False)
  checkpoint = torch.load("/path/to/EndoFinder.pth", map_location='cpu')
  checkpoint_model = checkpoint['model']
  model.load_state_dict(checkpoint_model, strict=False)
  model.eval()
  ```
______

## Reproducing evaluation results

To reproduce evaluation results, see [Evaluation](docs/Evaluation.md).
______

## Training EndoFinder models

For information on how to train EndoFinder models, see 
[Training](docs/Training.md).
______

## Reference

The code in this article was inspired by and references the following articles. 
Their code has been immensely helpful to me.

```
Masked Autoencoders Are Scalable Vision Learners
A Self-Supervised Descriptor for Image Copy Detection
```
______

## Citation

If you find our codebase useful, please consider giving a star :star: and cite as:

```
@inproceedings{yang2024endofinder,
  title={EndoFinder: Online Image Retrieval for Explainable Colorectal Polyp Diagnosis},
  author={Yang, Ruijie and Zhu, Yan and Fu, Peiyao and Zhang, Yizhe and Wang, Zhihua and Li, Quanlin and Zhou, Pinghong and Yang, Xian and Wang, Shuo},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={251--262},
  year={2024},
  organization={Springer}
}
```

