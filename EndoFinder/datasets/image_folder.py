import functools
import logging
import os.path

from torchvision.datasets.folder import is_image_file
from torchvision.datasets.folder import default_loader

@functools.lru_cache()
def get_image_paths(path):
    logging.info(f"Resolving files in: {path}")
    filenames = [f"{path}/{file}" for file in os.listdir(path)]
    return sorted([fn for fn in filenames if is_image_file(fn)])

class POLYPImageFolder:
    def __init__(self, image_path, mask_path, transform=None, loader=default_loader):
        self.files = get_image_paths(image_path)
        self.masks = get_image_paths(mask_path)
        self.loader = loader
        self.transform = transform

    def __getitem__(self, idx: int):
        assert 0 <= idx < len(self)
        img = self.loader(self.files[idx])
        mask = self.loader(self.masks[idx]).convert('L')
        record = {"input": img, "mask": mask, "instance_id": idx}
        if self.transform:
            record = self.transform(record)
        return record

    def __len__(self):
        return len(self.files)