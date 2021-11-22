import glob
import os

import torch
from torch.utils.data import Dataset
import torchvision
import tqdm

class Market_1501(Dataset):

  def __init__(self, data_dir, preload_img=False):
    self.data_dir = data_dir
    self.items = glob.glob(f"{data_dir}/*.jpg")
    # All images are intentionally brought to memory in order to speed up I/O
    # during training
    self.preload_img = preload_img
    if preload_img:
      images = []
      for img_path in tqdm(self.items):
        images.append(torchvision.io.read_image(img_path) / 255)
      self.imgs = torch.stack(tuple(images), dim=0)

    self.class_count = max(self._get_class_from_str(os.path.basename(img_name)) 
                                                for img_name in self.items) + 1
  
  def __len__(self):
    return len(self.items)

  @staticmethod
  def _get_class_from_str(filename):
    return int(filename[:4]) - 1
  
  def __getitem__(self, idx):
    img_name = os.path.basename(self.items[idx])
    img_path = self.items[idx]

    img = self.imgs[idx] if self.preload_img else torchvision.io.read_image(img_path) / 255
    img_class = self._get_class_from_str(img_name)

    return img, img_class
