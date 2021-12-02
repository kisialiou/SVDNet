import glob
import os
import re

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from tqdm import tqdm

class Market_1501(Dataset):

  def __init__(self, data_dir, return_orig_class=False, preload_img=False):
    self.data_dir = data_dir
    self.items = sorted(glob.glob(f"{data_dir}/*.jpg"))
    # All images are intentionally brought to memory in order to speed up I/O
    # during training
    self.preload_img = preload_img
    if preload_img:
      images = []
      for img_path in tqdm(self.items):
        images.append(torchvision.io.read_image(img_path) / 255)
      self.imgs = torch.stack(tuple(images), dim=0)

    all_labels = sorted(set(self._get_class_from_str(os.path.basename(img_name)) 
                                                for img_name in self.items))
    self.map_orig_to_trunc = {val:i for i, val in enumerate(all_labels)}
    self.map_trunc_to_orig = {i:val for i, val in enumerate(all_labels)}
    self.class_count = len(self.map_trunc_to_orig)

    self.return_orig_class = return_orig_class

  def __len__(self):
    return len(self.items)

  @staticmethod
  def _get_class_from_str(filename):
    return int(re.findall("^([-]?\d+)_*", filename)[0])
  
  def __getitem__(self, idx):
    img_name = os.path.basename(self.items[idx])
    img_path = self.items[idx]

    img = self.imgs[idx] if self.preload_img else torchvision.io.read_image(img_path) / 255
    img_class = self._get_class_from_str(img_name)

    return_class = img_class if self.return_orig_class else self.map_orig_to_trunc[img_class]

    return img, return_class


def apply_predictor(predictor, img_dir, batch_size=32):
    dataset = Market_1501(img_dir, return_orig_class=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    feats = []
    results_targ = []
    for data, targets in tqdm(loader):
      data = data.cuda()
      targets = targets.cuda()
      result = predictor(data)
      feats.append(result.detach())
      results_targ.append(targets)
    
    return torch.vstack(tuple(feats)), torch.hstack(tuple(results_targ)), dataset
