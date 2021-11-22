import glob
import os
import re

import torch
import torch.nn.functional as F


def save_train_model(model, optimizer, epoch, save_dir, tag):
  torch.save({
      "epoch": epoch,
      "model_state_dict": model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
  }, os.path.join(save_dir, f"{tag}_{epoch}.tar"))

def load_train_model(model, optimizer, checkpoint_path):
  ckpt = torch.load(checkpoint_path)
  model.load_state_dict(ckpt["model_state_dict"])
  optimizer.load_state_dict(ckpt["optimizer_state_dict"])

  return ckpt["epoch"]


def load_latest_train_model(model, optimizer, save_dir, tag):
  epochs = [int(re.findall(".*{_tag}_(\d+)\.tar".format(_tag=tag), c)[0])
                    for c in glob.glob(os.path.join(save_dir, tag) + "_*.tar")]
  if len(epochs) == 0:
    return 0

  max_epoch = max(int(re.findall(".*{_tag}_(\d+)\.tar".format(_tag=tag), c)[0])
                    for c in glob.glob(os.path.join(save_dir, tag) + "_*.tar"))
  
  return load_train_model(model, optimizer, 
                          os.path.join(save_dir, f"{tag}_{max_epoch}.tar"))


def train_model(model, dataloader, optimizer, ckpt_dir, start_epoch=0, epochs_num=60):
  """
  Implements 'step0' training. 
  """
  model.train()
  model.to("cuda:0")

  for epoch in range(start_epoch, epochs_num):
    for batch_idx, (data, target) in enumerate(dataloader):
      optimizer.zero_grad()
      data = data.to("cuda:0")
      target = target.to("cuda:0")
      predict = model(data)
      loss = F.nll_loss(predict, target)
      loss.backward()
      optimizer.step()

      if (batch_idx % 1000) == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item()))
      
    if (epoch % 2 == 0):
      print(f"Saving checkpoint after {epoch + 1} epochs")
      save_train_model(model, optimizer, epoch + 1, ckpt_dir, "step0")