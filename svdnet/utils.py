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


def load_model(model, checkpoint_path):
  ckpt = torch.load(checkpoint_path)
  model.load_state_dict(ckpt["model_state_dict"], strict=False)

  return ckpt["epoch"]


def load_latest_train_model(model, save_dir, tag):
  epochs = [int(re.findall(".*{_tag}_(\d+)\.tar".format(_tag=tag), c)[0])
                    for c in glob.glob(os.path.join(save_dir, tag) + "_*.tar")]
  if len(epochs) == 0:
    return 0

  max_epoch = max(int(re.findall(".*{_tag}_(\d+)\.tar".format(_tag=tag), c)[0])
                    for c in glob.glob(os.path.join(save_dir, tag) + "_*.tar"))
  
  return load_model(model,
                          os.path.join(save_dir, f"{tag}_{max_epoch}.tar"))


def train_batch_step(data, target, model, optimizer):
  optimizer.zero_grad()
  data = data.to("cuda:0")
  target = target.to("cuda:0")
  predict = model(data)
  loss = F.cross_entropy(predict, target)
  loss.backward()
  optimizer.step()

  curr_acc = (predict.argmax(dim=1) == target).sum() / target.shape[0]

  return loss, curr_acc


def train_model_step0(model, dataloader, optimizer, ckpt_dir, tag, start_epoch=0, epochs_num=60):
  """
  Implements 'step0' training. 
  """
  model.train()
  model.to("cuda:0")

  for epoch in range(start_epoch, epochs_num):
    losses = []
    accs = []
    for batch_idx, (data, target) in enumerate(dataloader):
      loss, curr_acc = train_batch_step(data, target, model, optimizer)
      losses.append(loss.item())
      accs.append(curr_acc)

      if (batch_idx % 100) == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item(), curr_acc))
    
    print(f"AVG Loss after epoch {epoch + 1}: {sum(losses) / len(losses)}")
    print(f"AVG Acc after epoch {epoch + 1}: {sum(accs) / len(accs)}")
    if (epoch % 10 == 0):
      print(f"Saving checkpoint after {epoch + 1} epochs")
      save_train_model(model, optimizer, epoch + 1, ckpt_dir, tag)
    
    if epoch in [20, 30, 40, 50]:
      for g in optimizer.param_groups:
        g['lr'] /= 2
  
  save_train_model(model, optimizer, epochs_num, ckpt_dir, tag)


def load_optimizer_dict(optimizer, state_dict):
  optimizer.load_state_dict(state_dict)
  for state in optimizer.state.values():
    for k, v in state.items():
        if torch.is_tensor(v):
            state[k] = v.cuda()


def train_rri(model, dataloader, optimizer, ckpt_dir, rri_start=0, rri_count=7, rri_epochs=20):
  init_state_dict = optimizer.state_dict()

  for rri in range(rri_start, rri_count):
    load_optimizer_dict(optimizer, init_state_dict)

    print(f"\n\n-----------RRI: {rri}-----------")
    model.decorrelate()
    model.restraint()
    train_model_step0(model, dataloader, optimizer, ckpt_dir, f"rri_{rri}_restraint", epochs_num=rri_epochs)
    model.relaxation()
    print(f"\n-----------RRI: {rri}. Relaxation-----------")
    train_model_step0(model, dataloader, optimizer, ckpt_dir, f"rri_{rri}_relax", epochs_num=rri_epochs)


def train_more_no_rri(model, dataloader, optimizer, ckpt_dir, rri_start=0, rri_count=7, rri_epochs=20):
  init_state_dict = optimizer.state_dict()

  for rri in range(rri_start, rri_count):
    load_optimizer_dict(optimizer, init_state_dict)

    print(f"\n\n-----------No RRI: {rri}-----------")
    train_model_step0(model, dataloader, optimizer, ckpt_dir, f"more_no_rri_{rri}_restraint", epochs_num=rri_epochs)
    print(f"\n-----------No RRI: {rri}. Relaxation-----------")
    train_model_step0(model, dataloader, optimizer, ckpt_dir, f"more_no_rri_{rri}_relax", epochs_num=rri_epochs)