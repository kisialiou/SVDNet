import torch
import torch.nn
import torch.linalg
import torch.nn.functional as F
import torchvision


class SVDNet(torch.nn.Module):

  def __init__(self, out_classes, eigen_out_feats=1024):
    super().__init__()
    self.backbone = torchvision.models.resnet50(pretrained=True)

    self.backbone.fc = torch.nn.Linear(in_features=self.backbone.fc.in_features,
                                       out_features=eigen_out_feats,
                                       bias=False)

    self.fc = torch.nn.Linear(in_features=eigen_out_feats,
                              out_features=out_classes)

  def forward(self, x):
    if self.training:
      x = self.backbone(x)
      x = self.fc(x)
      res = F.softmax(x, dim=0)
    else:
      res = self.backbone(x)
    
    return res

  def decorrelate(self):
    with torch.no_grad():
      # When full_matrices=False U - is left orhogonal, i.e. it has orthogonal columns
      U, S, _ = torch.linalg.svd(self.backbone.fc.weight.T.data, full_matrices=False)
      self.backbone.fc.weight.data = (U @ torch.diag(S)).T
  
  def restraint(self):
    self.backbone.fc.weight.requires_grad = False
  
  def relaxation(self):
    self.backbone.fc.weight.requires_grad = True