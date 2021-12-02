import torch
import torch.nn
import torch.linalg
import torch.nn.functional as F
import torchvision


class SVDNet(torch.nn.Module):

  def __init__(self, out_classes, eigen_out_feats=1024, eval_only=False):
    super().__init__()
    self.backbone = torchvision.models.resnet50(pretrained=True, progress=False)

    self.backbone.fc = torch.nn.Linear(in_features=self.backbone.fc.in_features,
                                       out_features=eigen_out_feats,
                                       bias=False)
    if not eval_only:
      self.fc = torch.nn.Linear(in_features=eigen_out_feats,
                                out_features=out_classes)

    self.gallery_feats = None

  def forward(self, x):
    if self.training:
      x = self.backbone(x)
      x = self.fc(x)
      res = F.softmax(x, dim=1)
    else:
      if self.gallery_feats is None:
        raise ValueError("Gallery is not provided. Inference can't be done.")
      inp_feats = self.backbone(x)
      res = self._get_gallery_top_k(inp_feats)
    
    return res

  def set_gallery(self, gallery_feats):
    """gallery_feats - each row is a vector representing some image from image gallery"""
    self.gallery_feats = gallery_feats

  def get_feat_vector(self, x):
    self.eval()
    return self.backbone(x)

  def _get_gallery_top_k(self, query):
    k = 5
    dists = torch.cdist(query, self.gallery_feats) # returns (query_size, gallery_size)
    return torch.topk(dists, k, dim=1, largest=False).indices

  def decorrelate(self):
    with torch.no_grad():
      # When full_matrices=False U - is left orhogonal, i.e. it has orthogonal columns
      U, S, _ = torch.linalg.svd(self.backbone.fc.weight.T.data, full_matrices=False)
      curr_device = self.backbone.fc.weight.device 
      self.backbone.fc.weight.data = (U @ torch.diag(S)).T
      self.backbone.fc.weight.to(curr_device)
  
  def restraint(self):
    self.backbone.fc.weight.requires_grad = False
  
  def relaxation(self):
    self.backbone.fc.weight.requires_grad = True