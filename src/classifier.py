import torch
from torch.nn.utils.weight_norm import WeightNorm

# -- linear classifier from https://github.com/facebookresearch/msn/
class LinearClassifier(torch.nn.Module):

    def __init__(self, dim, num_labels=1000, normalize=True, layer_norm=True):
        super(LinearClassifier, self).__init__()
        self.normalize = normalize
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.norm = torch.nn.LayerNorm(dim)
        self.linear = torch.nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        if self.layer_norm:
            x = self.norm(x)
        if self.normalize:
            x = torch.nn.functional.normalize(x)
        return self.linear(x)

# -- baseline++ from https://github.com/wyharveychen/CloserLookFewShot
class distLinear(torch.nn.Module):

    def __init__(self, dim, num_labels=1000, normalize=True):
        super(distLinear, self).__init__()
        self.L = torch.nn.Linear(dim, num_labels, bias=False)
        self.class_wise_learnable_norm = normalize
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0)
        if num_labels <= 200:
            self.scale_factor = 2
        else:
            self.scale_factor = 10

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized)
        scores = self.scale_factor * cos_dist
        return scores