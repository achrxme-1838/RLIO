import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer


# class PointNet_RLIO(nn.Module):
#     def __init__(self, normal_channel=True, action_discrete_ranges=None):
#         super(PointNet_RLIO, self).__init__()
#         if normal_channel:
#             channel = 6
#         else:
#             channel = 3

   
#         k =  sum(tensor.numel() for tensor in action_discrete_ranges.values())

#         self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, k)
#         self.dropout = nn.Dropout(p=0.4)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         """
#         Ourput: Q(s, a), feature transform matrix
#         """
        
#         x, trans, trans_feat = self.feat(x)
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = F.relu(self.bn2(self.dropout(self.fc2(x))))
#         x = self.fc3(x)

#         # x = F.log_softmax(x, dim=1)

#         return x, trans_feat  # Note: x = Q(s, a) -> need to be softmaxed in the training loop
    
#         # return x, trans_feat  # X: global feature, trans_feat: feature transform matrix(auxiliary)

class PointNet_RLIO(nn.Module):
    def __init__(self, normal_channel=True, action_discrete_ranges=None):
        super(PointNet_RLIO, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3

        k = sum(tensor.numel() for tensor in action_discrete_ranges.values()) if action_discrete_ranges else 0

        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Output: Q(s, a), feature transform matrix
        """
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
