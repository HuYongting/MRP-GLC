import torch
import torch.nn as nn
import torch.nn.functional
import numpy as np


class Flow_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gen_flows, gt_flows):
        return torch.mean(torch.abs(gen_flows - gt_flows))


class Intensity_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gen_frames, gt_frames):
        return torch.mean(torch.abs((gen_frames - gt_frames) ** 2))

class Intensity_5frame_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = Intensity_Loss().cuda()

    def forward(self, gen_frames, gt_frames):
        intels = []
        for i in range(5):
            gt = gt_frames[:, i*3:(i+1)*3, :, :]
            gen = gen_frames[:, i*3:(i+1)*3, :, :]
            intel = self.loss(gen,gt)
            intels.append(intel)
        return sum(intels)/len(intels)


class Gradient_Loss(nn.Module):
    def __init__(self, channels):
        super().__init__()

        pos = torch.from_numpy(np.identity(channels, dtype=np.float32))
        neg = -1 * pos
        # Note: when doing conv2d, the channel order is different from tensorflow, so do permutation.
        self.filter_x = torch.stack((neg, pos)).unsqueeze(0).permute(3, 2, 0, 1).cuda()
        self.filter_y = torch.stack((pos.unsqueeze(0), neg.unsqueeze(0))).permute(3, 2, 0, 1).cuda()

    def forward(self, gen_frames, gt_frames):
        # Do padding to match the  result of the original tensorflow implementation
        gen_frames_x = nn.functional.pad(gen_frames, [0, 1, 0, 0])
        gen_frames_y = nn.functional.pad(gen_frames, [0, 0, 0, 1])
        gt_frames_x = nn.functional.pad(gt_frames, [0, 1, 0, 0])
        gt_frames_y = nn.functional.pad(gt_frames, [0, 0, 0, 1])

        gen_dx = torch.abs(nn.functional.conv2d(gen_frames_x, self.filter_x))
        gen_dy = torch.abs(nn.functional.conv2d(gen_frames_y, self.filter_y))
        gt_dx = torch.abs(nn.functional.conv2d(gt_frames_x, self.filter_x))
        gt_dy = torch.abs(nn.functional.conv2d(gt_frames_y, self.filter_y))

        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        return torch.mean(grad_diff_x + grad_diff_y)

# ??????
class Adversarial_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fake_outputs):
        # TODO: compare with torch.nn.MSELoss ?
        return torch.mean((fake_outputs - 1) ** 2 / 2)

# ??????
class Discriminate_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real_outputs, fake_outputs):
        return torch.mean((real_outputs - 1) ** 2 / 2) + torch.mean(fake_outputs ** 2 / 2)



# from __future__ import absolute_import, print_function
import torch
from torch import nn


def feature_map_permute(input):
    s = input.data.shape
    l = len(s)

    # permute feature channel to the last:
    # NxCxDxHxW --> NxDxHxW x C
    if l == 2:
        x = input # NxC
    elif l == 3:
        x = input.permute(0, 2, 1)
    elif l == 4:
        x = input.permute(0, 2, 3, 1)
    elif l == 5:
        x = input.permute(0, 2, 3, 4, 1)
    else:
        x = []
        print('wrong feature map size')
    x = x.contiguous()
    # NxDxHxW x C --> (NxDxHxW) x C
    x = x.view(-1, s[1])
    return x

class EntropyLoss(nn.Module):
    def __init__(self, eps = 1e-12):
        super(EntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, x):
        b = x * torch.log(x + self.eps)
        b = -1.0 * b.sum(dim=1)
        b = b.mean()
        return b

class EntropyLossEncap(nn.Module):
    def __init__(self, eps = 1e-12):
        super(EntropyLossEncap, self).__init__()
        self.eps = eps
        self.entropy_loss = EntropyLoss(eps)

    def forward(self, input):
        score = feature_map_permute(input)
        ent_loss_val = self.entropy_loss(score)
        return ent_loss_val

class Consistency_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, list_a, list_b):
        c = [torch.mean(torch.abs((a - b)** 2)) for a,b in zip(list_a,list_b)]
        return sum(c)/len(c)

class ThreeConsistency_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, list_a, list_b):
        c = [torch.mean(torch.abs((a - b)** 2)) for a,b in zip(list_a,list_b)]
        return sum(c)/len(c)

def FourConsistencyLoss(a,b,c,d):
    sim1 = torch.mean(torch.abs((a - b)** 2))
    sim2 = torch.mean(torch.abs((b - c)** 2))
    sim3 = torch.mean(torch.abs((c - d)** 2))
    return (sim1+sim3+sim2)/3

def Sim(B,a,b,c,d):
    sim1 = FourConsistencyLoss(a[:B,:],a[B:2*B,:],a[2*B:3*B,:],a[2*B:3*B,:])
    sim2 = FourConsistencyLoss(b[:B,:],b[B:2*B,:],b[2*B:3*B,:],b[2*B:3*B,:])
    sim3 = FourConsistencyLoss(c[:B,:],c[B:2*B,:],c[2*B:3*B,:],c[2*B:3*B,:])
    sim4 = FourConsistencyLoss(d[:B,:],d[B:2*B,:],d[2*B:3*B,:],d[2*B:3*B,:])
    return (sim1+sim3+sim2+sim4)/4

class CharbonnierLoss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(CharbonnierLoss, self).__init__()
        self.eps = 1e-7

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

if __name__ == '__main__':
    n = torch.rand(1)
    c = torch.rand(1)
    a =  [n,n,n,n]
    b =  [c,n,n,n]
    net = Consistency_Loss()
    res = net(a,b)
    print(res)
