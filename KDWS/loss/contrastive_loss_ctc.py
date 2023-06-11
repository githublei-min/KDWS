import torch.nn as nn
import torch

class CTC(nn.Module):  # myself feature extractor:Conv k3s1-Tanh-Conv k3s1
    """
    A randomly initialized two-layer feature extractor,which has an architecture of Conv k3s1-Tanh-Conv k3s1.
    Inspiration comes from paper "Residual Local Feature Network for Efficient Super-Resolution"
    Reference: https://github.com/Booooooooooo/CSD/blob/main/PyTorch%20version/loss/contrast_loss.py
    """
    def __init__(self):
        super(CTC, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1),
            nn.Tanh(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1),
        )
        #----------kaiming initialization-------------#
        self.apply(self.init_weights)

    def init_weights(self, m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.extractor(x)
        return x

class CTC_Extractor(nn.Module):
    def __init__(self, requires_grad=False):
        super(CTC_Extractor, self).__init__()
        CTC_random_features = CTC().extractor
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), CTC_random_features[x])
        self.slice2.add_module(str(2), CTC_random_features[2])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_tanh1 = self.slice1(x)
        h2 = self.slice2(h_tanh1)
        out = [h_tanh1, h2]
        return out

class ContrastLoss(nn.Module):
    """
     Reference: https://github.com/Booooooooooo/CSD/blob/main/PyTorch%20version/loss/contrast_loss.py
    """
    def __init__(self, args):
        super(ContrastLoss, self).__init__()
        self.args = args
        self.ctc = CTC_Extractor().to(
            torch.device('cpu' if self.args.cpu else 'cuda'))
        self.l1 = nn.L1Loss().to(
            torch.device('cpu' if self.args.cpu else 'cuda'))

    def forward(self, pos, anchor, neg):
        anchor_ctc, pos_ctc, neg_ctc = self.ctc(anchor), self.ctc(pos), self.ctc(neg)
        return self.L1_forward(pos_ctc, anchor_ctc, neg_ctc)

    def L1_forward(self, pos, anchor, neg):
        loss = 0
        for i in range(len(pos)):
            neg_i = neg[i].unsqueeze(0)
            neg_i = neg_i.repeat(anchor[i].shape[0], 1, 1, 1, 1)
            neg_i = neg_i.permute(1, 0, 2, 3, 4) ### batchsize*negnum*color*patchsize*patchsize

            d_ts = self.l1(pos[i], anchor[i])
            d_sn = torch.mean(torch.abs(neg_i.detach() - anchor[i]).sum(0))

            contrastive = d_ts / (d_sn + 1e-7)
            loss += contrastive
        return loss

# if __name__ == '__main__':
#     sr = torch.rand((8, 3, 256, 256))
#     sr = sr.to(torch.device('cuda:0'))
#     hr = torch.rand((8, 3, 256, 256))
#     hr = hr.to(torch.device('cuda:0'))
#     neg = torch.rand((4, 4, 32, 32))
#     up = torch.nn.Upsample(scale_factor=8, mode='bilinear')
#     neg = up(neg)
#     neg = nn.Conv2d(neg.shape[-3], 3, 3, 1, 1)(neg)
#     print(neg.shape)
#     neg = neg.to(torch.device('cuda:0'))
#     print(neg.dtype)
#
#     from src.option import args
#     loss = ContrastLoss(args)
#     loss = loss(sr, hr, neg)
#     print(loss)

