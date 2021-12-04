import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

crop_size = 35
IMAGE_HEIGHT = int(1936/4)
IMAGE_WIDTH = int(1216/4)-crop_size

colorB = [0, 0, 255, 255, 69]
colorG = [0, 0, 255, 0, 47]
colorR = [0, 255, 0, 0, 142]
CLS=5
CLASS_COLOR = list()
for i in range(0, CLS):
    CLASS_COLOR.append([colorR[i], colorG[i], colorB[i]])
COLORS = np.array(CLASS_COLOR, dtype="float32")
COLORS = torch.from_numpy(COLORS)

class KLDivergenceLoss(nn.Module):
    def __init__(self, T=0.01, alpha=0.6):
        super(KLDivergenceLoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.L1 = nn.L1Loss()
        self.T = T
        self.alpha = alpha
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def give_color_to_seg_img(self, seg, n_classes):
        if len(seg.shape)==3:
            seg = seg[:,:,0]
        seg_img = torch.zeros((seg.shape[0],seg.shape[1],3)).to(torch.float).to(self.device)
        #colors = sns.color_palette("hls", n_classes) #DB
        colors = COLORS #DB
        for c in range(n_classes):
            segc = (seg == c)
            seg_img[:,:,0] += (segc*( colors[c][0]/255.0 ))
            seg_img[:,:,1] += (segc*( colors[c][1]/255.0 ))
            seg_img[:,:,2] += (segc*( colors[c][2]/255.0 ))

        return seg_img
        
    def forward(self, logits, target, thresh = 100):

        log2div = logits.flatten(start_dim=1).clamp(-thresh, thresh).sigmoid()
        tar2div = target.flatten(start_dim=1).clamp(-thresh, thresh).sigmoid().detach()
        kldiv_loss = nn.KLDivLoss(reduction="batchmean")(F.log_softmax((log2div / self.T), dim = 1), F.softmax((tar2div / self.T), dim = 1))*(self.alpha * self.T * self.T)
        
        
        # label_loss
        logits_ = torch.argmax(logits, dim=1).to(torch.uint8).reshape(4, IMAGE_WIDTH, IMAGE_HEIGHT)
        logits_ = self.give_color_to_seg_img(logits_, n_classes=CLS)
        target_ = torch.argmax(target, dim=1).to(torch.uint8).reshape(4, IMAGE_WIDTH, IMAGE_HEIGHT)
        target_ = self.give_color_to_seg_img(target_, n_classes=CLS)
        # loss for distillation
        label_loss = self.criterion(logits_, target_)
        
        
        return kldiv_loss + label_loss

