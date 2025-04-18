import torch
from torch.autograd import Variable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from datetime import datetime
from net.ECNet  import Net
from utils.tdataloader1 import get_loader
from utils.utils import clip_gradient, AvgMeter, poly_lr
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import StepLR
from net.utils import AverageMeter, batch_intersection_union, write_logger, set_random_seed
import net.utils as utils
file = open("log/BGNet.txt", "a")
torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
np.random.seed(2021)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda")

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()


def train(train_loader, model, optimizer, epoch):
    model.train()

    loss_record3, loss_record2, loss_record1, loss_recorde,loss_recorde4,loss_clm ,loss_cl1,loss_cl2,loss_cl3= AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        # ---- data prepare ----
        images, gts, edges3,edges2,edges1 = pack
        # images, gts, edges3,edges2,edges1 = pack
        images = Variable(images).cuda()
        gts = Variable(gts.unsqueeze(1)).cuda()
        # gts = Variable(gts).cuda()
        edges3 = Variable(edges3).cuda()
        edges1 = Variable(edges1).cuda()
        # ---- forward ----
        clm, o1, o2, o3, o4, oe, e1, e2, e3, e4, feat1 = model(images)

        # ---- loss function ----
        loss4 = structure_loss(o1, gts)
        loss3 = structure_loss(o2, gts)
        loss2 = structure_loss(o3, gts)
        loss1 = structure_loss(o4, gts)
        lossclm = structure_loss(clm, gts)
        lossoe = dice_loss(oe, edges3)
        losse1= dice_loss(e1,  edges1)
        losse2 = dice_loss(e2, edges1)
        losse3 = dice_loss(e3, edges1)
        losse4 = dice_loss(e4, edges1)

        #
        feat3 = F.interpolate(feat1, [64,64], mode='bilinear', align_corners=True)


        cfeature3 = F.avg_pool2d(feat3, kernel_size=4, stride=4)
        Ba3, Ch3, _, _ = cfeature3.shape
        cfeature3 = cfeature3.view(Ba3, Ch3, -1)
        cfeature3 = torch.transpose(cfeature3, 1, 2)
        cfeature3= F.normalize(cfeature3, dim=-1)

        mask_con3 =F.interpolate(gts, [256,256], mode='bilinear', align_corners=True)
        mask_con3 = F.avg_pool2d(mask_con3, kernel_size=4, stride=4)
        mask_con3 = (mask_con3 > 0.5).int().float()
        mask_con3 = mask_con3.view(Ba3, -1)
        mask_con3 = mask_con3.unsqueeze(dim=1)

        contrast_temperature = 0.1


        c_loss3 = utils.square_patch_contrast_loss(cfeature3, mask_con3, device, contrast_temperature)

        c_loss3 = c_loss3.mean(dim=-1)
        c_loss3 = c_loss3.mean()


        loss = (loss4+ loss3 + loss3 + loss2 + loss1) +lossclm+lossoe+(losse1+losse2+losse3+losse4)+(c_loss3)

        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        # ---- recording loss ----

        # ---- train visualization ----
        if i % 60 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-3: {:.4f}], [lateral-2: {:.4f}], [lateral-1: {:.4f}], [edge: {:,.4f}],[edge4: {:,.4f}],  [clm: {:,.4f}], [cl1: {:,.4f}], [cl2: {:,.4f}], [cl3: {:,.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record3.avg, loss_record2.avg, loss_record1.avg, loss_recorde.avg,loss_recorde4.avg, loss_clm.avg, loss_cl1.avg, loss_cl2.avg, loss_cl3.avg))
            file.write('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                       '[lateral-3: {:.4f}], [lateral-2: {:.4f}], [lateral-1: {:.4f}], [edge: {:,.4f}]\n'.
                       format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record3.avg, loss_record2.avg, loss_record1.avg, loss_recorde.avg))

    save_path = 'checkpoints/{}/'.format(opt.train_save)
    # scheduler.step()
    os.makedirs(save_path, exist_ok=True)
    if (epoch + 1) % 2 == 0 or (epoch + 1) == opt.epoch:
        torch.save(model.state_dict(), save_path + 'ECNet-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'ECNet-%d.pth' % epoch)
        file.write('[Saving Snapshot:]' + save_path + 'ECNet-%d.pth' % epoch + '\n')
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=200, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=4, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--train_path', type=str,
                        default='', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='')
    opt = parser.parse_args()

    # ---- build models ----
    model = Net().cuda()

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)
    # scheduler = StepLR(optimizer, step_size=20, gamma=0.8)
    image_root = '{}/f/'.format(opt.train_path)
    gt_root = '{}/m/'.format(opt.train_path)
    edge_root = '{}/e/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print("Start Training")

    for epoch in range(opt.epoch):
        poly_lr(optimizer, opt.lr, epoch, opt.epoch)
        train(train_loader, model, optimizer, epoch)

    file.close()
