import imageio
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from net.ECNet import Net
from utils.tdataloader import test_dataset
from utils.trainer import eval_mae, numpy2tensor
from torch.autograd import Variable
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./')


def get_IOU(maskpath, resultpath):
    # mask = get_array(maskpath)
    # result = get_array(resultpath)
    mask = maskpath
    result = resultpath
    # 计算iou
    S1 = 0  # 交集removal_train
    S2 = 0  # 并集
    for i in range(len(mask)):
        if mask[i] > 0.5 and result[i] > 0.5:  ##0~255为由黑到白，根据图片情况自行调整splicing
            S1 = S1 + 1
        if mask[i] > 0.5 or result[i] > 0.5:
            S2 = S2 + 1
        # if mask[i] < 100 and result[i] < 100:
        #     S3=S3+1
        # if mask[i] < 100 or result[i] < 100:
        #     S4=S4+1
    iou = S1 / S2
    f1 = (2 * S1) / (S1 + S2)
    # iouf=S3/S4
    # miou=(iou+iouf)/2
    return iou, f1


for _data_name in ['']:
    data_path = ''.format(_data_name)
    save_path = '.'.format(_data_name)
    opt = parser.parse_args()
    model = Net()
    model.load_state_dict(torch.load(opt.pth_path))

    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)

    image_root = '{}/f/'.format(data_path)
    gt_root = '{}/m/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    count = 0
    F1count = 0
    IOUcount = 0
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)

        image = image.cuda()
        clm, o1, o2, o3, o4, oe, e1, e2, e3, e4, feat1= model(image)
        # lateral_map_4= model(image)
        res=o1


        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        imageio.imwrite(save_path + name, (res * 255).astype(np.uint8))
