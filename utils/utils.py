import torch
import numpy as np
# from thop import profile
# from thop import clever_format

def square_patch_contrast_loss(feat, mask, device, temperature=0.6):
    # feat shape should be (Batch, Total_Patch_number, Feature_dimension)
    # mask should be (Batch, H, W)

    mem_mask = torch.eq(mask, mask.transpose(1, 2)).float()
    mem_mask_neg = torch.add(torch.negative(mem_mask), 1)

    feat_logits = torch.div(torch.matmul(feat, feat.transpose(1, 2)), temperature)
    identity = torch.eye(feat_logits.shape[-1]).to(device)
    neg_identity = torch.add(torch.negative(identity), 1).detach()

    feat_logits = torch.mul(feat_logits, neg_identity)

    feat_logits_max, _ = torch.max(feat_logits, dim=1, keepdim=True)
    feat_logits = feat_logits - feat_logits_max.detach()

    feat_logits = torch.exp(feat_logits)

    neg_sum = torch.sum(torch.mul(feat_logits, mem_mask_neg), dim=-1)
    denominator = torch.add(feat_logits, neg_sum.unsqueeze(dim=-1))
    division = torch.div(feat_logits, denominator + 1e-18)

    loss_matrix = -torch.log(division + 1e-18)
    loss_matrix = torch.mul(loss_matrix, mem_mask)
    loss_matrix = torch.mul(loss_matrix, neg_identity)
    loss = torch.sum(loss_matrix, dim=-1)

    loss = torch.div(loss, mem_mask.sum(dim=-1) - 1 + 1e-18)

    return loss


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay


def poly_lr(optimizer, init_lr, curr_iter, max_iter, power=0.9):
    lr = init_lr * (1 - float(curr_iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def warmup_poly(optimizer, init_lr, curr_iter, max_iter):
    warm_start_lr = 1e-7
    warm_steps = 1000

    if curr_iter<= warm_steps:
        warm_factor = (init_lr / warm_start_lr) ** (1 / warm_steps)
        warm_lr = warm_start_lr * warm_factor ** curr_iter
        for param_group in optimizer.param_groups:
            param_group['lr'] = warm_lr
    else:
        lr = init_lr * (1 - (curr_iter - warm_steps) / (max_iter - warm_steps)) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


# def CalParams(model, input_tensor):
#     """
#     Usage:
#         Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
#     Necessarity:
#         from thop import profile
#         from thop import clever_format
#     :param model:
#     :param input_tensor:
#     :return:
#     """
#     flops, params = profile(model, inputs=(input_tensor,))
#     flops, params = clever_format([flops, params], "%.3f")
#     print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))