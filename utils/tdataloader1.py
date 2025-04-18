import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import random
from scipy.ndimage import distance_transform_edt
random.seed(2021)
import torch
import PIL
import albumentations as A
class CamObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, edge_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.edges = sorted(self.edges)
        self.filter_files()
        self.size = len(self.images)
        self.kernel = np.ones((5, 5), np.uint8)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.ge_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.ge_transform1 = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize))])
        self.ge_transform2 = transforms.Compose([
            transforms.ToTensor()])
        self.label_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize))])
        self.as_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])
        self.as_tensor_2 = transforms.Compose([
            transforms.ToTensor()

        ])

        self.transform1 = A.Compose([
            A.Resize(self.trainsize, self.trainsize),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ])
        self.transform_L = A.Compose([
            A.Resize(int(self.trainsize), int(self.trainsize)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ])



    def getFlip(self):
        p = 0.5
        self.flip = transforms.RandomHorizontalFlip(p)
    def __getitem__(self, index):
        self.getFlip()
        image_ori = cv2.imread(self.images[index], cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        label = cv2.imread(self.gts[index], cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        edge = cv2.imread(self.edges[index], cv2.IMREAD_GRAYSCALE)
        ycbcr_image = cv2.cvtColor(image_ori, cv2.COLOR_RGB2YCrCb)
        # gt = self.binary_loader(self.gts[index])
        # gt = self.flip(gt)
        # gt = self.ge_transform(gt)
        # edge = cv2.dilate(edge, self.kernel, iterations=1)
        # edge = Image.fromarray(edge)
        # edge = self.flip(edge)
        # edge = self.ge_transform(edge)
        label = np.float32(label > 128)
        # edge = np.float32(edge > 128)
        # if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
        #     raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        # edge = self.flip(edge)
        # label= self.flip(label)
        # image=self.flip(image)
        # edge = cv2.dilate(edge, self.kernel, iterations=1)
        # edge = Image.fromarray(edge)
        augmented1 = self.transform1(image=image, mask=label)
        augmented2 = self.transform_L(image=ycbcr_image, mask=label)


        image_M, label = augmented1['image'], augmented1['mask']
        ycbcr_image, label_L = augmented2['image'], augmented2['mask']


        # edge = self.ge_transform(edge)
        ibdm4 = self.distribution_map(label, sigma=5)
        ibdm3 = self.distribution_map(label, sigma=4)
        ibdm2 = self.distribution_map(label, sigma=0.8)
        ibdm1 = self.distribution_map(label, sigma=1)
        ibdm0 = self.distribution_map(label, sigma=1)
        return self.as_tensor(image_M),  \
               torch.tensor(label, dtype=torch.float), \
               torch.tensor(ibdm3, dtype=torch.float), torch.tensor(ibdm2, dtype=torch.float),torch.tensor(ibdm1, dtype=torch.float)

    # def __getitem__(self, index):
    #     self.getFlip()
    #     image = self.rgb_loader(self.images[index])
    #     image1=image
    #     gt = self.binary_loader(self.gts[index])
    #     gt1 = cv2.imread(self.gts[index], cv2.IMREAD_GRAYSCALE)
    #     edge = cv2.imread(self.edges[index], cv2.IMREAD_GRAYSCALE)
    #
    #     image = self.flip(image)
    #     image = self.img_transform(image)
    #     gt = self.flip(gt)
    #     gt = self.ge_transform(gt)
    #     gt1 = cv2.dilate(gt1, self.kernel, iterations=1)
    #     gt1 = Image.fromarray(gt1)
    #     gt1 = self.flip(gt1)
    #     gt1 = self.ge_transform1(gt1)
    #     ibdm = self.distribution_map(gt1, sigma=10)
    #     gt1 = self.ge_transform2(gt1)
    #
    #     edge = cv2.dilate(edge, self.kernel, iterations=1)
    #     edge = Image.fromarray(edge)
    #     edge = self.flip(edge)
    #     edge = self.ge_transform(edge)
    #     # label = cv2.imread(self.gts[index], cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
    #     # label = np.float32(label > 128)
    #     # label = self.ge_transform(label)
    #     #
    #     ibdm=torch.tensor(ibdm, dtype=torch.float)
    #     return image, gt, edge,ibdm

    def distribution_map(self, mask, sigma):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 消除标注的问题孤立点

        dist1 = distance_transform_edt(mask)
        dist2 = distance_transform_edt(1 - mask)
        dist = dist1 + dist2
        dist = dist - 1

        f = lambda x, sigma: 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-x ** 2 / (2 * sigma ** 2))

        bdm = f(dist, sigma)

        bdm[bdm < 0] = 0

        return bdm * (sigma ** 2)
    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        edges = []
        for img_path, gt_path, edge_path in zip(self.images, self.gts, self.edges):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            edge = Image.open(edge_path)
            if img.size == gt.size and img.size == edge.size:
                images.append(img_path)
                gts.append(gt_path)
                edges.append(edge_path)
        self.images = images
        self.gts = gts
        self.edges = edges

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


class test_dataset:
    """load test dataset (batchsize=1)"""
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.transform2 = transforms.Compose([
            transforms.Resize((int(self.testsize), int(self.testsize))),
            transforms.ToTensor()
            ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0
        self.as_tensor_2 = transforms.Compose([
            transforms.ToTensor()

        ])
        self.transform_L = A.Compose([
            A.Resize(self.testsize, self.testsize),

        ])

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        # ycbcr_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        image = self.transform(image).unsqueeze(0)
        # image_ori = cv2.imread(self.images[self.index], cv2.IMREAD_COLOR)
        ycbcr_image =  self.DCT_loader(self.images[self.index])
        ycbcr_image=self.transform2(ycbcr_image).unsqueeze(0)
        # ycbcr_image=as_tensor_2
        # ycbcr_image=self.transform2(ycbcr_image)
        # augmented2 = self.transform_L(image=ycbcr_image)
        #
        #
        # ycbcr_image= augmented2['image']
        # ycbcr_image=PIL.Image.fromarray(ycbcr_image)
        # ycbcr_image = self.transform2(ycbcr_image).unsqueeze(0)
        # image2 = self.transform2(iamge1).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, ycbcr_image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def DCT_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('YCbCr')


class test_loader_faster(data.Dataset):
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.size = len(self.images)

    def __getitem__(self, index):
        images = self.rgb_loader(self.images[index])
        images = self.transform(images)

        img_name_list = self.images[index]

        return images, img_name_list

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, edge_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True):
    dataset = CamObjDataset(image_root, gt_root, edge_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

    return data_loader
