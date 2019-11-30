import cv2
import torch
import numpy as np
from ssd import build_ssd
from config import voc
from dataset import VOCDetection, VOCAnnotationTransform
from torch.autograd import Variable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels


def eval(net, testset, transform):
    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i + 1, num_images))
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        # x.to(device)

        y = net(x)
        detections = y.data
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                # if pred_num == 0:
                #     with open(filename, mode='a') as f:
                #         f.write('PREDICTIONS: '+'\n')
                score = detections[0, i, j, 0]
                # label_name = labelmap[i - 1]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                cv2.imread(testset._imgpath[0])
                cv2.rectangle(img,
                              (int(pt[0]), int(pt[1])),
                              (int(pt[2]), int(pt[3])),
                              COLORS[i % 3], 2)
                cv2.imwrite(testset._imgpath[0] + 'new.png', img)
                # print('wzdebug:', coords)
                # cv2.imshow('img', img)
                pred_num += 1
                # with open(filename, mode='a') as f:
                #     f.write(str(pred_num)+' label: '+label_name+' score: ' +
                #             str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
                j += 1


if __name__ == '__main__':
    cfg = voc
    net = build_ssd('test', cfg['min_dim'], cfg['num_classes'])
    net.load_state_dict(torch.load('weights/ssd300_VOC_2100.pth'))
    net.eval()
    testset = VOCDetection('./data/11.21/test', None, VOCAnnotationTransform())
    # net = net.to(device)
    # img = cv2.imread('./data/11.21/raw/1/Acquisition-19060477-200.png')
    # img = torch.from_numpy(img.transpose((2, 0, 1))).float().unsqueeze(0).to(device)
    eval(net, testset, BaseTransform(net.size, (104, 117, 123)))
