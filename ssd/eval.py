import cv2
import torch
import numpy as np
from ssd import build_ssd
from config import voc
from dataset import VOCDetection, VOCAnnotationTransform
from torch.autograd import Variable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
weight_num = ""
weight_name = "weights/" + weight_num + "/ssd300_VOC_"
pic_name = ""


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
            while detections[0, i, j, 0] >= 0.2:
                score = detections[0, i, j, 0]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                cv2.rectangle(img,
                              (int(pt[0]), int(pt[1])),
                              (int(pt[2]), int(pt[3])),
                              COLORS[i % 3], 2)
                score = round(score.item(), 3)
                cv2.putText(img, str(score), (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)
                cv2.imwrite(pic_name.replace(weight_num, "pic") + '.png', img)
                pred_num += 1
                j += 1


if __name__ == '__main__':
    cfg = voc
    net = build_ssd('test', cfg['min_dim'], cfg['num_classes'])

    for i in range(5):
        pic_name = weight_name + str((i + 1) * 4000) + ".pth"
        print("processing", pic_name)
        net.load_state_dict(torch.load(pic_name))
        net.eval()
        testset = VOCDetection('./data/11.21/test', None, VOCAnnotationTransform())
        eval(net, testset, BaseTransform(net.size, (104, 117, 123)))
