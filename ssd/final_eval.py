import cv2
import torch
import numpy as np
from ssd import build_ssd
from config import voc
from dataset import VOCDetection, VOCAnnotationTransform
from torch.autograd import Variable
from PIL import Image
from resnet import ResNetX, ResNetXS
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
dict_map = ["./model/local_model.pth", "./model/team_model.pth",
            "./model/id_model.pth", "./model/orien_model.pth"]
pic_dir = "./test_pic"

transform_test = transforms.Compose([
    transforms.CenterCrop(90),
    # transforms.ColorJitter(brightness=0.5, contrast=0.5)
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


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


def eval(net_map, testset, ssd_transform, resnet_transform):
    num_images = len(testset)
    for img_i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(img_i + 1, num_images))
        img = testset.pull_image(img_i)
        img_id, annotation = testset.pull_anno(img_i)
        x = torch.from_numpy(ssd_transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        y = net_map[0](x)
        detections = y.data
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.2:
                score = detections[0, i, j, 0]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                robot_img = img[(int(pt[1]) - 50):(int(pt[3]) + 50),
                                (int(pt[0]) - 50):(int(pt[2]) + 50)]
                robot_img = cv2.cvtColor(robot_img, cv2.COLOR_BGR2RGB)
                robot_img_ = Image.fromarray(robot_img)
                robot_tensor = resnet_transform(robot_img_)
                robot_tensor = transforms.ToTensor()(robot_tensor).unsqueeze(0).to(device)
                _, team = torch.max(net_map[1](robot_tensor).data, 1)
                _, robot_id = torch.max(net_map[2](robot_tensor).data, 1)
                orientation = net_map[3](robot_tensor).data.cpu().numpy()[0][0]

                cv2.rectangle(img,
                              (int(pt[0]), int(pt[1])),
                              (int(pt[2]), int(pt[3])),
                              COLORS[i % 3], 2)
                score = round(score.item(), 3)
                cv2.putText(img, "team: " + str(team.item()), (int(pt[0]), int(pt[1]) - 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                cv2.putText(img, "numb: " + str(robot_id.item()), (int(pt[0]), int(pt[1]) - 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                cv2.putText(img, "orie: " + str(orientation), (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                pred_num += 1
                j += 1
        cv2.imwrite(pic_dir + "/test_out/pic_" + str(img_i) + '.png', img)


if __name__ == '__main__':
    cfg = voc
    local_net = build_ssd('test', cfg['min_dim'], cfg['num_classes'])
    team_net = torchvision.models.resnet18(pretrained=True).to(device)
    id_net = ResNetX(18, True, 16).to(device)
    orien_net = ResNetXS(18, True).to(device)
    net_map = [local_net, team_net, id_net, orien_net]
    for i, net in enumerate(net_map):
        net.load_state_dict(torch.load(dict_map[i]))
        net.eval()
    testset = VOCDetection(pic_dir, None, VOCAnnotationTransform())
    eval(net_map, testset, BaseTransform(net_map[0].size, (104, 117, 123)), transform_test)
