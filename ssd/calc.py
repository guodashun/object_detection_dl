import xml.etree.ElementTree as ET
import os
from PIL import Image
import torch
import torch.utils.data as data
import cv2
import numpy as np

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor', 'robot')


class MultiRobotsDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(self.root, "imgs"))))
        self.boxes = list(sorted(os.listdir(os.path.join(self.root, "annotation"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "imgs", self.imgs[idx])
        box_path = os.path.join(self.root, "annotation", self.boxes[idx])
        img = Image.open(img_path).convert("RGB")
        image_id = torch.tensor([idx])
        boxes = []
        anno = ET.parse(box_path)
        for obj in anno.iterfind('object'):
            if obj.findtext('name') == "robot":
                obj = obj.iterfind('bndbox')
                xmin = obj.findtext("xmin")
                xmax = obj.findtext("xmax")
                ymin = obj.findtext("ymin")
                ymax = obj.findtext("ymax")
                boxes.append([xmin, xmax, ymin, ymax])

        target = {}
        target["image_id"] = image_id
        target["boxes"] = boxes

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def getX(self, idx):
        box_path = os.path.join(self.root, "annotation", self.boxes[idx])
        anno = ET.parse(box_path)
        i = 0
        x = []
        for obj in anno.iterfind('object'):
            if obj.findtext('name') == "robot":
                obj = obj.find('bndbox')
                xmin = obj.findtext("xmin")
                xmax = obj.findtext("xmax")
                ymin = obj.findtext("ymin")
                ymax = obj.findtext("ymax")
                x.append(int(xmax) - int(xmin))
                i += 1
        return i, x

    def __len__(self):
        return len(self.imgs)


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 #  image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 # image_sets=[('2019', 'trainval')],
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC19'):
        self.root = root
        # self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = list(sorted(os.listdir(os.path.join(self.root, "annotation"))))
        self._imgpath = list(sorted(os.listdir(os.path.join(self.root, "imgs"))))
        # self.ids = list()
        # for (year, name) in image_sets:
        #     rootpath = os.path.join(self.root, 'VOC' + year)
        #     for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
        #         self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self._imgpath)

    def pull_item(self, idx):
        target = ET.parse(os.path.join(self.root, "annotation", self._annopath[idx])).getroot()
        img = cv2.imread(os.path.join(self.root, "imgs", self._imgpath[idx]))
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, idx):
        return cv2.imread(os.path.join(self.root, "imgs", self._imgpath[idx]), cv2.IMREAD_COLOR)

    def pull_anno(self, idx):
        anno = ET.parse(os.path.join(self.root, "annotation", self._annopath[idx])).getroot()
        gt = self.target_transform(anno, 1, 1)
        return idx, gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


# test
if __name__ == '__main__':
    module = MultiRobotsDataset("./data/11.21/marked/", None)
    # _, target = module.__getitem__(3)
    # print(target["boxes"][0][0])
    # module = MultiRobotsDataset("./data/11.21/marked/")
    sum_x = 0
    amount = 0
    best_x = 0
    worst_x = 9999
    for i in range(len(module)):
        amount_i, sum_x_i = module.getX(i)
        amount += amount_i
        for x in sum_x_i:
            sum_x += x
            if x > best_x:
                best_x = x
            if x < worst_x:
                worst_x = x
    avg = sum_x / amount
    fang_sum = 0
    for i in range(len(module)):
        _, sum_x_i = module.getX(i)
        for x in sum_x_i:
            fang_sum += pow(x - avg, 2)
    fang = fang_sum / amount
    print("avg:", avg, "fang:", fang)
    print("amount:", amount)
    print("best_x:", best_x, "worst_x:", worst_x)
    # print(module._annopath[150], module._imgpath[150])
