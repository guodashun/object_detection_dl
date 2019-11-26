import xml.etree.ElementTree as ET
import os
from PIL import Image
import torch


class MultiRobotsDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(self.root, "imgs"))))
        self.boxes = list(sorted(os.listdir(os.path.join(self.root, "annotation_robot"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "imgs", self.imgs[idx])
        box_path = os.path.join(self.root, "annotation_robot", self.boxes[idx])
        img = Image.open(img_path).convert("RGB")
        image_id = torch.tensor([idx])
        boxes = []
        anno = ET.parse(box_path)
        for obj in anno.iterfind('object'):
            if obj.findtext('name') == "robot":
                xmin = obj.findtext("bndbox/xmin")
                xmax = obj.findtext("bndbox/xmax")
                ymin = obj.findtext("bndbox/ymin")
                ymax = obj.findtext("bndbox/ymax")
                boxes.append([xmin, xmax, ymin, ymax])

        target = {}
        target["image_id"] = image_id
        target["boxes"] = boxes

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


# test
if __name__ == '__main__':
    module = MultiRobotsDataset("./data/11.21/marked/", None)
    _, target = module.__getitem__(3)
    print(target["boxes"][0][0])
