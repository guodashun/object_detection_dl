import os
import torch.utils.data as data
from PIL import Image
import numpy as np
import math


class RobotTeamClassification(data.Dataset):
    def __init__(self, root, transform=None, phase="train", item=None):
        self.root = root
        self.transform = transform
        if phase == "train" or phase == "test":
            self.annopath = os.path.join(self.root, phase + ".txt")
        else:
            print("ERROR: Phase: " + phase + " not recognized")
            return
        if item != "team" and item != "orientation" and item != "number":
            print("ERROR: Item: " + item + " not recognized")
            return
        self.phase = phase
        self.item = item
        # self._annopath = list(sorted(os.listdir(os.path.join(self.root, "annotation"))))
        self._imgpath = list(sorted(os.listdir(os.path.join(self.root, self.phase + "_set"))))
        # self.anno = [np.loadtxt(i) for i in open(self.annopath).readlines()]
        self.anno = np.loadtxt(self.annopath)

    def __getitem__(self, idx):
        # img = cv2.imread(os.path.join(self.root, self.phase + "_set", self._imgpath[idx]))
        img = Image.open(os.path.join(self.root, self.phase + "_set", self._imgpath[idx])).convert("RGB")
        # anno = ET.parse(os.path.join(self.root, "annotation", self._annopath[idx])).getroot()
        # obj = anno.iterfind("object")
        # target = obj.findtext("team")
        if self.item == "team":
            target = self.anno[idx][0]
        elif self.item == "orientation":
            target = self.anno[idx][1] / math.pi * 180
        elif self.item == "number":
            target = self.anno[idx][2]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self._imgpath)


if __name__ == '__main__':
    test = RobotTeamClassification(root="/home/zjunlict-vision-1/luckky/dl/resnet/data", item="team")
    print(test.__getitem__(7))
