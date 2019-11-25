import xml.etree.ElementTree as ET
import os

tag_path = "./data/11.21/marked/robot"
tags = os.listdir(tag_path)

for tag_dir in tags:
    tag = ET.parse(tag_path + "/" + tag_dir)
    cnt = 0
    for obj in tag.iterfind('object'):
        if obj.findtext('name') == "robot":
            cnt = cnt + 1
            print("The %dth robot is " % (cnt), obj.findtext("bndbox/xmin"))
