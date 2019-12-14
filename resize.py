import cv2
import os
import xml.etree.ElementTree as ET


input_path = "./data/11.21/test"
output_path = "./data/11.21/test_out"
pic_height = 1000
pic_width = 1500
resize_step = 25
class_name = "robot"


def raw2Field():
    imgs = os.listdir(input_path)
    for img_dir in imgs:
        img = cv2.imread(input_path + "/" + img_dir)
        # print("img size:", img.shape)
        cropped = img[550:1550, 400:1900]
        # print("cropped size:", cropped.shape)
        cv2.imwrite(output_path + "/" + img_dir, cropped)


def big2Small():
    imgs = os.listdir(input_path + "/imgs")
    for img_dir in imgs:
        img = cv2.imread(input_path + "/imgs" + "/" + img_dir)
        anno_dir = img_dir.replace(".png", ".xml")
        anno_path = os.path.join(input_path, "annotation", anno_dir)
        anno = ET.parse(anno_path)
        boxes = []
        for obj in anno.iterfind('object'):
            if obj.findtext('name') == class_name:
                obj = obj.find('bndbox')
                xmin = obj.findtext("xmin")
                xmax = obj.findtext("xmax")
                ymin = obj.findtext("ymin")
                ymax = obj.findtext("ymax")
                boxes.append([xmin, xmax, ymin, ymax])
        # print("img size:", img.shape)
        gf = ["xmin", "xmax", "ymin", "ymax"]
        anno_tpl = anno
        obj_tpl = anno.find("object")
        objs = anno_tpl.findall("object")
        anno_tpl_root = anno_tpl.getroot()
        for obj in objs:
            anno_tpl_root.remove(obj)
        anno_tpl.write(os.path.join(output_path, "annotation/" + "template.xml"))
        for i, box in enumerate(boxes):
            diff = [0] * 4
            box = list(map(int, box))
            diff[0] = min(resize_step, box[0])
            diff[1] = min(pic_width - box[1], resize_step)
            diff[2] = min(resize_step, box[2])
            diff[3] = min(pic_height - box[3], resize_step)
            # print(img.shape) # sha bi CV2
            # print(max(0, box[0] - resize_step), min(pic_width, box[1] + resize_step), max(0, box[2] - resize_step), min(pic_height, box[3] + resize_step))
            cropped = img[max(0, box[2] - resize_step):min(pic_height, box[3] + resize_step),
                          max(0, box[0] - resize_step):min(pic_width, box[1] + resize_step)]
            new_anno = ET.parse(os.path.join(output_path, "annotation/" + "template.xml"))
            new_anno_root = new_anno.getroot()
            new_anno_root.append(obj_tpl)
            if new_anno.findtext('object/name') == class_name:
                bndbox = new_anno.find('object/bndbox')
                new_text = bndbox.find(gf[0])
                new_text.text = str(0 + diff[0])
                new_text = bndbox.find(gf[1])
                new_text.text = str(cropped.shape[0] - diff[1])
                new_text = bndbox.find(gf[2])
                new_text.text = str(0 + diff[2])
                new_text = bndbox.find(gf[3])
                new_text.text = str(cropped.shape[1] - diff[3])
            # print(cropped.shape)
            # print(os.path.join(output_path, "imgs/" + img_dir.replace(".png", "") + "-" + str(i) + ".png"))
            # print(os.path.join(output_path, "annotation/" + anno_dir.replace(".xml", "") + "-" + str(i) + ".xml"))
            cv2.imwrite(os.path.join(output_path, "imgs/" + img_dir.replace(".png", "") + "-" + str(i) + ".png"), cropped)
            new_anno.write(os.path.join(output_path, "annotation/" + anno_dir.replace(".xml", "") + "-" + str(i) + ".xml"))

        # print("cropped size:", cropped.shape)
        # cv2.imwrite(output_path + "/" + img_dir, cropped)


if __name__ == '__main__':
    # raw2Field()
    big2Small()
    pass
