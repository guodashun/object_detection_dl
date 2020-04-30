# For labelImg annotation to yolo training annotation
import cv2
import os
import xml.etree.ElementTree as ET
import random

input_path = "../data/11.21/marked"
output_path = "./data/custom"
train_rate = 0.7
pic_size = [1500, 1000]
classes_names = [i.strip('\n') for i in open(output_path + "/classes.names", 'r').readlines()]

# anno, train and validation
def write():
	raw_dataset_dir = os.listdir(input_path + "/imgs")
	train_list = open(output_path + '/train.txt', 'w')
	for i in raw_dataset_dir:
		anno_name = i.replace(".png", "")
		anno_convert(anno_name)
		train_list.write(output_path + "/images/" + str(i) + '\n')

	print("pls copy your imgs to \'data/custom/images\'!")


def anno_convert(anno_name):
    anno = ET.parse(input_path + "/annotation/" + anno_name + ".xml")
    anno_list = open(output_path + "/labels/" + anno_name + ".txt", "a")
    for obj in anno.iterfind('object'):
        anno_data = [] # idx, x, y, w, h
        anno_data.append(classes_names.index(obj.findtext('name')))
        obj = obj.find('bndbox')
        xmin = int(obj.findtext("xmin"))
        xmax = int(obj.findtext("xmax"))
        ymin = int(obj.findtext("ymin"))
        ymax = int(obj.findtext("ymax"))
        anno_data.append((xmin + xmax) / 2 / pic_size[0])
        anno_data.append((ymin + ymax) / 2 / pic_size[1])
        anno_data.append((xmax - xmin) / pic_size[0])
        anno_data.append((ymax - ymin) / pic_size[1])
        for i in anno_data:
            anno_list.write(str(i) + ' ')
        anno_list.write('\n')
        print("get dataset name :", anno_name) # for debug
    anno_list.close()


# separate train and validation
def separate():
	train_list = open(output_path + '/train.txt', 'r+')
	valid_list = open(output_path + '/valid.txt', 'w')
	train_size = raw_dataset_dir.size()
	dataset_size = train_size
	while(train_size > dataset_size * train_rate):
		train_set = train_list.readlines()
		idx = random.randint(0,train_size)
		valid_list.write(train_set[idx])
		train_list.writelines(line for i, line in enumerate(train_list) if i != idx)
		train_size = train_size - 1
	train_list.close()
	valid_list.close()

if __name__ == '__main__':
	write()
	# separate()
	# when you fisrt run separatem, pls backup your train_list!
	pass