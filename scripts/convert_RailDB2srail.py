#! python3
# -*- encoding: utf-8 -*-

import os
import cv2
import numpy as np
import pandas as pd
import json
import random
from PIL import Image
from data.constant import RailDB_row_anchor as row_anchor

def find_closest_row_anchor( img):

        img_array = np.array(img)

        min_y = np.argmax(img_array.sum(axis=1) > 0) / img.size[1] * 288

        closest_row_anchors = [row for row in row_anchor if row[0] <= min_y]

        if not closest_row_anchors:
            
            return row_anchor[0], 0

        closest_row_anchor = min(closest_row_anchors, key=lambda x: abs(x[0] - min_y))


        index_of_closest_row_anchor = row_anchor.index(closest_row_anchor)

        return closest_row_anchor, index_of_closest_row_anchor

def json2gt_txt(path_json):  
    with open(path_json, 'r') as path_json:
        jsonx = json.load(path_json)
    
        labelarray = [0,0,0,0]
        for shape in jsonx['shapes']:
            label = int(shape['label'])
            
            if label == 1:
                labelarray[0] = 1
            if label == 2:
                labelarray[1] = 1
            if label == 3:
                labelarray[2] = 1
            if label == 4:
                labelarray[3] = 1

        labelstr = str(labelarray[0])+ " " + str(labelarray[1])+ " " + str(labelarray[2])+ " " + str(labelarray[3])
    return labelstr

def generate_equidistant_points(label_image_path, num_points=18):

    label_image = cv2.imread(label_image_path, cv2.IMREAD_GRAYSCALE)
    row_anchor, _ = find_closest_row_anchor(Image.open(label_image_path))
    

    height, width = label_image.shape
    if height != 288:
        scale_f = lambda x : int((x * 1.0/288) * height)


        row_anchor = list(map(scale_f,row_anchor)) # 

    sorted_row_anchor = sorted(row_anchor, reverse=True)


    label_points_list = []

    for color in np.unique(label_image)[1:5]: 

        indices = np.argwhere(label_image == color) 

        result = []
        for row in sorted_row_anchor:
            matching_rows = indices[indices[:, 0] == row]
            
            if len(matching_rows) > 0:
                tmp = (int(sum(matching_rows[:,1]) / len(matching_rows)),row)
                
            else:
                tmp = (-1,row)
            result.append(tmp)

        label_points_list.append(result)
        
    return label_points_list

def batch_process_labels(label_dir, output_dir, num_points=18):

    os.makedirs(output_dir, exist_ok=True)


    for sub_dir_name in os.listdir(label_dir):
        
        for filename in os.listdir(os.path.join(label_dir, sub_dir_name)):
            if filename.endswith(".png"):
                label_image_path = os.path.join(label_dir, sub_dir_name, filename)
                

                points_list = generate_equidistant_points(label_image_path, num_points)

                output_file_path = os.path.join(output_dir, sub_dir_name, f"{os.path.splitext(filename)[0]}.lines.txt")

                with open(output_file_path, 'w') as file:
                    for points in points_list:

                        line = " ".join([f"{x} {y}" for x, y in points])
                        file.write(f"{line}\n")

def generate_list(root_dir, meta_file, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir,"test_split"), exist_ok=True)

   

    pd_reader = pd.read_csv(meta_file)
    random.seed(2022)

    rd_ind = list(range(len(pd_reader)))
    random.shuffle(rd_ind)
    pd_reader = pd_reader.reindex(index=rd_ind)


    len_image_train = int(len(pd_reader)*0.8)
    pd_reader_train = pd_reader.iloc[:len_image_train]
    img_list_train = list(pd_reader_train['name'])

    pd_reader_test = pd_reader.iloc[len_image_train:]
    img_list_test = list(pd_reader_test['name'])

    with open(os.path.join(output_dir,'train_gt.txt'), 'w') as ftxt:
        for img_name in img_list_train:

    
            jpeg_name = 'pic/' + img_name[:-12] + '/' + img_name
            label_name = 'mask/' + img_name[:-12] + '/' + img_name.replace('jpeg', 'png')
            json_name = 'anno/' + img_name[:-12] + '/' + img_name.replace('jpeg', 'json')
            labelstr = json2gt_txt(os.path.join(root_dir,json_name))

            ftxt.writelines(jpeg_name +' '+  label_name + ' ' + labelstr + "\n")

    with open(os.path.join(output_dir,'test.txt'), 'w') as ftxt:
        for img_name in img_list_test:
            # print(img_name)
    
            jpeg_name = 'pic/' + img_name[:-12] + '/' + img_name
            label_name = 'mask/' + img_name[:-12] + '/' + img_name.replace('jpeg', 'png')
            json_name = 'anno/' + img_name[:-12] + '/' + img_name.replace('jpeg', 'json')
            labelstr = json2gt_txt(os.path.join(root_dir,json_name))

            ftxt.writelines(jpeg_name +' '+  label_name + ' ' + labelstr + "\n")


    for name in ['sun', 'rain',	'night','line',	'cross','curve','slope','near',	'far']: 
        
        img_list = list(pd_reader_test['name'][pd_reader_test[name].astype(bool)])
        
        
        with open(os.path.join(output_dir,"test_split",name+'.txt'), 'w') as ftxt:
            for img_name in img_list:
            
        
                jpeg_name = 'pic/' + img_name[:-12] + '/' + img_name

                ftxt.writelines(jpeg_name+"\n")

root_dir = "../SynData/RailDB/"
label_dir = os.path.join(root_dir,"mask") 
output_dir = os.path.join(root_dir,"pic")  
meta_file =  root_dir+"meta.csv"
output_list_dir= os.path.join(root_dir,"list")   
# convert label_img to json（18 points）
batch_process_labels(label_dir, output_dir)


generate_list(root_dir, meta_file, output_list_dir)

