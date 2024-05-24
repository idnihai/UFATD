#! python3
# -*- encoding: utf-8 -*-

import os
import numpy as np
import concurrent.futures


def list_txt_files(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                file_paths.append(os.path.join(root, file))
    return file_paths



image_h = 1080
# while training resize the image to 288x800，288 is height
resize_h = 288
step = 70
step_in = 5
num_classes = 3
# 
# folder_path = "../SynData/culane/" 
# folder_path = "../SynData/DL-Rail/videos/"  
# folder_path = "../SynData/RailDB/pic/" 
folder_path = "../SynData/S-Rail/images_dir/"  
num_points = 18 # num points 
num_workers = 20 # cpu workers


file_paths = list_txt_files(folder_path) 

print(len(file_paths))

def f(x):

    return np.where((x >= 0) & (x <= 1),np.sqrt(1 - (1 - x)**2), -np.sqrt(1 - (1 - x)**2) + 2)


def generate_point_1(min_value, max_value = 288, sequence_length = 18):

    num_step = sequence_length-1

    ratio = np.linspace(0, 2, sequence_length) 

    initial_step = (max_value - min_value) / (num_step)
    steps = np.round( f(ratio) * initial_step  ).astype(int)



    adjusted_sequence = []

    current_value  = min_value

    for i in range( sequence_length ):
        current_value = current_value + steps[i]
        if current_value > max_value:
            current_value = max_value
        adjusted_sequence.append( current_value )
        
    return adjusted_sequence

def generate_point_2(x1,x2, max_value = 288, sequence_length = 18):

    list1 = np.round(np.arange(x1,x2,step_in) ).astype(int)
    
    list2 = np.round(np.linspace(x2, max_value,sequence_length-list1.shape[0]) ).astype(int)

    sequence =  np.concatenate((list1, list2)).tolist()


    return sequence



def process_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            
            try:
                coordinates = np.array([float(coord) for coord in line.strip().split(' ') if coord != ''])
                y_coordinates = coordinates[1::2]  
            
                
                scaled_and_rounded_coordinate = np.round(min(y_coordinates) / image_h * resize_h)

                data.append([scaled_and_rounded_coordinate])
                
            except:
                print(file_path)
                
    
    return data



data = []

with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:

    results = executor.map(process_file, file_paths)
    for result in results:
        data.extend(result)
data = np.array(data)

print("start calculate")

median = np.round( np.median(data[:,0], axis=0) ).astype(int)
min = np.round( np.min(data[:,0]) ).astype(int)
max = np.round( np.max(data[:,0]) ).astype(int)

q1 = np.round( np.percentile(data[:,0], 25, axis=0) ).astype(int)
q3 = np.round( np.percentile(data[:,0], 75, axis=0) ).astype(int)
mean = np.round( np.mean(data[:,0], axis=0) ).astype(int)
print(q1,q3,min,median,mean,max)



first_point_list = np.round(np.arange(min, max,step)).astype(int)
representative_lists = []

for i in range(num_classes):
    x1 = first_point_list[i]
    if i!=num_classes:
        x2 = first_point_list[i+1]
    else:
        x2 = min

    representative_lists.append([int(x) for x in np.linspace(x1, 287, num_points)] )


representative_lists = [sorted(lst) for lst in representative_lists]


print(representative_lists)

