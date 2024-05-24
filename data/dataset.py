import torch
from PIL import Image
import os
import pdb
import numpy as np

from data.mytransforms import find_start_pos


def loader_func(path):
    return Image.open(path)


class LaneTestDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform=None):
        super(LaneTestDataset, self).__init__()
        self.path = path
        self.img_transform = img_transform
        with open(list_path, 'r') as f:
            self.list = f.readlines()
        self.list = [l[1:] if l[0] == '/' else l for l in self.list]  


    def __getitem__(self, index):
        name = self.list[index].split()[0]
        img_path = os.path.join(self.path, name)

        
        img = loader_func(img_path)       
        img = img.convert('RGB')

        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, name

    def __len__(self):
        return len(self.list)


class LaneClsDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform = None,target_transform = None,simu_transform = None, griding_num=50, load_name = False,
                row_anchor = None, segment_transform=None, num_lanes = 4):
        super(LaneClsDataset, self).__init__()
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.segment_transform = segment_transform
        self.simu_transform = simu_transform  
        self.path = path
        self.griding_num = griding_num
        self.load_name = load_name

        self.num_lanes = num_lanes

        with open(list_path, 'r') as f:
            self.list = f.readlines()
        
        self.row_anchor = row_anchor
        

    def __getitem__(self, index):
        l = self.list[index]
        l_info = l.split()
        img_name, label_name = l_info[0], l_info[1]
        if img_name[0] == '/':
            img_name = img_name[1:]
            label_name = label_name[1:]

        label_path = os.path.join(self.path, label_name)
        label = loader_func(label_path)

        
        img_path = os.path.join(self.path, img_name)
        img = loader_func(img_path)
        if img.mode == "RGBA": img = img.convert('RGB')

        row_anchor, index_of_row_anchor = self.find_closest_row_anchor(label) 
        
        if self.simu_transform is not None:
            img, label = self.simu_transform(img, label)
        lane_pts = self._get_index(label, row_anchor) 


        w, h = img.size
        cls_label = self._grid_pts(lane_pts, self.griding_num, w)
        # make the coordinates to classification label


        if self.img_transform is not None:
            img = self.img_transform(img)


        if self.load_name:
            return img, cls_label, img_name, index_of_row_anchor
        
        return img, cls_label, index_of_row_anchor

    def __len__(self):
        
        return len(self.list)

    def _grid_pts(self, pts, num_cols, w):
        # pts : numlane,n,2
        num_lane, n, n2, num_class = pts.shape
        col_sample = np.linspace(0, w - 1, num_cols) 

        assert n2 == 2
        to_pts = np.zeros(( n, num_lane, num_class))
        for ind in range(num_class):
            for i in range(num_lane):
                pti = pts[ i, :, 1, ind]

                to_pts[ :, i, ind] = np.asarray([int(pt // (col_sample[1] - col_sample[0])) 
                                                 if pt != -1 else num_cols  for pt in pti]) 

        return to_pts.astype(int)


    def _get_index(self, label, row_anchor):
        w, h = label.size

        sample_tmp_list = []
        if h != 288:
            scale_f = lambda x : int((x * 1.0/288) * h)

            for row_anchor in self.row_anchor:

                sample_tmp = list(map(scale_f,row_anchor)) 
                sample_tmp_list.append(sample_tmp)


        all_idx = np.zeros(( self.num_lanes,len(sample_tmp),2,len(sample_tmp_list))) 
        for ind, sample_tmp in enumerate(sample_tmp_list):
            for i,r in enumerate(sample_tmp):
                label_r = np.asarray(label)[int(round(r))]
                
                for lane_idx in range(1, self.num_lanes + 1):
                    pos = np.where(label_r == lane_idx)[0]
                    if len(pos) == 0:
                        all_idx[ lane_idx - 1, i, 0, ind] = r
                        all_idx[lane_idx - 1, i, 1, ind] = -1
                        continue
                    pos = np.mean(pos)
                    
                    all_idx[ lane_idx - 1, i, 0, ind] = r # y
                    all_idx[ lane_idx - 1, i, 1, ind] = pos # x

        # data augmentation: extend the lane to the boundary of image
        all_idx_cp = all_idx.copy()
        
        for ind in range(len(self.row_anchor)):
            for i in range(self.num_lanes):
                if np.all(all_idx_cp[i,:,1, ind] == -1):
                    continue
                # if there is no lane

                valid = all_idx_cp[i,:,1, ind] != -1
                # get all valid lane points' index
                valid_idx = all_idx_cp[i,valid,:, ind]
                # get all valid lane points
                if valid_idx[-1,0] == all_idx_cp[0,-1,0, ind]:
                    # if the last valid lane point's y-coordinate is already the last y-coordinate of all rows
                    # this means this lane has reached the bottom boundary of the image
                    # so we skip

                    continue
                if len(valid_idx) < 6:
                    continue
                # if the lane is too short to extend

                valid_idx_half = valid_idx[len(valid_idx) // 2:,:]  # 选择有效车道线点的后半部分
                p = np.polyfit(valid_idx_half[:,0], valid_idx_half[:,1],deg = 1)
                start_line = valid_idx_half[-1,0]
                pos = find_start_pos(all_idx_cp[i,:,0, ind],start_line) + 1
                
                fitted = np.polyval(p,all_idx_cp[i,pos:,0, ind])
                fitted = np.array([-1  if y < 0 or y > w-1 else y for y in fitted])

                assert np.all(all_idx_cp[i,pos:,1, ind] == -1)
                all_idx_cp[i,pos:,1, ind] = fitted
            if -1 in all_idx[ :, :, 0, ind]:
                pdb.set_trace()
            
        return all_idx_cp



    def find_closest_row_anchor(self, img):

        img_array = np.array(img)
        min_y = np.argmax(img_array.sum(axis=1) > 0) / img.size[1] * 288
        closest_row_anchors = [row for row in self.row_anchor if row[0] <= min_y]
        
        if not closest_row_anchors:  return self.row_anchor[0], 0

        closest_row_anchor = min(closest_row_anchors, key=lambda x: abs(x[0] - min_y))
        
        index_of_closest_row_anchor = self.row_anchor.index(closest_row_anchor)

        return closest_row_anchor, index_of_closest_row_anchor

