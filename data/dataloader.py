import torch, os
import random

import torchvision.transforms as transforms
import data.mytransforms as mytransforms
from data.constant import tusimple_row_anchor, culane_row_anchor, SRail_row_anchor, RailDB_row_anchor, dlrail_row_anchor
from data.dataset import LaneClsDataset, LaneTestDataset

def cache_train_val(path,):
    cache_dir = '.cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    with open(path, 'r') as file:
        lines = file.readlines()

    random.shuffle(lines)
    
    total_lines = len(lines)
    train_split = int(total_lines * 0.85)
   
    train_data = lines[:train_split]
    val_data = lines[train_split:]

    val_data = [line.split()[0]+'\n' for line in val_data]

    with open('.cache/train_gt.txt', 'w') as file:
        file.writelines(train_data)

    with open('.cache/val.txt', 'w') as file:
        file.writelines(val_data)

def get_train_loader(batch_size, data_root, griding_num, dataset, distributed, num_lanes):
    target_transform = transforms.Compose([
        mytransforms.FreeScaleMask((288, 800)),
        mytransforms.MaskToTensor(),
    ])
    segment_transform = transforms.Compose([
        mytransforms.FreeScaleMask((36, 100)),
        mytransforms.MaskToTensor(),
    ])
    img_transform = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    simu_transform = mytransforms.Compose2([
        mytransforms.RandomRotate(6),
        mytransforms.RandomUDoffsetLABEL(100),
        mytransforms.RandomLROffsetLABEL(200)
    ])

    # get anchor
    if dataset == 'CULane':
        row_anchor = culane_row_anchor
    elif dataset == 'S-Rail':
        row_anchor = SRail_row_anchor
    elif dataset == 'RailDB':
        row_anchor = RailDB_row_anchor
    elif dataset == 'dlrail':
        row_anchor = dlrail_row_anchor
    elif dataset == 'Tusimple':
        row_anchor = tusimple_row_anchor

    if dataset == 'CULane'  :
        train_dataset = LaneClsDataset(data_root,
                                           os.path.join(data_root, 'list/train_gt.txt'),
                                           img_transform=img_transform, target_transform=target_transform,
                                           simu_transform = simu_transform,
                                           segment_transform=segment_transform, 
                                           row_anchor = row_anchor,
                                           griding_num=griding_num,  num_lanes = num_lanes)
        
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            
        else:
            train_sampler = torch.utils.data.RandomSampler(train_dataset)
            
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler = train_sampler, num_workers=8, pin_memory=True)
        

        return train_loader 


    elif dataset == 'S-Rail' or dataset == 'RailDB' or dataset == 'dlrail':

        cache_train_val(os.path.join(data_root, 'list/train_gt.txt'))
        
        train_dataset = LaneClsDataset(data_root,
                                           '.cache/train_gt.txt',
                                           img_transform=img_transform, target_transform=target_transform,
                                           simu_transform = simu_transform,
                                           segment_transform=segment_transform, 
                                           row_anchor = row_anchor,
                                           griding_num=griding_num, num_lanes = num_lanes)


    

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler = train_sampler, num_workers=8, pin_memory=True)
    

    return train_loader

def get_test_loader(cfg):
    batch_size =cfg.batch_size
    data_root = cfg.data_root
    dataset =cfg.dataset 
    distributed = cfg.distributed
    img_transforms = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    if cfg.test_label:
        print("Start tesing")
        if dataset == 'CULane' or  dataset == 'S-Rail' or dataset == 'RailDB' or dataset == 'dlrail' :
            test_dataset = LaneTestDataset(data_root,os.path.join(data_root, 'list/test.txt'),img_transform = img_transforms)
            
        elif dataset == 'Tusimple':
            test_dataset = LaneTestDataset(data_root,os.path.join(data_root, 'test.txt'), img_transform = img_transforms)
            

        if distributed:
            sampler = SeqDistributedSampler(test_dataset, shuffle = False)
        else:
            sampler = torch.utils.data.SequentialSampler(test_dataset)
        loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler = sampler, num_workers=4)
        return loader
    else:
        print("Start val")
        if dataset == 'CULane' :  
            test_dataset = LaneTestDataset(data_root,os.path.join(data_root, 'list/val.txt'),img_transform = img_transforms)

        elif dataset == 'S-Rail' or dataset == 'RailDB' or dataset == 'dlrail' :
            test_dataset = LaneTestDataset(data_root,'.cache/val.txt',img_transform = img_transforms)

        if distributed:
            sampler = SeqDistributedSampler(test_dataset, shuffle = False)
        else:
            sampler = torch.utils.data.SequentialSampler(test_dataset)
        loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler = sampler, num_workers=4)
        return loader



class SeqDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    '''
    Change the behavior of DistributedSampler to sequential distributed sampling.
    The sequential sampling helps the stability of multi-thread testing, which needs multi-thread file io.
    Without sequentially sampling, the file io on thread may interfere other threads.
    '''
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas, rank, shuffle)
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))


        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size


        num_per_rank = int(self.total_size // self.num_replicas)

        # sequential sampling
        indices = indices[num_per_rank * self.rank : num_per_rank * (self.rank + 1)]

        assert len(indices) == self.num_samples

        return iter(indices)