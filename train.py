import torch, datetime
import numpy as np

from model.model import parsingNet
from data.dataloader import get_train_loader

from utils.dist_utils import dist_print, dist_tqdm
from utils.factory import get_metric_dict, get_loss_dict, get_scheduler
from utils.metrics import update_metrics, reset_metrics
from utils.common import merge_config, save_model, cp_projects, save_model_by_epoch
from utils.common import get_work_dir, get_logger

from evaluation.eval_wrapper import eval_lane

import time

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



def inference(net, data_label):


    img, cls_label, anchor_label = data_label
    img, cls_label, anchor_label = img.cuda(), cls_label.long().cuda(), anchor_label.long().cuda()
    cls_out, anchor_cls = net(img)

    
    return {'cls_out': cls_out, 'cls_label': cls_label, 
            'anchor_cls': anchor_cls, 'anchor_label': anchor_label}


def resolve_val_data(results):
    
    results['cls_out'] = torch.argmax(results['cls_out'], dim=1)
    results['anchor_cls'] = torch.argmax(results['anchor_cls'], dim=1)
    

    return results


def calc_loss(loss_dict, results, logger, global_step, for_train=True):
    loss = 0

    for i in range(len(loss_dict['name'])):

        data_src = loss_dict['data_src'][i]

        datas = [results[src] for src in data_src]

        loss_cur = loss_dict['op'][i](*datas)

        if global_step % 20 == 0 and for_train:
            logger.add_scalar('loss/'+loss_dict['name'][i], loss_cur, global_step)
        
        loss += loss_cur * loss_dict['weight'][i]
    return loss


def train(net, train_loader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict):
    net.train()
    progress_bar = dist_tqdm(train_loader)
    t_data_0 = time.time()
    for b_idx, data_label in enumerate(progress_bar):
        t_data_1 = time.time()
        reset_metrics(metric_dict)
        global_step = epoch * len(train_loader) + b_idx

        t_net_0 = time.time()
        
        results = inference(net, data_label)
        loss = calc_loss(loss_dict, results, logger, global_step)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(global_step)
        t_net_1 = time.time()
        
        results = resolve_val_data(results)
        
        update_metrics(metric_dict, results)

        if global_step % 20 == 0:
            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                logger.add_scalar('metric/' + me_name, me_op.get(), global_step=global_step)
        logger.add_scalar('meta/lr_model', optimizer.param_groups[0]['lr'], global_step=global_step)
        logger.add_scalar('meta/lr_cls', optimizer.param_groups[1]['lr'], global_step=global_step)
        logger.add_scalar('meta/lr_classification', optimizer.param_groups[2]['lr'], global_step=global_step)
        

        if hasattr(progress_bar,'set_postfix'):
            kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
            progress_bar.set_postfix(loss = '%.3f' % float(loss), 
                                    data_time = '%.3f' % float(t_data_1 - t_data_0), 
                                    net_time = '%.3f' % float(t_net_1 - t_net_0), 
                                    **kwargs)
        t_data_0 = time.time()







if __name__ == "__main__":

    seed = 3407
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed )


    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    work_dir = get_work_dir(cfg)

    distributed = cfg.distributed
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    dist_print(datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]') + ' start training...')
    dist_print(cfg)
    assert cfg.backbone in ['18','34','34fca', '50','101','152','50next','101next','50wide','101wide','mobilenet_v2', 'squeezenet1_0', 'vit_b_16',]


    train_loader  = get_train_loader(cfg.batch_size, cfg.data_root, cfg.griding_num, cfg.dataset, distributed, cfg.num_lanes)
    
    net = parsingNet(pretrained = True, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cfg.cls_num_per_lane, cfg.num_lanes, cfg.num_classes),num_classes=cfg.num_classes).cuda()

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [args.local_rank])
   
        
    if cfg.optimizer == 'Adam':
        optimizer = torch.optim.Adam([
            {'params': net.model.parameters(), 'lr':cfg.lr_backbone },
            {'params': net.cls.parameters(), 'lr': cfg.lr_cls },
            {'params': net.classification_network.parameters(), 'lr': cfg.classification},  
            {'params': net.pool.parameters(), 'lr': cfg.lr_backbone},
    ], weight_decay=cfg.weight_decay) 
    
    elif cfg.optimizer == 'SGD':
        optimizer = torch.optim.SGD([
            {'params': net.model.parameters(), },
            {'params': net.cls.parameters(), },
            {'params': net.classification_network.parameters(), 'lr':cfg.classification },
            {'params': net.pool.parameters(), }, 
        ], cfg.lr_global, momentum=cfg.momentum,weight_decay=cfg.weight_decay) 

    if cfg.finetune is not None:
        dist_print('finetune from ', cfg.finetune)
        state_all = torch.load(cfg.finetune)['model']
        state_clip = {}  # only use backbone parameters
        for k,v in state_all.items():
            if 'model' in k:
                state_clip[k] = v
        net.load_state_dict(state_clip, strict=False)
    if cfg.resume is not None:
        dist_print('==> Resume model from ' + cfg.resume)
        resume_dict = torch.load(cfg.resume, map_location='cpu')
        net.load_state_dict(resume_dict['model'])
        if 'optimizer' in resume_dict.keys():
            optimizer.load_state_dict(resume_dict['optimizer'])
        resume_epoch = int(os.path.split(cfg.resume)[1][2:5]) + 1
    else:
        resume_epoch = 0



    scheduler = get_scheduler(optimizer, cfg, len(train_loader))
    dist_print(len(train_loader))
    metric_dict = get_metric_dict(cfg)
    
    logger = get_logger(work_dir, cfg)
    cp_projects(args.auto_backup, work_dir)



    max_res = 0
    res = None

    for epoch in range(resume_epoch, cfg.epoch+1):

        
        if epoch < cfg.unsqueeze_epoch:
            if epoch < 5:
                for param in net.model.parameters():
                    param.requires_grad = False  
            else:
                for param in net.model.parameters():
                    param.requires_grad = True  
                    
            for param in net.classification_network.parameters():
                param.requires_grad = False  
             
            anchor_loss_w = 0.0


        else:
            for param in net.parameters():
                param.requires_grad = True  # 解冻所有层
            anchor_loss_w = cfg.anchor_loss_w
        loss_dict = get_loss_dict(cfg, anchor_loss_w)
        




        train(net, train_loader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict)

        res = eval_lane(net, cfg, ep = epoch, logger = logger)

        if res is not None and res >= max_res:
            max_res = res
            save_model(net, optimizer, epoch, work_dir, distributed)
        logger.add_scalar('CuEval/X',max_res,global_step = epoch)

        if  epoch == cfg.epoch:
            
            save_model_by_epoch(net, optimizer, epoch,work_dir, distributed)
    logger.close()
