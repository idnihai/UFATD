from data.dataloader import get_test_loader
from utils.dist_utils import is_main_process, dist_print, dist_tqdm, synchronize
import os, json, torch, scipy
import numpy as np
import platform
import cv2
from data.constant import culane_row_anchor, SRail_row_anchor, RailDB_row_anchor, dlrail_row_anchor
from PIL import Image

def generate_lines(dataset, data_root, out, anchor_cls, shape, names, output_path, griding_num, localization_type='abs', flip_updown=False ):

    col_sample = np.linspace(0, shape[1] - 1, griding_num)
    col_sample_w = col_sample[1] - col_sample[0]
    for j in range(out.shape[0]): # j: index of image in batch
        out_j = out[j].data.cpu().numpy()
        
        if flip_updown:
            out_j = out_j[:, ::-1, :]
        if localization_type == 'abs':
            out_j = np.argmax(out_j, axis=0)
            out_j[out_j == griding_num] = -1
            out_j = out_j + 1
        elif localization_type == 'rel':
            prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
            idx = np.arange(griding_num) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            out_j = np.argmax(out_j, axis=0)
            loc[out_j == griding_num] = 0
            out_j = loc
        else:
            raise NotImplementedError
        name = names[j]

        line_save_path = os.path.join(output_path, name[:name.rfind('.')] + '.lines.txt')
        
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if dataset == 'CULane':
            row_anchor = culane_row_anchor
        elif dataset == 'S-Rail':
            row_anchor = SRail_row_anchor
        elif dataset == 'RailDB':
            row_anchor = RailDB_row_anchor
        elif dataset == 'dlrail':
            row_anchor = dlrail_row_anchor
        

        
        cls_num_per_lane = len(row_anchor[0])
        if dataset == 'S-Rail' or dataset == 'dlrail' or dataset == 'RailDB':
            with open(line_save_path, 'w') as fp:
                for i in range(out_j.shape[1]): 
                    if np.sum(out_j[:, i] != 0) > 2: 
                        for k in range(out_j.shape[0]):

                            if out_j[k, i] > 0:

                                    fp.write(
                                    '%d %d ' % (int(out_j[k, i] * col_sample_w * 1920 / 800) - 1, int(1080 * (row_anchor[anchor_cls[j]][cls_num_per_lane - 1 - k] / 288)) - 1))  #

                                    
                        fp.write('\n')

        elif dataset == 'CULane':
            with open(line_save_path, 'w') as fp:
                for i in range(out_j.shape[1]):
                    if np.sum(out_j[:, i] != 0) > 2:
                        for k in range(out_j.shape[0]):
                            if out_j[k, i] > 0:
                                fp.write(
                                    '%d %d ' % (int(out_j[k, i] * col_sample_w * 1640 / 800) - 1, int(590*  (row_anchor[anchor_cls[j]][cls_num_per_lane - 1 - k] / 288)) - 1))
                        fp.write('\n')
        
def run_test(cfg,dataset,net, data_root, exp_name, work_dir, griding_num):
    
    output_path = os.path.join(work_dir, exp_name)
    if not os.path.exists(output_path) and is_main_process():
        os.mkdir(output_path)
    synchronize()
    loader = get_test_loader(cfg)
    
    for i, data in enumerate(dist_tqdm(loader)):
        imgs, names = data
        imgs = imgs.cuda()
        with torch.no_grad():
            out = net(imgs)

        if len(out) == 2:
            out, anchor_cls = out
            
            _, anchor_cls = torch.max(anchor_cls, 1)  

            out = torch.stack([group_[..., ind] for group_, ind in zip(out, anchor_cls)], dim=0)
            

        generate_lines(dataset, data_root, out, anchor_cls, imgs[0,0].shape,names,output_path,griding_num,localization_type = 'rel',flip_updown = True)

def eval_lane(net, cfg,  ep = None, logger = None):
    net.eval()
    if cfg.dataset == 'CULane' or cfg.dataset == 'S-Rail' or cfg.dataset == 'dlrail' or cfg.dataset == 'RailDB':
        run_test(cfg, cfg.dataset,net,cfg.data_root, 'culane_eval_tmp', cfg.test_work_dir, cfg.griding_num)
        synchronize()   # wait for all results
        if is_main_process():
            res = call_culane_eval(cfg, cfg.dataset,cfg.data_root, 'culane_eval_tmp', cfg.test_work_dir)
            TP,FP,FN = 0,0,0
            for k, v in res.items():
                val = float(v['Fmeasure']) if 'nan' not in v['Fmeasure'] else 0
                val_tp,val_fp,val_fn = int(v['tp']),int(v['fp']),int(v['fn'])
                TP += val_tp
                FP += val_fp
                FN += val_fn
                dist_print(k,val)
                if logger is not None:
                    if k == 'res_cross':
                        logger.add_scalar('CuEval_cls/'+k,val_fp,global_step = ep)
                        continue
                    logger.add_scalar('CuEval_cls/'+k,val,global_step = ep)
            if TP + FP == 0:
                P = 0
                print("nearly no results!")
            else:
                P = TP * 1.0/(TP + FP)
            if TP + FN == 0:
                R = 0
                print("nearly no results!")
            else:
                R = TP * 1.0/(TP + FN)
            if (P+R) == 0:
                F = 0
            else:
                F = 2*P*R/(P + R)
            dist_print(F)
            if logger is not None:
                logger.add_scalar('CuEval/total',F,global_step = ep)
                logger.add_scalar('CuEval/P',P,global_step = ep)
                logger.add_scalar('CuEval/R',R,global_step = ep)
        synchronize()
        if is_main_process():
            return F
        else:
            return None
            


def read_helper(path):
    lines = open(path, 'r').readlines()[1:]
    lines = ' '.join(lines)
    values = lines.split(' ')[1::2]
    keys = lines.split(' ')[0::2]
    keys = [key[:-1] for key in keys]
    res = {k : v for k,v in zip(keys,values)}
    return res
def evaluate(args):
    eval_cmd, data_dir, detect_dir, listX, w_lane, iou, im_w, im_h, frame, outX = args
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (eval_cmd, data_dir, detect_dir, data_dir, listX, w_lane, iou, im_w, im_h, frame, outX))

def call_culane_eval(cfg, dataset, data_dir, exp_name,output_path):
    if data_dir[-1] != '/':
        data_dir = data_dir + '/'
    detect_dir=os.path.join(output_path,exp_name)+'/'
    eval_cmd = './evaluation/culane/evaluate'
    if platform.system() == 'Windows':
        eval_cmd = eval_cmd.replace('/', os.sep)

    w_lane=30
    iou=0.5;  # Set iou to 0.3 or 0.5
    
    if dataset == 'S-Rail' or dataset == 'dlrail' or dataset == 'RailDB':
        im_w=1920
        im_h=1080
    
    elif dataset == 'CULane':
        im_w=1640
        im_h=590
    frame=1
    if cfg.test_label:
            
        if dataset == 'S-Rail':
            list0 = os.path.join(data_dir,'list/test_split/0.txt')
            list1 = os.path.join(data_dir,'list/test_split/1.txt')
            list2 = os.path.join(data_dir,'list/test_split/2.txt')
            list3 = os.path.join(data_dir,'list/test_split/3.txt')
            
            if not os.path.exists(os.path.join(output_path,'txt')):
                os.mkdir(os.path.join(output_path,'txt'))
            out0 = os.path.join(output_path,'txt','0.txt')
            out1=os.path.join(output_path,'txt','1.txt')
            out2=os.path.join(output_path,'txt','2.txt')
            out3=os.path.join(output_path,'txt','3.txt')


            # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list0,w_lane,iou,im_w,im_h,frame,out0))
            os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list0,w_lane,iou,im_w,im_h,frame,out0))
            # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list1,w_lane,iou,im_w,im_h,frame,out1))
            os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list1,w_lane,iou,im_w,im_h,frame,out1))
            # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list2,w_lane,iou,im_w,im_h,frame,out2))
            os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list2,w_lane,iou,im_w,im_h,frame,out2))
            # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list3,w_lane,iou,im_w,im_h,frame,out3))
            os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list3,w_lane,iou,im_w,im_h,frame,out3))
        
            res_all = {}
            res_all['res_0'] = read_helper(out0)
            res_all['res_1']= read_helper(out1)
            res_all['res_2']= read_helper(out2)
            res_all['res_3'] = read_helper(out3)
            
            return res_all
        elif dataset == 'dlrail':
            list0 = os.path.join(data_dir,'list/test_split/test0_normal.txt')
            
            
            if not os.path.exists(os.path.join(output_path,'txt')):
                os.mkdir(os.path.join(output_path,'txt'))
            out0 = os.path.join(output_path,'txt','out0_normal.txt')


            # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list0,w_lane,iou,im_w,im_h,frame,out0))
            os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list0,w_lane,iou,im_w,im_h,frame,out0))
            
            res_all = {}
            res_all['res_normal'] = read_helper(out0)

            
            return res_all
        elif dataset == 'RailDB':

            list0 = os.path.join(data_dir,'list/test_split/cross.txt')
            list1 = os.path.join(data_dir,'list/test_split/curve.txt')
            list2 = os.path.join(data_dir,'list/test_split/far.txt')
            list3 = os.path.join(data_dir,'list/test_split/line.txt')
            list4 = os.path.join(data_dir,'list/test_split/near.txt')
            list5 = os.path.join(data_dir,'list/test_split/night.txt')
            list6 = os.path.join(data_dir,'list/test_split/rain.txt')
            list7 = os.path.join(data_dir,'list/test_split/slope.txt')
            list8 = os.path.join(data_dir,'list/test_split/sun.txt')


            
            
            if not os.path.exists(os.path.join(output_path,'txt')):
                os.mkdir(os.path.join(output_path,'txt'))

            out0 = os.path.join(output_path,'txt','cross.txt')
            out1=os.path.join(output_path,'txt','curve.txt')
            out2=os.path.join(output_path,'txt','far.txt')
            out3=os.path.join(output_path,'txt','line.txt')
            out4=os.path.join(output_path,'txt','near.txt')
            out5=os.path.join(output_path,'txt','night.txt')
            out6=os.path.join(output_path,'txt','rain.txt')
            out7=os.path.join(output_path,'txt','slope.txt')
            out8=os.path.join(output_path,'txt','sun.txt')


            params = [
                (eval_cmd, data_dir, detect_dir, listX, w_lane, iou, im_w, im_h, frame, outX) 
                for listX, outX in zip([list0, list1, list2, list3, list4, list5, list6, list7, list8], [out0, out1, out2, out3, out4, out5, out6, out7, out8])
            ]


            for param in params:
                evaluate(param)

            res_all = {}
            res_all['cross'] = read_helper(out0)
            res_all['curve']= read_helper(out1)
            res_all['far']= read_helper(out8)
            res_all['line'] = read_helper(out4)
            res_all['near'] = read_helper(out3)
            res_all['night']= read_helper(out5)
            res_all['rain'] = read_helper(out2)
            res_all['slope']= read_helper(out6)
            res_all['sun']= read_helper(out7)


            
            return res_all
        elif dataset == 'CULane':
            list0 = os.path.join(data_dir,'list/test_split/test0_normal.txt')
            list1 = os.path.join(data_dir,'list/test_split/test1_crowd.txt')
            list2 = os.path.join(data_dir,'list/test_split/test2_hlight.txt')
            list3 = os.path.join(data_dir,'list/test_split/test3_shadow.txt')
            list4 = os.path.join(data_dir,'list/test_split/test4_noline.txt')
            list5 = os.path.join(data_dir,'list/test_split/test5_arrow.txt')
            list6 = os.path.join(data_dir,'list/test_split/test6_curve.txt')
            list7 = os.path.join(data_dir,'list/test_split/test7_cross.txt')
            list8 = os.path.join(data_dir,'list/test_split/test8_night.txt')
            if not os.path.exists(os.path.join(output_path,'txt')):
                os.mkdir(os.path.join(output_path,'txt'))
            out0 = os.path.join(output_path,'txt','out0_normal.txt')
            out1=os.path.join(output_path,'txt','out1_crowd.txt')
            out2=os.path.join(output_path,'txt','out2_hlight.txt')
            out3=os.path.join(output_path,'txt','out3_shadow.txt')
            out4=os.path.join(output_path,'txt','out4_noline.txt')
            out5=os.path.join(output_path,'txt','out5_arrow.txt')
            out6=os.path.join(output_path,'txt','out6_curve.txt')
            out7=os.path.join(output_path,'txt','out7_cross.txt')
            out8=os.path.join(output_path,'txt','out8_night.txt')

            eval_cmd = './evaluation/culane/evaluate'
            if platform.system() == 'Windows':
                eval_cmd = eval_cmd.replace('/', os.sep)

            # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list0,w_lane,iou,im_w,im_h,frame,out0))
            os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list0,w_lane,iou,im_w,im_h,frame,out0))
            # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list1,w_lane,iou,im_w,im_h,frame,out1))
            os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list1,w_lane,iou,im_w,im_h,frame,out1))
            # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list2,w_lane,iou,im_w,im_h,frame,out2))
            os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list2,w_lane,iou,im_w,im_h,frame,out2))
            # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list3,w_lane,iou,im_w,im_h,frame,out3))
            os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list3,w_lane,iou,im_w,im_h,frame,out3))
            # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list4,w_lane,iou,im_w,im_h,frame,out4))
            os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list4,w_lane,iou,im_w,im_h,frame,out4))
            # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list5,w_lane,iou,im_w,im_h,frame,out5))
            os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list5,w_lane,iou,im_w,im_h,frame,out5))
            # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list6,w_lane,iou,im_w,im_h,frame,out6))
            os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list6,w_lane,iou,im_w,im_h,frame,out6))
            # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list7,w_lane,iou,im_w,im_h,frame,out7))
            os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list7,w_lane,iou,im_w,im_h,frame,out7))
            # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list8,w_lane,iou,im_w,im_h,frame,out8))
            os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list8,w_lane,iou,im_w,im_h,frame,out8))
            res_all = {}
            res_all['res_normal'] = read_helper(out0)
            res_all['res_crowd']= read_helper(out1)
            res_all['res_night']= read_helper(out8)
            res_all['res_noline'] = read_helper(out4)
            res_all['res_shadow'] = read_helper(out3)
            res_all['res_arrow']= read_helper(out5)
            res_all['res_hlight'] = read_helper(out2)
            res_all['res_curve']= read_helper(out6)
            res_all['res_cross']= read_helper(out7)
            return res_all
    
    else:
        if dataset == 'S-Rail' or dataset == 'dlrail' or dataset == 'RailDB':
            list0 = '.cache/val.txt'
            
            
            if not os.path.exists(os.path.join(output_path,'txt')):
                os.mkdir(os.path.join(output_path,'txt'))
            out0 = os.path.join(output_path,'txt','val.txt')
            
            

            eval_cmd = './evaluation/culane/evaluate'
            if platform.system() == 'Windows':
                eval_cmd = eval_cmd.replace('/', os.sep)

            # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list0,w_lane,iou,im_w,im_h,frame,out0))
            os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list0,w_lane,iou,im_w,im_h,frame,out0))
            
            res_all = {}
            res_all['res_0'] = read_helper(out0)
            
            
            return res_all
        elif dataset == 'CULane':
            list0 = os.path.join(data_dir,'list/val.txt')

            if not os.path.exists(os.path.join(output_path,'txt')):
                os.mkdir(os.path.join(output_path,'txt'))
            out0 = os.path.join(output_path,'txt','val.txt')


            
            

            eval_cmd = './evaluation/culane/evaluate'
            if platform.system() == 'Windows':
                eval_cmd = eval_cmd.replace('/', os.sep)

            # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list0,w_lane,iou,im_w,im_h,frame,out0))
            os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list0,w_lane,iou,im_w,im_h,frame,out0))
            
            res_all = {}
            res_all['res_0'] = read_helper(out0)

            return res_all
