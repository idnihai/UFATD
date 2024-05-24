import torch, os
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
from evaluation.eval_wrapper import eval_lane
import torch
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    dist_print('start testing...')
    assert cfg.backbone in ['18','34','34fca','50','101','152','mobilenet_v2','vit_b_16','squeezenet1_0','50next','101next','50wide','101wide']
    cls_num_per_lane = cfg.cls_num_per_lane


    net = parsingNet(pretrained = False, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cfg.cls_num_per_lane, cfg.num_lanes, cfg.num_classes),
                     num_classes=cfg.num_classes).cuda() 

    state_dict = torch.load(cfg.test_model, map_location = 'cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict = False)

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [args.local_rank])

    if not os.path.exists(cfg.test_work_dir):
        os.mkdir(cfg.test_work_dir)

    cfg.test_label = True
    eval_lane(net, cfg)