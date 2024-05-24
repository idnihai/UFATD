import torch
import time
import numpy as np
from model.model import parsingNet
from thop import profile
from thop import clever_format


torch.backends.cudnn.benchmark = True

net = parsingNet(pretrained = False, backbone='50',cls_dim = (200+1,18,2,3)).cuda()

net.eval()

x = torch.zeros((1,3,288,800)).cuda() + 1
for i in range(100):
    y = net(x)

t_all = []
for i in range(500):
    t1 = time.time()
    y = net(x)
    t2 = time.time()
    t_all.append(t2 - t1)

print('average time:', np.mean(t_all) / 1)
print('average fps:',1 / np.mean(t_all))

print('fastest time:', min(t_all) / 1)
print('fastest fps:',1 / min(t_all))

print('slowest time:', max(t_all) / 1)
print('slowest fps:',1 / max(t_all))

flops, params = profile(net, inputs=(x, ))
flops, params = clever_format([flops, params], "%.3f")
print(f"FLOPs: {flops}, Params: {params}")


