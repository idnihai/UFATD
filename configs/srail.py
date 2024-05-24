
# DATA
dataset='S-Rail' 
data_root = '../SynData/S-Rail/'

# TRAIN
epoch = 100
batch_size = 32
optimizer = 'Adam'  #['SGD','Adam'] 


lr_backbone = 4e-4
lr_cls =  1e-3 
classification = 5e-5


weight_decay = 1e-4
momentum = 0.9
scheduler = 'cos' #['multi', 'cos']
steps = [25,38] #
gamma  = 0.5 
warmup = 'linear'
warmup_iters = 695 

unsqueeze_epoch = 15

# NETWORK
griding_num = 200
backbone = '18'

# LOSS
sim_loss_w = 0.0
shp_loss_w = 0.0
anchor_loss_w = 0.05


# EXP
note = ''
log_path = "./tmp"

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

distributed =False

# TEST
test_model = None
test_work_dir = "./results/"

num_lanes = 2
num_classes = 3
cls_num_per_lane = 18


test_label = False
