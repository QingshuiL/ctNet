

import torch
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from src.lib.models.networks.efficientnet_b4_ import get_efficientnet
from src.lib.models.networks.msra_resnet import get_pose_net as get_pose_net
from src.lib.models.networks.dlav0 import get_pose_net as get_dlav0
from src.lib.models.networks.pose_dla_dcn import get_pose_net as get_dla_dcn
from src.lib.models.networks.resnet_dcn import get_pose_net as get_pose_net_dcn
from src.lib.models.networks.large_hourglass import get_large_hourglass_net
from src.lib.models.networks.res_fpn import  get_pose_res_fpn
from src.lib.models.networks.resnext import get_resxt



def arch_(model):
    if model == 'efficientnet-b0':
        print('arch is b0')
        Net = EfficientNet.from_pretrained('efficientnet-b0')
        dummy_imput = torch.rand(1, 3, 224, 224)
        tenboard_dir = 'C:/Users/10295/CV/CenterNet-master/runs/b0'
        with SummaryWriter(tenboard_dir,comment='Net') as w:
            w.add_graph(Net, (dummy_imput,),verbose=True)
            
    elif model == 'efficientnet-b4':
        print('arch is b4')
        Net = EfficientNet.from_pretrained('efficientnet-b4')
        dummy_imput = torch.rand(1, 3, 380, 380)
        tenboard_dir = 'C:/Users/10295/CV/CenterNet-master/runs/b4'
        with SummaryWriter(tenboard_dir,comment='Net') as w:
            w.add_graph(Net, (dummy_imput,),verbose=True)
    else: 
        print('arch error!')
    
def get_effi(model_path):
    heads = {'hm': 10, 'wh': 2, 'reg': 2}
    head_conv = 256
    model = get_efficientnet(num_layers=4,heads=heads, head_conv=head_conv)
    dummy_imput = torch.rand(1, 3, 512, 512)
    tenboard_dir = 'C:/Users/10295/CV/CenterNet-master/runs/b4-1'
    with SummaryWriter(tenboard_dir,comment='model') as w:
        w.add_graph(model, (dummy_imput,),verbose=True)

def efficientnet():
    heads = {'hm': 10, 'wh': 2, 'reg': 2}
    head_conv = 256
    model = get_efficientnet(num_layers=4,heads=heads, head_conv=head_conv) 
    return model

def resdcn18():
    heads = {'hm': 10, 'wh': 2, 'reg': 2}
    head_conv = 256
    model = get_pose_net_dcn(num_layers=18,heads=heads, head_conv=head_conv) 
    return model

def res_fpn():
    heads = {'hm': 10, 'wh': 2, 'reg': 2}
    head_conv = 256
    model = get_pose_res_fpn(num_layers=101,heads=heads, head_conv=head_conv) 
    return model

def hourglass():    # 实际只需 heads参数
    heads = {'hm': 10, 'wh': 2, 'reg': 2}
    head_conv = 256
    model = get_large_hourglass_net(num_layers=101,heads=heads, head_conv=head_conv)
    return model

def resnext():
    heads = {'hm': 10, 'wh': 2, 'reg': 2}
    head_conv = 256
    model = get_resxt(num_layers=101,heads=heads, head_conv=head_conv)
    return model    

if __name__ == "__main__":
    # model_arch = 'efficientnet-b4'
    # arch_(model_arch)
    # model_path = 'C:/Users/10295/CV/CenterNet-master/exp/ctdet/visdrone2019_effi/model_last.pth'
    m = resnext()
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    device = torch.device("cpu")
    model = m.to(device)
    x = torch.randn((3, 512, 512)).to(device)
    # output = model(x)
    # print(model)
    summary(model, (3, 512, 512), device="cup")


# Run the command line:
# cd C:\Users\10295\CV\CenterNet-master
# tensorboard --logdir= ./paint
# Then open http://0.0.0.0:6006/ into your web browser





