import torch
import PIL
import numpy as np
import pickle
from IPython import embed
try:
    import Image
except ImportError:
    from PIL import Image
import torch.nn.functional as F

# from torchvision import models
from torch_backbone import resnet34
from torch import nn
from collections import OrderedDict


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.backbone = resnet34()

        self.relu = nn.ReLU()
        self.shuf = nn.PixelShuffle(2)
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        # self.blur = nn.AvgPool2d(2, stride=1, padding=(1, 0, 1, 0))
        
        self.convert0_bn0 = nn.BatchNorm2d(512)
        self.convert0_conv1 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.convert0_bn1 = nn.BatchNorm2d(1024)
        self.convert0_conv2 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.convert0_bn2 = nn.BatchNorm2d(512)
        self.shuffle0_conv = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.shuffle0_bn = nn.BatchNorm2d(1024)
        self.previous0_bn = nn.BatchNorm2d(256)

        self.convert1_conv1 = nn.Conv2d(512, 768, kernel_size=3, stride=1, padding=1, bias=False)
        self.convert1_bn1 = nn.BatchNorm2d(768)
        self.convert1_conv2 = nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=1, bias=False)
        self.convert1_bn2 = nn.BatchNorm2d(768)
        self.shuffle1_conv = nn.Conv2d(768, 1536, kernel_size=1, stride=1, padding=0, bias=False)
        self.shuffle1_bn = nn.BatchNorm2d(1536)
        self.previous1_bn = nn.BatchNorm2d(128)

        self.convert2_conv1 = nn.Conv2d(512, 768, kernel_size=3, stride=1, padding=1, bias=False)
        self.convert2_bn1 = nn.BatchNorm2d(768)
        self.convert2_conv2 = nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=1, bias=False)
        self.convert2_bn2 = nn.BatchNorm2d(768)
        self.gamma = nn.Parameter(torch.tensor([0.]))
        # self.query = nn.Conv1d(768, 768//8, kernel_size=1, stride=1, padding=0, bias=False)
        # self.key   = nn.Conv1d(768, 768//8, kernel_size=1, stride=1, padding=0, bias=False)
        # self.value = nn.Conv1d(768, 768, kernel_size=1, stride=1, padding=0, bias=False)
        self.query = nn.Conv2d(768, 768//8, kernel_size=1, stride=1, padding=0, bias=False)
        self.key   = nn.Conv2d(768, 768//8, kernel_size=1, stride=1, padding=0, bias=False)
        self.value = nn.Conv2d(768, 768, kernel_size=1, stride=1, padding=0, bias=False)
        self.shuffle2_conv = nn.Conv2d(768, 1536, kernel_size=1, stride=1, padding=0, bias=False)
        self.shuffle2_bn = nn.BatchNorm2d(1536)
        self.previous2_bn = nn.BatchNorm2d(64)

        self.convert3_conv1 = nn.Conv2d(448, 672, kernel_size=3, stride=1, padding=1, bias=False)
        self.convert3_bn1 = nn.BatchNorm2d(672)
        self.convert3_conv2 = nn.Conv2d(672, 672, kernel_size=3, stride=1, padding=1, bias=False)
        self.convert3_bn2 = nn.BatchNorm2d(672)
        self.shuffle3_conv = nn.Conv2d(672, 1344, kernel_size=1, stride=1, padding=0, bias=False)
        self.shuffle3_bn = nn.BatchNorm2d(1344)
        self.previous3_bn = nn.BatchNorm2d(64)
        
        self.final_conv1 = nn.Conv2d(400, 300, kernel_size=3, stride=1, padding=1, bias=False)
        self.final_bn1 = nn.BatchNorm2d(300)
        self.final_conv2 = nn.Conv2d(300, 300, kernel_size=3, stride=1, padding=1, bias=False)
        self.final_bn2 = nn.BatchNorm2d(300)

        self.output_conv1 = nn.Conv2d(300, 1200, kernel_size=1, stride=1, padding=0, bias=True)
        self.output_conv2 = nn.Conv2d(303, 303, kernel_size=3, stride=1, padding=1, bias=True)
        self.output_conv3 = nn.Conv2d(303, 303, kernel_size=3, stride=1, padding=1, bias=True)
        self.output_conv4 = nn.Conv2d(303, 3, kernel_size=1, stride=1, padding=0, bias=True)
        

    # def pad(self, x, c, wh):
    #     x = torch.cat([x, torch.zeros(size=(1,c,1,wh), dtype=x.dtype)], dim=2)
    #     x = torch.cat([x, torch.zeros(size=(1,c,wh+1,1), dtype=x.dtype)], dim=3)
    #     return x

    def self_atten(self, x, query, key, value, gamma):
        f,g,h = query(x),key(x),value(x)
        size = x.size()
        x = x.view(*size[:2],-1)
        f = f.view(*f.size()[:2],-1)
        g = g.view(*g.size()[:2],-1)
        h = h.view(*h.size()[:2],-1)
        # c = torch.sigmoid(c)
        beta = F.softmax(torch.bmm(f.permute(0,2,1).contiguous(), g), dim=1)
        o = gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()

    def forward(self, x):
        # preprocess
        x = x.div(255.0)
        mean = torch.tensor([0.485, 0.456, 0.406])[:,None,None]
        std = torch.tensor([0.229, 0.224, 0.225])[:,None,None]
        x = (x - mean) / std

        # backbone
        x0,x1,x2,x3,x4 = self.backbone(x)

        # step1
        s4 = self.relu(self.convert0_bn0(x4))
        s4 = self.convert0_bn1(self.relu(self.convert0_conv1(s4)))
        s4 = self.convert0_bn2(self.relu(self.convert0_conv2(s4)))
        s4 = self.relu(self.shuffle0_bn(self.shuffle0_conv(s4)))
        # s4 = self.blur(self.pad(self.shuf(s4), 256, 32))
        s4 = self.blur(self.pad(self.shuf(s4)))
        p3 = self.previous0_bn(x3)
        c3 = self.relu(torch.cat([s4, p3], dim=1))
        
        # step2
        s3 = self.convert1_bn1(self.relu(self.convert1_conv1(c3)))
        s3 = self.convert1_bn2(self.relu(self.convert1_conv2(s3)))
        s3 = self.relu(self.shuffle1_bn(self.shuffle1_conv(s3)))
        # s3 = self.blur(self.pad(self.shuf(s3), 384, 64))
        s3 = self.blur(self.pad(self.shuf(s3)))
        p2 = self.previous1_bn(x2)
        c2 = self.relu(torch.cat([s3, p2], dim=1))

        # step3
        s2 = self.convert2_bn1(self.relu(self.convert2_conv1(c2)))
        s2 = self.convert2_bn2(self.relu(self.convert2_conv2(s2)))
        s2 = self.self_atten(s2, self.query, self.key, self.value, self.gamma)
        s2 = self.relu(self.shuffle2_bn(self.shuffle2_conv(s2)))
        # s2 = self.blur(self.pad(self.shuf(s2), 384, 128))
        s2 = self.blur(self.pad(self.shuf(s2)))
        p1 = self.previous2_bn(x1)
        c1 = self.relu(torch.cat([s2, p1], dim=1))

        # step4
        s1 = self.convert3_bn1(self.relu(self.convert3_conv1(c1)))
        s1 = self.convert3_bn2(self.relu(self.convert3_conv2(s1)))
        s1 = self.relu(self.shuffle3_bn(self.shuffle3_conv(s1)))
        # s1 = self.blur(self.pad(self.shuf(s1), 336, 256))
        s1 = self.blur(self.pad(self.shuf(s1)))
        p0 = self.previous3_bn(x0)
        c0 = self.relu(torch.cat([s1, p0], dim=1))

        # step5
        c = self.final_bn1(self.relu(self.final_conv1(c0)))
        c = self.final_bn2(self.relu(self.final_conv2(c)))
        
        # step6
        c = self.shuf(self.relu(self.output_conv1(c)))
        c = torch.cat([c, x], dim=1)
        c_res = self.relu(self.output_conv2(c))
        c_res = self.relu(self.output_conv3(c_res))
        c += c_res
        c = self.output_conv4(c)
        c = torch.sigmoid(c)
        c = c*6-3

        # postprocess
        c = c * std + mean
        c = c.clamp(min=0,max=1)
        c *= 255

        return c



def remove_bn_static(st):
    st_new = OrderedDict()
    for k in st:
        if 'num_batches_tracked' not in k:
            st_new[k] = st[k]
    return st_new

def gamma_first(st):
    gamma_key = [s for s in st if 'gamma' in s]
    assert len(gamma_key) == 1
    st_new = OrderedDict()
    st_new[gamma_key[0]] = st[gamma_key[0]]
    for k in st:
        if 'gamma' not in k:
            st_new[k] = st[k]
    return st_new

def map_key(st, keys):
    st_keys = list(st.keys())
    st_new = OrderedDict()
    for st_k,k in zip(st_keys, keys):
        print(st_k, '-------->', k)
        st_new[k] = st[st_k]
    for st_k in st_keys[len(keys):]:
        st_new[st_k] = st[st_k]
    return st_new

def weight_first(st):
    st_new = OrderedDict()
    cached = False
    for k in st:
        if 'bias' in k and any(s in k for s in ['layers.8', 'layers.10', 'layers.11']):
            k_ = k
            cached = True
        else:
            st_new[k] = st[k]
            if cached:
                st_new[k_] = st[k_]
    assert len(st) == len(st_new)
    return st_new

def conv1to2(st):
    st_new = OrderedDict()
    for k in st:
        if any(s in k for s in ['query', 'key', 'value']):
            st_new[k] = st[k][:,:,:,None]
        else:
            st_new[k] = st[k]
    assert len(st) == len(st_new)
    return st_new

st = torch.load('net.param')
st = remove_bn_static(st)
st = gamma_first(st)
st = weight_first(st)
st = conv1to2(st)

net = Net()
st_net = net.state_dict()
st_net = remove_bn_static(st_net)
st = map_key(st, list(st_net.keys()))
print(len(st_net), len(st))
net.load_state_dict(st)
# embed()

# out = net(torch.zeros(size=(1,3,512,512)))
# print(out.shape)

path = 'test_images/image.png'
# data = torch.from_numpy(np.asarray(PIL.Image.open(path).convert('RGB')))
# data = torch.from_numpy(np.asarray(PIL.Image.open(path).convert('RGB').resize((128,128))))
data = torch.from_numpy(np.asarray(PIL.Image.open(path).convert('RGB').resize((256,256))))
data = data.permute((2,0,1))[None].float()
# data = data.div(255.0)
# mean = torch.tensor([0.485, 0.456, 0.406])[:,None,None]
# std = torch.tensor([0.229, 0.224, 0.225])[:,None,None]
# data = (data - mean) / std

out = net(data)
# out = out * std + mean
# out = out.clamp(min=0,max=1)
# out *= 255
res = out[0].permute(1,2,0).detach().numpy().astype(np.uint8)
PIL.Image.fromarray(res).save('res_torch.jpg')

output_onnx = 'deoldify.onnx'
input_names = ["input"]
output_names = ["out"]
inputs = data
diff = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False, input_names=input_names, output_names=output_names, keep_initializers_as_inputs=True, opset_version=11)
print(diff.abs().max())

# embed()