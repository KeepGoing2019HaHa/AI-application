import torch
import PIL
import numpy as np
import pickle
from collections import OrderedDict
from IPython import embed
try:
    import Image
except ImportError:
    from PIL import Image


net = torch.load('net.pth')
st = net.state_dict()
st_new = {}
for k in st:
    if 'num_batches_tracked' in k:
        print(k)
        st_new[k] = torch.tensor(0.0)
    else:
        st_new[k] = st[k]
net.load_state_dict(st_new)
# embed()

path = 'test_images/image.png'
data = torch.from_numpy(np.asarray(PIL.Image.open(path).convert('RGB')))
data = data.permute((2,0,1))[None].float()

data = data.div(255.0)
mean = torch.tensor([0.485, 0.456, 0.406])[:,None,None]
std = torch.tensor([0.229, 0.224, 0.225])[:,None,None]
data = (data - mean) / std

with open('input.pkl', 'wb') as fp:
    pickle.dump(data.numpy(), fp)
# embed()

out = net(data)

with open('output.pkl', 'wb') as fp:
    pickle.dump(out.detach().numpy(), fp)

out = out * std + mean
out = out.clamp(min=0,max=1) 
out *= 255


res = out[0].permute(1,2,0).detach().numpy().astype(np.uint8)
PIL.Image.fromarray(res).save('res.jpg')