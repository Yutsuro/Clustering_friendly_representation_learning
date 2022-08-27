import torchvision

import torch

from torch2trt import torch2trt

print("check0")

model = torchvision.models.resnet18(pretrained=True).cuda().half().eval()

print("check1")

data = torch.randn((1, 3, 224, 224)).cuda().half()

print("check2")

model_trt = torch2trt(model, [data], fp16_mode=True)

print("check3")

output_trt = model_trt(data)

output = model(data)

print(output.flatten()[0:10])
print(output_trt.flatten()[0:10])
print('max error: %f' % float(torch.max(torch.abs(output - output_trt))))

torch.save(model_trt.state_dict(), 'resnet18_trt.pth')

from torch2trt import TRTModule

model_trt = TRTModule()

model_trt.load_state_dict(torch.load('resnet18_trt.pth'))