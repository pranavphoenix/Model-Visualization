import torch
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
----------------------
#Define the model here
-----------------------
model = WaveMix(
    num_classes = 1000,
    depth = 12,
    mult = 2,
    ff_channel = 256,
    final_dim = 256,
    dropout = 0.5
)

model.to(device)

print(mdodel) #to find the names of layers

model.load_state_dict(torch.load('PATH TO MODEL WEIGHTS'))


def feature_map(x):
  x = x.squeeze(0)
  print(x.shape)
  gray_scale = torch.sum(x,0)
  print(gray_scale.shape)
  x = gray_scale / x.shape[0]
  print(x.shape)
  output = x.cpu().detach().numpy()
  return output

#Input in the form C x H X W
#Display Input
input = testset[12][0]  #PATH TO INPUT
plt.imshow(torch.permute(input, (1, 2, 0)))
plt.show() 

#Display Output of first layer
input = input.unsqueeze(0)
output_conv1 = model.conv1(input.to(device))
plt.imshow(feature_map(output_conv1))
plt.show()

#Use the output of previous layer as input to next layer
output_conv2 = model.conv2(output_conv1) 
plt.imshow(feature_map(output_conv2))
plt.show()

#Keep doing this for all layers





