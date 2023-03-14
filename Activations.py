

def initialize_weights(m):
  
  if isinstance(m, nn.Conv2d):
      nn.init.constant_(m.weight.data, 1)
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)

  elif isinstance(m, nn.ConvTranspose2d):
      nn.init.constant_(m.weight.data, 1)
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
          
  elif isinstance(m, nn.BatchNorm2d):
      nn.init.constant_(m.weight.data, 1)
      nn.init.constant_(m.bias.data, 0)

  elif isinstance(m, nn.Linear):
      nn.init.constant_(m.weight.data, 1)
      nn.init.constant_(m.bias.data, 0)

model.apply(initialize_weights)
handle = model.layers[4].register_forward_hook(hook_fn)
# Create an input tensor

input = torch.zeros(1,3, 256, 256)
input[0, :, 128, 128] = torch.tensor([1., 1., 1.])
# input[0, :, 127, 128] = torch.tensor([1., 1., 1.])
# input[0, :, 128, 127] = torch.tensor([1., 1., 1.])
# input[0, :, 127, 127] = torch.tensor([1., 1., 1.])

input_tensor = input
# Pass the input tensor through the network
model.eval()
# model.bias.zero_()
model(input_tensor)

# Remove the hook
handle.remove()
