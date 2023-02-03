import torch
import torchvision
import matplotlib.pyplot as plt

# Load a pretrained model from torchvision
model = torchvision.models.resnet18(pretrained=False)

# Define a forward hook to get the activations for a given layer
def hook_fn(module, input, output):
    activations = output.detach().cpu().numpy()
    plt.imshow(activations[0, 0])
    plt.axis('off')
    plt.show()

# Register the hook on a particular layer
handle = model.conv1.register_forward_hook(hook_fn)

# Create an input tensor
input_tensor = torch.randn(1, 3, 224, 224)

# Pass the input tensor through the network
model(input_tensor)

# Remove the hook
handle.remove()
