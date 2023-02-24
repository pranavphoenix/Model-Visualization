import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

transform = T.ToPILImage()
img = transform(torch.rand(3, 256, 256))
display(img)
