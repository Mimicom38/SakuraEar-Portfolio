import torch
import torchvision
from torchvision import transforms

image_size = 224
mean = (0.485, 0.456, 0.406) 
std = (0.229, 0.224, 0.225) 

data_transform = {
  'train': transforms.Compose([ 
    transforms.RandomResizedCrop( 
      image_size, scale=(0.5,1.0)
    ),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=[-15, 15]),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.RandomErasing(0.5),
  ]),
  'val': transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)

  ])

}
