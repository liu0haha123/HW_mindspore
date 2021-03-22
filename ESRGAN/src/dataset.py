from PIL import Image
import os
from mindspore.dataset.vision import c_transforms,py_transforms

class ESRGAN_Dataset():
    def __init__(self,data_path):
        self.data_path = data_path
        self.image_file_name = sorted(os.listdir(os.path.join(self.data_path,"hr")))

    def __getitem__(self, item):
        file_name = self.image_file_name[item]
        high_resolution = Image.open(os.path.join('datasets', 'hr', file_name)).convert('RGB')
        low_resolution = Image.open(os.path.join('datasets', 'lr', file_name)).convert('RGB')
        images = (high_resolution,low_resolution)
        return images

    def __len__(self):
        return len(self.image_file_name)

def Totensor():
    return py_transforms.ToTensor()
def decode_ops():
    return py_transforms.Decode()
def Random_VFlip(p=0.5):
    return py_transforms.RandomVerticalFlip(p)
def Random_HFlip(p=0.5):
    return py_transforms.RandomHorizontalFlip(p)
