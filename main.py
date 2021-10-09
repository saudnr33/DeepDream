

# import all needed resources
%matplotlib inline

from PIL import Image
from io import BytesIO
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import requests
from torchvision import transforms, models
from tqdm import tqdm
from glob import glob
from torch.autograd import Variable
from DeepDream import*




Path = ""
img = load_image(Path).to(device)



layers_new = {
              '22': 'relu4_2',
              "26": "relu4_4",
}


model = None
img = deep_dream(img,model, layers,  octave_scale=1.3, pyramid_levels=5, spatial_shift_size = (0, 0), num_gradient_ascent_iterations= 20, lr = 0.0001)
show_image(img, figsize=(12, 8))
# plt.show()
