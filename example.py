import torch
import numpy as np
from PIL import Image
import torchturbojpeg as ttjpeg

ttencoder = ttjpeg.Encoder(75,1024*1024)

example_image = Image.open('example_input.jpg')
image_cuda = torch.tensor(np.array(example_image)).permute(2, 0, 1).cuda().contiguous()

bytes_tt = ttencoder.encode(image_cuda)
with open('example_tt.jpg', 'wb') as f:
    f.write(bytes(bytes_tt))