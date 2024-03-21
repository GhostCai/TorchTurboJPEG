import io
import time
import torch
import torchturbojpeg
from PIL import Image

imgs = [(torch.rand(3, 256, 256)*255).to(torch.uint8).cuda() for _ in range(100)]
ttjpeg = torchturbojpeg.Encoder(75,1024*1024)

torch.cuda.synchronize()
time_start = time.time()
for i in range(100):
    ttjpeg.encode(imgs[i])
torch.cuda.synchronize()
time_end = time.time()
print("torchturbojpeg.decode:", time_end - time_start, "s")

torch.cuda.synchronize()
time_start = time.time()
for i in range(100):
    byte_array = io.BytesIO()
    Image.fromarray(imgs[i].permute(1, 2, 0).cpu().numpy()).save(byte_array, format="JPEG")
    byte_array = byte_array.getvalue()
torch.cuda.synchronize()
time_end = time.time()
print("PIL.Image.save:", time_end - time_start, "s")



    


