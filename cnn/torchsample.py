from PIL import Image
import torchvision.transforms.functional as TF
import torch
from cnn.dedup_encoder import DE


image = Image.open(r"..\data\hymenoptera_data\train\ants\24335309_c5ea483bb8.jpg")
x = TF.to_tensor(image)
#x = torch.unsqueeze(x, 1)
print(x.shape)
mymodel = DE()
y = mymodel.forward(x)
print(y.shape)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



