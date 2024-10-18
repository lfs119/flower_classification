import torch
from model import AlexNet,  custom_resnet50
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json


# 预处理
data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load image
img = Image.open("flower_data/val/roses/145862135_ab710de93c_n.jpg")
plt.imshow(img)
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# read class_indict
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# create model
model = custom_resnet50(num_classes=5)
# load model weights
model_weight_path = "./ResNet50_40.pth"
#, map_location='cpu'
model.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

# 关闭 Dropout
model.eval()
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img))     # 将输出压缩，即压缩掉 batch 这个维度
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
print(class_indict[str(predict_cla)], predict[predict_cla].item())
plt.title(f'class: {class_indict[str(predict_cla)]}, probability: {predict[predict_cla].item()}')
plt.show()
