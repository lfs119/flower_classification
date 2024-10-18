# 导入包
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from model import AlexNet, custom_resnet50
import os
import json
import time
import random
from PIL import Image


# 使用GPU训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(os.path.join("train.log"), "a") as log:
    log.write(str(device) + "\n")

# 数据预处理
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 随机裁剪，再缩放成 224×224
                                 transforms.RandomHorizontalFlip(p=0.5),  # 水平方向随机翻转，概率为 0.5, 即一半的概率翻转, 一半的概率不翻转
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),

    "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

# 获取图像数据集的路径
data_root = os.path.abspath(os.path.join(os.getcwd()))  # get data root path 返回上上层目录
image_path = data_root + "/flower_data"  # flower data_set path

jpgs = random.choices(os.listdir(image_path + "/train/tulips"), k=4)
plt.figure(figsize=(10, 10))
for i in range(1, 5):
    plt.subplot(2,2,i)
    img = Image.open(image_path + "/train/tulips/" + jpgs[i-1])
    img = np.array(img)
    plt.imshow(img)
    plt.axis("off")
    plt.title(jpgs[i-1], fontdict={"size":12})
plt.show()

# 导入训练集并进行预处理
train_dataset = datasets.ImageFolder(root=image_path + "/train",
                                     transform=data_transform["train"])
train_num = len(train_dataset)  # 3306

# 按batch_size分批次加载训练集
train_loader = torch.utils.data.DataLoader(train_dataset,  # 导入的训练集
                                           batch_size=32,  # 每批训练的样本数
                                           shuffle=True,  # 是否打乱训练集
                                           num_workers=0)  # 使用线程数，在windows下设置为0

# 导入、加载 验证集
# 导入验证集并进行预处理
validate_dataset = datasets.ImageFolder(root=image_path + "/val",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)  # 364

# 加载验证集
validate_loader = torch.utils.data.DataLoader(validate_dataset,  # 导入的验证集
                                              batch_size=32,
                                              shuffle=True,
                                              num_workers=0)

# 存储 索引：标签 的字典
# 字典，类别：索引 {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
# 将 flower_list 中的 key 和 val 调换位置
cla_dict = dict((val, key) for key, val in flower_list.items())

# 将 cla_dict 写入 json 文件中
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

# 训练过程
epoch_num = 40
net = custom_resnet50(num_classes=5, init_weights=True)  # 实例化网络（输出类型为5，初始化权重）
net.to(device)  # 分配网络到指定的设备（GPU/CPU）训练
loss_function = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.Adam(net.parameters(), lr=0.0002)  # 优化器（训练参数，学习率）

save_path = f'./ResNet50_{epoch_num}.pth'
best_acc = 0.0
train_losses = []
val_accurate_list = []

for epoch in range(epoch_num):
    ########################################## train ###############################################
    net.train()  # 训练过程中开启 Dropout
    running_loss = 0.0  # 每个 epoch 都会对 running_loss  清零
    time_start = time.perf_counter()  # 对训练一个 epoch 计时

    for step, data in enumerate(train_loader, start=0):  # 遍历训练集，step从0开始计算
        images, labels = data  # 获取训练集的图像和标签
        optimizer.zero_grad()  # 清除历史梯度

        outputs = net(images.to(device))  # 正向传播
        loss = loss_function(outputs, labels.to(device))  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 优化器更新参数
        running_loss += loss.item()

        # 打印训练进度（使训练过程可视化）
        rate = (step + 1) / len(train_loader)  # 当前进度 = 当前step / 训练一轮epoch所需总step
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        with open(os.path.join("train.log"), "a") as log:
            log.write(str("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss)) + "\n")
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    with open(os.path.join("train.log"), "a") as log:
        log.write(str('%f s' % (time.perf_counter() - time_start)) + "\n")
    print('%f s' % (time.perf_counter() - time_start))

    ########################################### validate ###########################################
    net.eval()  # 验证过程中关闭 Dropout
    acc = 0.0
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]  # 以output中值最大位置对应的索引（标签）作为预测输出
            acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        val_accurate_list.append(val_accurate)

        # 保存准确率最高的那次网络参数
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        with open(os.path.join("train.log"), "a") as log:
            log.write(str('[epoch %d] train_loss: %.3f  test_accuracy: %.3f \n' %
                          (epoch + 1, running_loss / step, val_accurate)) + "\n")
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f \n' %
              (epoch + 1, running_loss / step, val_accurate))
with open(os.path.join("train.log"), "a") as log:
    log.write(str('Finished Training') + "\n")
print('Finished Training')
print(best_acc)
# plt.figure(figsize=(10, 5))
# plt.plot(range(1, epoch_num+1), train_losses, label='Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss Over Epochs')
# plt.legend()
# plt.grid(True)
# plt.savefig('training_loss.png')  # 保存图像到文件
# 训练结束后，绘制图表
fig, ax1 = plt.subplots(figsize=(10, 6))
epochs = range(1, len(train_losses) + 1)
# 创建共享x轴的第二个y轴，用于绘制验证准确率
ax2 = ax1.twinx()

# 绘制训练损失，使用左侧的y轴
ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='b')
ax1.tick_params('y', colors='b')
ax1.set_title('Training Loss and Validation Accuracy per Epoch')
ax1.grid(True, which="both", ls="--")

# 绘制验证准确率，使用右侧的y轴
ax2.plot(epochs, val_accurate_list, 'r-', label='Validation Accuracy')
ax2.set_ylabel('Accuracy', color='r')
ax2.tick_params('y', colors='r')

# 设置图例
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))

# 显示图表
plt.tight_layout()
plt.show()