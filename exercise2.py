import os
cat_dir = "./catdog_data/train/cats"
dog_dir = "./catdog_data/train/dogs"
print("cats里文件：", os.listdir(cat_dir))
print("dogs里文件：", os.listdir(dog_dir))


import random
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ===================== 任务1：读取数据集，统计数量、查看样本 =====================
# 定义猫狗图片文件夹路径
cat_dir = "./catdog_data/train/cats"
dog_dir = "./catdog_data/train/dogs"

# 拿到所有图片文件名列表
cat_img_paths = [os.path.join(cat_dir, name) for name in os.listdir(cat_dir)]
dog_img_paths = [os.path.join(dog_dir, name) for name in os.listdir(dog_dir)]

cat_num = len(cat_img_paths)
dog_num = len(dog_img_paths)
print(f"【任务1】猫图片总数：{cat_num}")
print(f"【任务1】狗图片总数：{dog_num}")

# 随机选取6张图片可视化
plt.figure(figsize=(12, 6))
all_img_paths = cat_img_paths + dog_img_paths
sample_paths = random.sample(all_img_paths, 6)

for idx, img_path in enumerate(sample_paths):
    img = Image.open(img_path).convert("RGB")
    plt.subplot(2, 3, idx+1)
    plt.imshow(img)
    if "cat" in img_path.lower():
        plt.title("猫")
    else:
        plt.title("狗")
    plt.axis("off")
plt.tight_layout()
plt.savefig("catdog_original_sample.png")
plt.close()
print("【任务1】原始样本图已保存 catdog_original_sample.png")

# 查看图片尺寸信息
print("\n【任务1】样本图片尺寸信息：")
for p in sample_paths[:3]:
    im = Image.open(p)
    w, h = im.size
    print(f"图片: {os.path.basename(p)} | 宽:{w}, 高:{h}")


# ===================== 任务2：自定义数据集 + 预处理，构造Dataloader =====================
# 1. 定义预处理流水线
transform = transforms.Compose([
    transforms.Resize((128, 128)),   # 统一缩放到128×128
    transforms.ToTensor(),           # 转Tensor，HWC→CHW，像素0~255映射到0~1
])

# 2. 自定义Dataset类
class CatDogDataset(Dataset):
    def __init__(self, img_path_list, transform=None):
        self.img_paths = img_path_list
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        path = self.img_paths[index]
        img = Image.open(path).convert("RGB")
        # 打标签：猫=0，狗=1
        if "cat" in path.lower():
            label = 0
        else:
            label = 1
        if self.transform is not None:
            img = self.transform(img)
        return img, label

# =================划分训练集、验证集、测试集=================
from torch.utils.data import random_split

# 完整数据集
full_dataset = CatDogDataset(all_img_paths, transform=transform)

# 按70%训练，15%验证，15%测试划分
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

print(f"【任务2】训练集:{len(train_dataset)}张，验证集:{len(val_dataset)}张，测试集:{len(test_dataset)}张")

# 生成3个DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

print(f"【任务2】训练集:{len(train_dataset)}张，验证集:{len(val_dataset)}张，测试集:{len(test_dataset)}张")
print(f"【任务2】Dataloader，每个batch=8")


# 取出一个batch，查看预处理后tensor形状，对比预处理前后
one_batch_img, one_batch_label = next(iter(train_loader))
print(f"\n【任务2】一个batch的图片tensor形状: {one_batch_img.shape}")
# shape: [batch_size, channel, height, width] → [8,3,128,128]

# 可视化预处理之后的图片
plt.figure(figsize=(12,6))
# 拿到当前batch实际图片数量，最多画4张
max_draw = one_batch_img.shape[0]
for i in range(max_draw):
    # tensor CHW -> numpy HWC 用于plt展示
    img_np = one_batch_img[i].permute(1,2,0).numpy()
    plt.subplot(2,2,i+1)
    plt.imshow(img_np)
    cls_name = "猫" if one_batch_label[i]==0 else "狗"
    plt.title(f"预处理后:{cls_name}")
    plt.axis("off")
plt.tight_layout()
plt.savefig("catdog_preprocessed_sample.png")
plt.close()
print("【任务2】预处理后样本图已保存 catdog_preprocessed_sample.png")


# ===================== 任务3：搭建CNN卷积神经网络 =====================
import torch.nn as nn
import torch.nn.functional as F

# 搭建CNN模型类
class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()
        # 卷积块1：卷积层 + 激活 + 池化
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 池化层，2×2，步长2
        # 卷积块2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # 卷积块3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # 全连接层（分类头）
        self.fc1 = nn.Linear(64 * 16 * 16, 128) # 128输入图，经过3次池化，尺寸16×16
        self.fc2 = nn.Linear(128, 2) # 输出2类：猫(0)、狗(1)

    def forward(self, x):
        # 前向传播，定义数据流动的路径
        x = self.pool(F.relu(self.conv1(x))) # conv1 -> relu激活 -> 池化
        x = self.pool(F.relu(self.conv2(x))) # conv2 -> relu激活 -> 池化
        x = self.pool(F.relu(self.conv3(x))) # conv3 -> relu激活 -> 池化
        # 展平：把多维张量拉成一维，送入全连接层
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 实例化模型
model = CatDogCNN()
print("\n【任务3】CNN网络结构：")
print(model)


# 拿一个batch举例测试，可以看每层输出tensor形状
test_img_batch, _ = next(iter(train_loader))
output = model(test_img_batch)
print(f"\n【任务3】输入batch tensor形状 {test_img_batch.shape}")
print(f"【任务3】模型最终输出tensor形状 {output.shape}")

import torch
import torch.nn as nn
import torch.optim as optim

# 超参数
epoch_num = 10
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

best_val_acc = 0.0
best_model_path = "best_model.pth"

# 记录loss、acc，
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

for epoch in range(epoch_num):
    # =========训练阶段=========
    model.train()
    train_total = 0
    train_correct = 0
    train_running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (preds == labels).sum().item()
    
    train_avg_loss = train_running_loss / len(train_loader)
    train_avg_acc = train_correct / train_total
    train_loss_list.append(train_avg_loss)
    train_acc_list.append(train_avg_acc)
    # 训练结束后，保存模型权重
    torch.save(model.state_dict(), "best_model.pth")


    # =========验证阶段=========
    model.eval()
    val_total = 0
    val_correct = 0
    val_running_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, preds = torch.max(outputs.data,1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()

if len(val_loader) > 0:
    val_avg_loss = val_running_loss / len(val_loader)
    val_avg_acc = val_correct / val_total
    val_loss_list.append(val_avg_loss)
    val_acc_list.append(val_avg_acc)
else:
    val_avg_loss = 0.0
    val_avg_acc = 0.0
    print("警告：验证集val_loader为空！没有读取到验证图片")
    val_loss_list.append(val_avg_loss)
    val_acc_list.append(val_avg_acc)


    if len(val_loader) > 0:
        val_avg_loss = val_running_loss / len(val_loader)
        val_avg_acc = val_correct / val_total
    else:
        val_avg_loss = 0
        val_avg_acc = 0
        print("⚠️警告：验证集为空！检查数据集图片")

    val_loss_list.append(val_avg_loss)
    val_acc_list.append(val_avg_acc)

    # =====================【任务8：保存最优模】=====================
    if val_avg_acc > best_val_acc:
        best_val_acc = val_avg_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ 更新最优模型 | Epoch:{epoch+1}, Best Val Acc:{best_val_acc:.4f}")
    # ============================================================================

    print(f"Epoch [{epoch+1}/{epoch_num}] "
          f"Train Loss:{train_avg_loss:.4f}, Train Acc:{train_avg_acc:.4f} | "
          f"Val Loss:{val_avg_loss:.4f}, Val Acc:{val_avg_acc:.4f}")

print("训练全部结束，最优模型已保存至", best_model_path)




# ===================== 任务4：跟踪每层Tensor形状变化 =====================
# 重新定义带形状打印的CNN
class CatDogCNN_ViewShape(nn.Module):
    def __init__(self):
        super(CatDogCNN_ViewShape, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        print(f"输入图片张量shape: {x.shape}")
        # 格式：[batch, channel, height, width]

        x = self.conv1(x)
        print(f"经过conv1卷积后: {x.shape}")
        x = F.relu(x)
        print(f"经过relu1激活后: {x.shape}") # 激活不改变shape，只修改数值
        x = self.pool(x)
        print(f"经过pool1池化后: {x.shape}")

        x = self.conv2(x)
        print(f"经过conv2卷积后: {x.shape}")
        x = F.relu(x)
        print(f"经过relu2激活后: {x.shape}")
        x = self.pool(x)
        print(f"经过pool2池化后: {x.shape}")

        x = self.conv3(x)
        print(f"经过conv3卷积后: {x.shape}")
        x = F.relu(x)
        print(f"经过relu3激活后: {x.shape}")
        x = self.pool(x)
        print(f"经过pool3池化后: {x.shape}")

        x = x.view(-1, 64 * 16 * 16)
        print(f"展平flatten之后: {x.shape}")

        x = F.relu(self.fc1(x))
        print(f"经过fc1全连接层: {x.shape}")
        x = self.fc2(x)
        print(f"模型最终输出: {x.shape}")
        return x

# 实例化这个查看维度的模型
model_view = CatDogCNN_ViewShape()

# 取出【单张图片】
single_img, single_label = next(iter(train_loader))
single_img = single_img[0:1, :, :, :] # 只取第0张，保持维度 [1,3,128,128]

print("\n=====【任务4】单张图片流过CNN，各层张量维度 =====")
out = model_view(single_img)





# ========== 任务5：测试集图片卷积特征可视化 ==========
import matplotlib.pyplot as plt
import torch

# 存储卷积输出特征图
feature_maps = []
def hook_forward(module, input, output):
    feature_maps.append(output)

# 实例化网络
model = CatDogCNN()
# 注册钩子，捕获第一层卷积 Conv1 的输出
hook_handle = model.conv1.register_forward_hook(hook_forward)

# 【从测试集取图片】
test_images, test_labels = next(iter(test_loader))
# 取测试集第一张图片，保留batch维度
test_single_img = test_images[0:1, :, :, :]

# 前向推理，不计算梯度
with torch.no_grad():
    _ = model(test_single_img)

# 取出第一层卷积输出
conv1_feature = feature_maps[0][0]

# 绘制多个通道特征图
plt.figure(figsize=(14, 7))
for ch in range(16):
    plt.subplot(4, 4, ch+1)
    feat = conv1_feature[ch].cpu().numpy()
    plt.imshow(feat, cmap="gray")
    plt.title(f"通道{ch+1}")
    plt.axis("off")

plt.suptitle("任务5：测试集图片，第一层卷积特征图")
plt.tight_layout()
plt.savefig("task5_test_feature_map.png")
plt.close()
hook_handle.remove()

print("任务5执行完成，图片保存 task5_test_feature_map.png")



# ================== 任务6：数据增强对照实验 ==================
import torchvision.transforms as transforms
from torchvision.utils import save_image

# 1. 定义两套预处理：【无增强】 和 【带增强】
transform_no_aug = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

transform_with_aug = transforms.Compose([
    transforms.Resize((140,140)),
    transforms.RandomResizedCrop((128,128), scale=(0.7,1.0)), # 随机裁剪再缩放到128
    transforms.RandomHorizontalFlip(p=0.5), # 50%概率水平翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 颜色扰动
    transforms.ToTensor()
])

# 拿测试集一张原图
raw_img, raw_label = next(iter(test_loader))
img_tensor_origin = raw_img[0]
# tensor -> PIL图像，用来喂给transform
from torchvision.transforms import ToPILImage
to_pil = ToPILImage()
pil_img = to_pil(img_tensor_origin)

# 分别做两种预处理
img_no_aug = transform_no_aug(pil_img)
img_aug = transform_with_aug(pil_img)

# 绘图对比
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(img_no_aug.permute(1,2,0).numpy())
plt.title("无数据增强")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(img_aug.permute(1,2,0).numpy())
plt.title("带增强：翻转+裁剪+颜色扰动")
plt.axis("off")

plt.tight_layout()
plt.savefig("task6_aug_compare.png")
plt.close()
print("【任务6】数据增强对照图保存完成 task6_aug_compare.png")



# ========== 任务7：查找预测错误样本 ==========
import torch
import matplotlib.pyplot as plt

model.eval()  # 模型切评估模式
wrong_img_list = []
wrong_label_gt = []   # ground truth真实标签
wrong_label_pred = [] # 模型预测标签

with torch.no_grad():
    for imgs, labels in test_loader:
        outputs = model(imgs)
        pred = torch.argmax(outputs, dim=1)
        # 找出预测不等于真实标签的下标
        error_mask = pred != labels
        
        # 保存出错样本
        if torch.any(error_mask):
            wrong_im = imgs[error_mask]
            wrong_gt = labels[error_mask]
            wrong_p = pred[error_mask]
            
            for w_img, wg, wp in zip(wrong_im, wrong_gt, wrong_p):
                wrong_img_list.append(w_img)
                wrong_label_gt.append(wg.item())
                wrong_label_pred.append(wp.item())

        # 找到几张错图就提前停止，不用遍历全部测试集
        if len(wrong_img_list) >= 4:
            break

# 可视化错误样本
plt.figure(figsize=(10, 6))
for idx in range(len(wrong_img_list)):
    plt.subplot(2,2,idx+1)
    # tensor还原图片：CHW -> HWC，反归一化
    img_tensor = wrong_img_list[idx].cpu()
    img = img_tensor.permute(1,2,0).numpy()
    # 把归一化的图片还原到0~1区间
    img = (img - img.min())/(img.max()-img.min())
    plt.imshow(img)
    
    gt_name = "猫" if wrong_label_gt[idx]==0 else "狗"
    pred_name = "猫" if wrong_label_pred[idx]==0 else "狗"
    plt.title(f"真实:{gt_name}, 预测:{pred_name}")
    plt.axis("off")

plt.suptitle("模型预测错误样本")
plt.tight_layout()
plt.savefig("task7_wrong_samples.png")
plt.close()

print(f"一共找到 {len(wrong_img_list)} 张预测错误图片，图片保存 task7_wrong_samples.png")


# ========= 任务8==========

from PIL import Image
import torchvision.transforms as transforms



# --------推理预处理--------
# 复制训练阶段用的transform，不能随便改参数
infer_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_single_img(img_path, model):
    # 切换模型到评估模式，关闭dropout、BN的训练行为
    model.eval()
    # 关闭梯度计算，推理不需要求梯度，节省内存、提速
    with torch.no_grad():
        # 读取图片
        img = Image.open(img_path).convert("RGB")
        # 执行预处理
        img_tensor = infer_transform(img)
        # 增加batch维度：模型输入格式 [batch, channel, H, W]
        img_tensor = img_tensor.unsqueeze(0)
        
        # 模型推理
        pred_out = model(img_tensor)
        # 取出概率最大的类别索引
        pred_idx = torch.argmax(pred_out, dim=1)
        # 映射标签
        class_map = {0:"猫",1:"狗"}
        result = class_map[int(pred_idx)]
    return result

# =========调用示例=========
test_img_path = "catdog_data/train/cats/cat1.jpg" 
res = predict_single_img(test_img_path, model)
print(f"模型预测结果：{res}")


