# ============ 任务二 ============ 
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

# 1. 读取图片，把图片放在和代码同一个文件夹里，修改图片文件名
img_pil = Image.open("road.jpg")  

# 转为numpy数组 HWC格式：高度H，宽度W，通道C
img_np = np.array(img_pil)

from PIL import Image
img = Image.fromarray(img_np)
img = img.resize((320,180)) 
img_np = np.array(img)

print("===== Numpy数组信息 HWC =====")
print(f"shape(H,W,C) = {img_np.shape}")
print(f"数据类型dtype = {img_np.dtype}")
print(f"像素最大值 = {img_np.max()}")
print(f"像素最小值 = {img_np.min()}")

# 取一个像素，比如y=100行，x=200列
y_pos, x_pos = 100, 200
pixel_rgb = img_np[y_pos, x_pos]
print(f"坐标({y_pos},{x_pos}) 像素RGB值：{pixel_rgb}")
print("含义：[R通道数值, G通道数值, B通道数值]")

# 转换为PyTorch Tensor，并且转为CHW格式（模型输入标准格式）
img_tensor = torch.from_numpy(img_np).permute(2,0,1)

print("\n===== Torch Tensor信息 CHW =====")
print(f"tensor shape(C,H,W) = {img_tensor.shape}")
print(f"tensor dtype = {img_tensor.dtype}")

# plt.imshow(img_pil)
# plt.title("原图")
# plt.show()



# =========== 任务四 =========== 
import numpy as np

def conv2d_manual(image: np.ndarray, kernel: np.ndarray, stride=1, pad_h=0, pad_w=0):
    """
    手写2D卷积，仅numpy实现，不调用任何现成卷积函数
    :param image: 输入图片数组，支持HWC(RGB) / HW(灰度图)
    :param kernel: 卷积核，二维矩阵，例如3×3
    :return: 卷积输出图片数组
    """
    # 判断是灰度图(H,W)还是RGB图(H,W,C)
    if len(image.shape) == 3:
        h, w, c = image.shape
    else:
        h, w = image.shape
        c = 1
        # 灰度图增加通道维度，统一逻辑
        image = image.reshape(h, w, c)

    # 获取卷积核尺寸
    k_h, k_w = kernel.shape
    # 计算padding，实现same卷积，输出尺寸和原图一样
    pad_h = k_h // 2
    pad_w = k_w // 2

    # 给图片四周填充0
    img_pad = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0,0)), mode="constant")
    # 创建空白输出数组
    output = np.zeros_like(image, dtype=np.float32)

    # 双重循环：从上到下，从左到右滑动卷积核
    for y in range(h):
        for x in range(w):
            # 取出当前卷积核覆盖的图片区域
            patch = img_pad[y : y + k_h, x : x + k_w, :]
            # 对每个通道单独卷积：对应位置相乘，求和
            for ch in range(c):
                output[y, x, ch] = np.sum(patch[:,:,ch] * kernel)
    
    # 限制像素数值在0~255，防止越界
    output = np.clip(output, 0, 255)
    # 转回uint8图片格式
    output = output.astype(np.uint8)
    return output

# 使用示例：构造测试图片 + 测试卷积核，调用上面的conv2d_manual
import numpy as np

# 造一个简单小测试图
test_img = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])
# 简单3×3卷积核
test_kernel = np.array([
    [1,0,-1],
    [1,0,-1],
    [1,0,-1]
])

# 调用手写的卷积函数
result = conv2d_manual(test_img, test_kernel, stride=1, pad_h=0, pad_w=0)
print("卷积输出结果：")
print(result)




# ========== 任务五==========
# 1. 均值模糊核（3×3）：把邻域像素取平均，可以把图像变平滑，降噪
kernel_mean = np.array([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
])

# 2. 锐化核：增强边缘细节，可以让图片轮廓更清晰
kernel_sharpen = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

# 3. Sobel边缘检测（竖直边缘）：提取垂直方向轮廓
kernel_sobel_y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

# 4. 简易高斯模糊3×3（近似高斯，比均值模糊更自然的平滑）
kernel_gauss = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
]) / 16

# 依次调用手写卷积函数，生成4张结果图
img_mean = conv2d_manual(img_np, kernel_mean)
img_sharpen = conv2d_manual(img_np, kernel_sharpen)
img_sobel = conv2d_manual(img_np, kernel_sobel_y)
img_gauss = conv2d_manual(img_np, kernel_gauss)

# 绘图展示4张结果
import matplotlib.pyplot as plt

plt.figure(figsize=(14,9))
plt.subplot(2,2,1)
plt.title("均值模糊")
plt.imshow(img_mean)

plt.subplot(2,2,2)
plt.title("锐化")
plt.imshow(img_sharpen)

plt.subplot(2,2,3)
plt.title("Sobel竖直边缘")
plt.imshow(img_sobel)

plt.subplot(2,2,4)
plt.title("高斯模糊")
plt.imshow(img_gauss)

plt.tight_layout()
plt.savefig("conv_result.png") 
plt.close() 
print("卷积结果图片已保存 conv_result.png")
