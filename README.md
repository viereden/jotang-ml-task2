## 任务9 模型结构图（CatDogCNN）
```mermaid
flowchart LR
    A["输入图片<br/>128×128×3 RGB"] --> B["Conv1 卷积层<br/>Conv2d(3,16,3,padding=1)"]
    B --> C["ReLU激活"]
    C --> D["MaxPool2d 池化<br/>下采样"]

    D --> E["Conv2 卷积层<br/>Conv2d(16,32,3,padding=1)"]
    E --> F["ReLU激活"]
    F --> G["MaxPool2d 池化<br/>下采样"]

    G --> H["Conv3 卷积层<br/>Conv2d(32,64,3,padding=1)"]
    H --> I["ReLU激活"]
    I --> J["MaxPool2d 池化<br/>下采样"]

    J --> K["Flatten 展平"]
    K --> L["Linear 全连接层1"]
    L --> M["ReLU激活"]
    M --> N["Linear 全连接层2"]
    N --> O["输出<br/>2分类：猫 / 狗"]
