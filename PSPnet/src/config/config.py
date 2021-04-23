# 以python 对象的方式定义与网络有关的固定参数

pspnet_resnet50_GPU = {
    #   输入图片的大小 默认顺序[h,w,c]
    "input_size": [473, 473, 3],
    #   分类个数
    "num_classes": 21,
    "num_classes_ADE": 150,
    # ResNet 提取后的feature大小
    "feature_size": 15,
    #   主干网络预训练权重的使用
    "pretrained": True,
    "backbone": "resnet50",
    "ignore_label": 255,
    # 忽略背景
    "aux_branch": False,
    # 学习率相关参数
    "lr_init": 0.01,
    "lr_end": 0.03,
    "lr_max": 0.03,
    "warmup_epochs": 0,
    # 优化器相关参数
    "momentum": 0.9,
    # 网络声明时的其他参数
    "name": "pspnet_resnet50",
    "pretrained_path": "./data/resnet.ckpt"
}
