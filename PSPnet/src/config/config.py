# 以python 对象的方式定义与网络有关的固定参数

pspnet_resnet50_GPU = {
    # 日志目录
    "log_dir": "./logs/",
    #   输入图片的大小 默认顺序[h,w,c]
    "input_size": [473, 473, 3],
    #   分类个数+1
    #   20+1
    "num_classes": 21,
    "num_classes_ADE": 150,
    # ResNet 提取后的feature大小 仅限VOC2012
    "feature_size": 15,
    #   主干网络预训练权重的使用
    "pretrained": True,
    "backbone": "resnet50",
    #   是否使用辅助分支
    #   会占用大量显存
    "ignore_label": 255,
    # 忽略背景
    "aux_branch": False,
    # 是否分布式训练 默认否
    "run_distribute": False,
    # 学习率相关参数
    "lr_init": 0.01,
    "lr_end": 0.03,
    "lr_max": 0.03,
    "warmup_epochs": 0,
    # 优化器相关参数
    "momentum": 0.9,
    # 网络声明时的其他参数
    "name": "pspnet_resnet50",
    "pretrained_link": "https://download.mindspore.cn/model_zoo/official/cv/resnet/resnet50_v1.5_ascend_0.3.0_cifar10_official_classification_20200718/resnet50.ckpt",
    "pretrained_path": "./data/resnet.ckpt"

}
