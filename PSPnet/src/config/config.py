# 以python 对象的方式定义与网络有关的固定参数

pspnet_resnet50_GPU = {
# 日志目录
"log_dir":"../logs/",
#   输入图片的大小 默认顺序[h,w,c]
"input_size":[473,473,3],
#   分类个数+1
#   20+1
"resize_shape":[400,400],
#增广过程中的随机放缩大小
"num_classes":21,
#   建议选项：
#   种类少（几类）时，设置为True
#   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
#   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
"dice_loss": False,
#   主干网络预训练权重的使用
"pretrained" : False,
"backbone" : "resnet50",
#   是否使用辅助分支
#   会占用大量显存
"aux_branch": False,
# 是否分布式训练 默认否
"run_distribute":False,
# 学习率相关参数
"lr_init": .0,
"lr_end": 0.03,
"lr_max": 0.03,
"warmup_epochs": 0,
# 优化器相关参数
"momentum": 0.9,
"weight_decay": 4e-5,
}