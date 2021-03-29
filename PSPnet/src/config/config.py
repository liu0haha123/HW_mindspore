pspnet_resnet50 = {
# 日志目录
"log_dir":"../logs/",
#   输入图片的大小
"input_size":[473,473,3],
#   分类个数+1
#   20+1
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
#   下采样的倍数
#   16显存占用小
#   8显存占用大
"downsample_factor" : 16
}