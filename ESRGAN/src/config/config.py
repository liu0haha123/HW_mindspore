ESRGAN_config = {
    # image setting
    "input_size": 32,
    "gt_size": 128,
    "ch_size": 3,
    "scale": 4,
    # Generator setting
    "G_nf": 64,
    "G_nb": 23,
    # discriminator setting
    "D_nf": 64,
    # training setting
    "niter": 400000,
    "lr_G": [1e-4, 5e-5, 2e-5, 1e-5],
    "lr_D": [1e-4, 5e-5, 2e-5, 1e-5],
    "lr_steps": [50000, 100000, 200000, 300000],

    "w_pixel": 1e-2,
    "w_feature": 1.0,
    "w_gan": 5e-3,

    "save_steps": 5000,
    "down_factor": 4,
    "vgg_pretrain_path":"./src/model/vgg19_ImageNet.ckpt"
}


PSNR_config = {
    # image setting
    "input_size": 32,
    "gt_size": 128,
    "ch_size": 3,
    "scale": 4,
    # Generator setting
    "G_nf": 64,
    "G_nb": 23,

    # training setting
    "niter": 400000,
    "lr": [2e-4, 1e-4, 5e-5, 2e-5],
    "lr_steps": [200000, 400000, 600000, 800000],

    "save_steps": 5000,
    "down_factor": 4
}
