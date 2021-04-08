ESRGAN_config = {
# image setting
"input_size": 32,
"gt_size": 128,
"ch_size": 3,
"scale": 4,
# Generator setting
"G_nf":64,
"G_nb":23,
# discriminator setting
"D_nf":64,
# training setting
"niter": 400000,
"lr_G": 1e-4,
"lr_D": 1e-4,
"lr_steps": [50000, 100000, 200000, 300000],
"lr_rate": 0.5,

"adam_beta1_G": 0.9,
"adam_beta2_G": 0.99,
"adam_beta1_D": 0.9,
"adam_beta2_D": 0.99,

"w_pixel": 1e-2,
"pixel_criterion": "l1",

"w_feature": 1.0,
"feature_criterion":"l1",

"w_gan": 5e-3,
"gan_type": "gan",
# gan | ragan

"save_steps": 5000
}


PSNR_config = {
# image setting
"input_size": 32,
"gt_size": 128,
"ch_size": 3,
"scale": 4,
# Generator setting
"G_nf":64,
"G_nb":23,

# training setting
"niter": 400000,
"lr": 2e-4,
"lr_steps": [200000, 400000, 600000, 800000],
"lr_rate": 0.5,

"adam_beta1_G": 0.9,
"adam_beta2_G": 0.99,

"w_pixel": 1.0,
"pixel_criterion": "l1",

"save_steps": 5000
}