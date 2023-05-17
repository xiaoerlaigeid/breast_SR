# python test.py --pretrain_model_G check_points/ESRGAN-V3_256/latest_G.pth \
#     --data_path /home/zhikai/save_image_66_1w_256 --lr_size 256


# python test.py --pretrain_model_G check_points/ESRGAN-V3_256/latest_G.pth \
#     --data_path "/home/zhikai/stylegan3/256_stylegan2_1w" --lr_size 256


# python test.py --pretrain_model_G check_points/ESRGAN-V3_256/latest_G.pth \
#     --data_path "/home/zhikai/stylegan3/stylegan2_256_8000_new1w" --lr_size 256


python test.py --pretrain_model_G check_points/ESRGAN-V3_256/latest_G.pth \
    --data_path "/home/zhikai/stylegan3/stylegan2_256_7600_new1w" --lr_size 256

python test.py --pretrain_model_G check_points/ESRGAN-V3_256/latest_G.pth \
    --data_path "/home/zhikai/stylegan3/stylegan2_256_8400_new1w" --lr_size 256
