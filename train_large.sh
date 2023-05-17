python train.py --hr_path /home/Data/breast_gen_data/challenge_data/ --lr_path /home/Data/breast_gen_data/lr-256/ \
    --batch_size 2 --lr_size 256 --hr_size 512  --checkpoint_dir check_points/ESRGAN-V3_nf64_nb24_256/ \
    --training_state check_points/ESRGAN-V3_nf96_nb24_256/state/ --niter 200000 --G_nf 64 --nb 16