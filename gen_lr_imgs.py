import cv2
from glob import glob
from tqdm import tqdm
import random
import time
from threadpool import ThreadPool, makeRequests
import os
# all_list = glob('/home/Data/breast_gen_data/challenge_data/*') + glob('data/celebahq-512/*')

root_dir = '/home/Data/breast_gen_data/challenge_data/'
all_list = []
for root, dirs, files in os.walk(root_dir, topdown=False):
    for name in files:
        all_list.append(os.path.join(root, name))
target_size = 256
# there are 3 quality ranges for each img
quality_ranges = [(15, 75)]
output_path = '/home/Data/breast_gen_data/lr-256'


def saving(path):
    assert '.png' in path
    img = cv2.imread(path)
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    for qr in quality_ranges:
        # quality = int(random.random() * (qr[1] - qr[0]) + qr[0])
        output_path_final = output_path+ '/' + path.split('/')[-2]
        if not os.path.exists(output_path_final):
            print("output_path_final",output_path_final)
            os.mkdir(output_path_final)
        cv2.imwrite(os.path.join(output_path_final,path.split('/')[-1]), img,
                    [int(cv2.IMWRITE_PNG_COMPRESSION), 0]) #.replace('.jpg', '_q%d.jpg' % quality)

# output_path_final = output_path+ '/' + path.split('/')[-2]
# if not os.path.exists(output_path_final):
#     print("output_path_final",output_path_final)
#     os.mkdir(output_path_final)


with tqdm(total=len(all_list), desc='Resizing images') as pbar:
    def callback(req, x):
        pbar.update()
    t_pool = ThreadPool(15)
    requests = makeRequests(saving, all_list, callback=callback)
    for req in requests:
        t_pool.putRequest(req)
    t_pool.wait()