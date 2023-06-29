import os
from tqdm import tqdm
from natsort import natsorted
import time

def command_run_byte_track(path_video,exp_file,model = "pretrained/bytetrack_x_mot20.tar",fps = 20):

    return f"python3 tools/demo_track.py video --path {path_video} -f {exp_file} -c {model} --fps {fps} --fp16 --fuse --save_result"



folder_path_videos = "/media/anlabadmin/data_ubuntu/yolox/many_people"
list_videos = natsorted(os.listdir(folder_path_videos))
exp_file = "/media/anlabadmin/data_ubuntu/yolox/ByteTrack/exps/example/mot/yolox_x_mix_mot20_ch.py"
model = "/media/anlabadmin/data_ubuntu/yolox/Weight/Weight10epochs/latest_ckpt.pth_yolo_x_9.tar"



for video in tqdm(list_videos):
    start = time.time()
    path_video = os.path.join(folder_path_videos ,video)
    command = command_run_byte_track(path_video,exp_file,model = model,fps = 30)    
    os.system(command = command)
    end = time.time()
    print(f"{video}||Total minute {(end -start)/60} minutes")



