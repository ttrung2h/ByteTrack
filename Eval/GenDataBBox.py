import argparse
import os
from tqdm import tqdm
import os.path as osp
import time
import cv2
import torch
import sys
from loguru import logger
sys.path.append('ByteTrack')
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
from natsort import natsorted
import json
device = torch.device("cuda")


ckpt_file = '/media/anlabadmin/data_ubuntu/yolox/Weight/Weight10epochs/latest_ckpt.pth_yolo_x_9.tar'
exp_file = '/media/anlabadmin/data_ubuntu/yolox/ByteTrack/exps/example/mot/yolox_x_mix_mot20_ch.py'
exp = get_exp(exp_file, None)
model = exp.get_model().to(device)
logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
model.eval()
logger.info("loading checkpoint")
ckpt = torch.load(ckpt_file, map_location="cpu")
# load the model state dict
model.load_state_dict(ckpt["model"])
logger.info("loaded checkpoint done.")
model = model.half()
trt_file = None
decoder = None
rgb_means = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
test_size = exp.test_size
def coco_to_yolo(x1, y1, w, h, image_w, image_h):
    if x1<0:
        x1=0
    if y1<0:
        y1=0
    return [((2*x1 + w)/(2*image_w)) , ((2*y1 + h)/(2*image_h)), w/image_w, h/image_h]
def predictBBox(pathImg,foldersave):
    img = cv2.imread(pathImg)
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    img, ratio = preproc(img, test_size, rgb_means, std)
    img = torch.from_numpy(img).unsqueeze(0).float().to(device)
    img = img.half()
    rs = []
    with torch.no_grad():
        outputs = model(img)
        if decoder is not None:
            outputs = decoder(outputs, dtype=outputs.type())
        outputs = postprocess(
            outputs, exp.num_classes, exp.test_conf, exp.nmsthre
        )
    outputs = outputs[0]
    
    # set up for outputs[0] is None
    results = []


    if outputs != None:
        if outputs.shape[1] == 5:
            scores = outputs[:, 4]
            bboxes = outputs[:, :4]
        else:
            output_results = outputs.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        scale = min(test_size[0] / float(height), test_size[1] / float(width))
        bboxes /= scale
        for i in range(len(scores)):
            if scores[i]>=0.5:
                x1,x2,w,h = coco_to_yolo(bboxes[i][0],bboxes[i][1], bboxes[i][2]-bboxes[i][0], bboxes[i][3]-bboxes[i][1], width, height)
                results.append(f"{osp.basename(pathImg)[:-4]} {scores[i]:.2f} {bboxes[i][0]} {bboxes[i][1]} {bboxes[i][2]} {bboxes[i][3]}\n")
        
        with open(os.path.join(foldersave,osp.basename(pathImg).split('.')[0]+'.txt'), 'w') as f:
            f.writelines(results)  
    return results


# camera = 'ch08'
camera_list = ["ch04"]
# camera_list = ["hard_yolox_nano_weight_base"]
for camera in camera_list:
    print(f"====================={camera}==========================")
    # set up folder contain result
    os.mkdir('/media/anlabadmin/data_ubuntu/yolox/Result_detection/model_x_10epochs_1800frames/'+camera)
    folder_result = '/media/anlabadmin/data_ubuntu/yolox/Result_detection/model_x_10epochs_1800frames/'+camera+'/LabelPredict_'
    basename = os.path.basename(ckpt_file).split('.tar')[0]
    save_image_predict_folder = folder_result+basename
    
    
    os.mkdir(save_image_predict_folder)
    print(basename)


    image_folder = f"/media/anlabadmin/data_ubuntu/yolox/Frames_one_minute/{camera}"
    # image_folder = "/media/anlab/800gb/trungnh/save_frame/1400_frames/"+camera


    # Write to txt format
    results = []

    for filename in tqdm(natsorted(os.listdir(image_folder))):
        fullpath = os.path.join(image_folder,filename)
        tmp = predictBBox(fullpath,save_image_predict_folder)
        results+=tmp


    save_file_result_gen = '/media/anlabadmin/data_ubuntu/yolox/Result_detection/model_x_10epochs_1800frames/'+camera+'/ResultGen_'+basename+'.txt'
    with open(os.path.join(save_file_result_gen), 'w') as f:
        f.writelines(results)  

    listFilename = []
    for filename in natsorted(os.listdir(image_folder)):
        listFilename.append(filename[:-4]+'\n')

    with open(os.path.join('/media/anlabadmin/data_ubuntu/yolox/Result_detection/model_x_10epochs_1800frames/'+camera+'/listFilename_'+basename+'.txt'), 'w') as f:
        f.writelines(listFilename)  
    print(f"=====================Done {camera}==========================")
    