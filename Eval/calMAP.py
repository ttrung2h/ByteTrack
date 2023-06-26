import argparse
import os
from tqdm import tqdm
import os.path as osp
import time
import cv2
import torch
import xml.etree.ElementTree as ET
import pickle
import numpy as np
import sys
import shutil
def voc_ap(rec, prec, use_07_metric=False):
    """
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        # Append sentinel values at the end
        # prec = [0.0] + prec + [0.0]
        # rec = [0.0] + rec + [1.0]

        # # Compute the precision envelope
        # for i in range(len(prec) - 2, -1, -1):
        #     prec[i] = max(prec[i], prec[i+1])

        # # Find indices where recall changes value
        # indices = [i for i in range(1, len(rec)) if rec[i] != rec[i-1]]

        # # Compute AP as the sum of (delta recall) * precision
        # ap = sum((rec[indices[i]] - rec[indices[i-1]]) * prec[indices[i]] for i in range(1, len(indices)))

    return ap
def parse_rec(filename):
    """Parse a PASCAL VOC xml file"""
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        # obj_struct["pose"] = obj.find("pose").text
        # obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(float(bbox.find("xmin").text)),
            int(float(bbox.find("ymin").text)),
            int(float(bbox.find("xmax").text)),
            int(float(bbox.find("ymax").text)),
        ]
        objects.append(obj_struct)

    return objects
def voc_eval(
    detpath,
    annopath,
    imagesetfile,
    classname,
    cachedir,
    ovthresh=0.5,
    use_07_metric=False,
    limit = None
):
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, "annots.pkl")
    # read list of images
    with open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    # print(imagenames)
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print(f"Reading annotation for {i + 1}/{len(imagenames)}")
            if limit != None and i == limit:
                break
        # save
        print(f"Saving cached annotations to {cachefile}")
        with open(cachefile, "wb") as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, "rb") as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    classname = 'Human'
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([0 for x in R]).astype(bool)
        # difficult = np.array([x["difficult"] for x in R]).astype(bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}
    # read dets
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    if len(lines) == 0:
        return 0, 0, 0

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
    # print(BB)
    # exit()
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    sum_GT = 0
    

    # create list false positive images
    fp_list_images = []
    # caculate grounding true box
    # print(image_ids)
    # exit()
    for frame,info in class_recs.items():
        sum_GT +=info["bbox"].shape[0]
    
    
    # Traversal each box prediction 
    for d in range(nd):
        R = class_recs[image_ids[d]]
        # print(R)
        # exit()
        bb = BB[d, :].astype(float)
        # print(bb)
        # exit()
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)
        


        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            # print(BBGT[:, 0],bb[0])
            # exit()
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0) - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
    # print(ovmax)
    # exit()
        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0
            fp_list_images.append(image_ids[d])
        

    fp_list_images = set(fp_list_images)
    # caculate fp,tp,fn,recall,percision
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth

    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)



    # Caculate number fn,tp,fp
    FN = int(np.max(npos) - np.max(tp))
    TP = int(np.max(tp))
    FP = int(np.max(fp))
    
    
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap, tp, fp,TP,FP,FN,sum_GT,fp_list_images




    


def move_fp_images(list_fp_images,base_name,image_draw_fodel):
    new_folder = f"/media/anlab/800gb/trungnh/Fp_softhard_images/fp_images_{base_name}"
    os.mkdir(new_folder)

    for img in list_fp_images:
        img = img+".jpg"
        old_name = os.path.join(image_draw_fodel,img)
        new_name = os.path.join(new_folder,img)
        shutil.copy(old_name,new_name)



# camera_list = ["ch06","ch07","ch08"]
camera_list = ["soft_hard_yolox_s_weight_9_cof_05"]
# image_draw_folder = "/media/anlab/800gb/trungnh/Draw/soft_hard_data_69_05"


ckpt_file = "/media/anlabadmin/data_ubuntu/yolox/Weight/Weight_yolox_s/weight/latest_ckpt.pth_yolo_s_9.tar"
basename = os.path.basename(ckpt_file).split('.tar')[0] 

for camera in camera_list:
    print(f"====================={camera}==========================")
    detpath = '/media/anlabadmin/data_ubuntu/yolox/Result_detection/'+camera+'/ResultGen_'+basename+'.txt'
    annopath = '/media/anlabadmin/data_ubuntu/yolox/Val_images/Image_soft_hard_loaichenguoi/Annotations/{:s}.xml'
    imagesetfile = '/media/anlabadmin/data_ubuntu/yolox/Result_detection/'+camera+'/listFilename_'+basename+'.txt'
    classname = 'Human'
    cachedir = '/media/anlabadmin/data_ubuntu/yolox/Result_detection/'+camera+'/tmp'

    # ovthresh = 0.1
    use_07_metric = True
    result_file = '/media/anlabadmin/data_ubuntu/yolox/Result_detection/'+f'{camera}/result_map_{basename}.txt' 
    average = 0 
    with open(detpath,'r') as f:
        num_box_predict = len(f.readlines())
    with open(result_file,"a") as f:
        for ovthresh in [0.5]:
        # Run evaluation
            rec, prec, ap, tp, fp,TP,FP,FN,sum_GT,fp_list_images = voc_eval(
                detpath,
                annopath,
                imagesetfile,
                classname,
                cachedir,
                ovthresh,
                use_07_metric
            )

        
            out_statemant = f"AP for ovthresh:{ovthresh} = {ap*100:.4f}"
            # average = average+ap*100
            f.write(out_statemant+"\n")
            print(out_statemant)
            print(f"TP : {TP},FP : {FP}, FN : {FN}")
            
            
            # num boxes
            print(f"Number of PREDICT boxes : {num_box_predict}")
            print(f"Number of GT boxes : {sum_GT}")
            print(f"TP : {TP},FP : {FP}, FN : {FN},Number of PREDICT boxes : {num_box_predict},Number of GT boxes : {sum_GT}")
            print(f"TP_rate : {TP/num_box_predict},FP_rate : {FP/num_box_predict}, FN_rate : {FN/sum_GT}")


            # write tp,fp,fn
            f.write(f"TP : {TP},FP : {FP}, FN : {FN},Number of PREDICT boxes : {num_box_predict},Number of GT boxes : {sum_GT}\n")
            f.write(f"TP_rate : {TP/num_box_predict},FP_rate : {FP/num_box_predict}, FN_rate : {FN/sum_GT}")
            
        # f.write(f"Average AP: {average/9}")
    print(f"=====================Done {camera}==========================")


    # move fp images
    # move_fp_images(fp_list_images,base_name=basename,image_draw_fodel=image_draw_folder)
    



