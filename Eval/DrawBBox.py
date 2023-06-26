import os
import shutil
import cv2
from natsort import natsorted
import xml.etree.ElementTree as ET
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
# camera = "ch03"

def create_new_dir(new_folder):
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
        return False
    else:
        print("Folder exist. We do not need to create new folder")
        return True


def process_draw_predict(filename,predict_anotation_folder,image_folder,save_draw_folder,confi_thresh = 0.1):
    "Draw prediction box for image"
    image = cv2.imread(os.path.join(image_folder,filename))
    f = open(predict_anotation_folder+"/"+filename.split('.')[0]+".txt", "r")
    for line in f.readlines():
        rs = (line.strip().split(' ')[1:])
        yolo_bbox1 =(float(rs[0]),float(rs[1]),float(rs[2]),float(rs[3]),float(rs[4]))
        if yolo_bbox1[0] >= confi_thresh:
            x1, y1, x2, y2 = int(yolo_bbox1[1]),int(yolo_bbox1[2]),int(yolo_bbox1[3]),int(yolo_bbox1[4])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            image = cv2.putText(image, str(yolo_bbox1[0]), (int((yolo_bbox1[1]+yolo_bbox1[3])/2),int((yolo_bbox1[2]+yolo_bbox1[4])/2)), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 0, 0), 2, cv2.LINE_AA)
        t2 = time.time()
        cv2.imwrite(os.path.join(save_draw_folder,filename),image)


# Draw grounding true image
def process_draw_true(filename,image_folder,annotation_folder,save_draw_folder):
    image = cv2.imread(os.path.join(image_folder,filename))
    filexmlname = filename.split('.')[0]+".xml"
    
    tree = ET.parse(os.path.join(annotation_folder,filexmlname))
    root = tree.getroot()
    bndbox_values = []
    for bndbox in root.iter('bndbox'):
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        bndbox_values.append((xmin, ymin, xmax, ymax))
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 255), 3)
    cv2.imwrite(os.path.join(save_draw_folder,filename),image)



# def draw_predict_json(filename):
#     image = cv2.imread(os.path.join(image_folder,filename))
if __name__ == "__main__":
    list_camera = ["draw_yolo_x_m_weight_9"]
    
    
    confi_thresh = 0.1
    draw_grouding_true = True
    draw_predict = True
    
    # Traversal each camera in list camera and draw grounding true or predict box  
    for camera in tqdm(list_camera):
        annotation_folder = "/media/anlabadmin/data_ubuntu/yolox/Val_images/Image_soft_hard_loaichenguoi/Annotations" 
        image_folder = f"/media/anlabadmin/data_ubuntu/yolox/Val_images/Image_soft_hard_loaichenguoi/Images"
        predict_anotation_folder = f"/media/anlabadmin/data_ubuntu/yolox/Result_detection/soft_hard_yolox_m_weight_9_cof_05/LabelPredict_latest_ckpt.pth_yolo_m_9"
        
        # create folder for saving
        save_draw_folder = "/media/anlabadmin/data_ubuntu/yolox/Draw_images"+camera
        folder_exist = create_new_dir(save_draw_folder)

        if folder_exist == True:
            image_folder = save_draw_folder
        
        
        
        # # Draw real bounding box
        if draw_grouding_true:
            for filename in tqdm(natsorted(os.listdir(image_folder))):
                process_draw_true(filename,image_folder,annotation_folder,save_draw_folder)
            image_for_drawing_predict = save_draw_folder
        else:
            image_for_drawing_predict = image_folder

        
        # Draw prediction
        if draw_predict:
            with ThreadPoolExecutor() as executor:
                futures = []
                for filename in tqdm(natsorted(os.listdir(image_for_drawing_predict))):
                    future = executor.submit(process_draw_predict, filename,predict_anotation_folder,image_for_drawing_predict,save_draw_folder,confi_thresh)
                    futures.append(future)
                for future in futures:
                    future.result()
        
        