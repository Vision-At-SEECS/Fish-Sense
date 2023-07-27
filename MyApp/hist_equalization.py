from matplotlib import pyplot as plt
from scipy.misc import ascent
import cv2
import os 
#from imutils.perspective import four_point_transform
#from imutils import contours
#from imutils import perspective
import imutils
import sys
import torch, torchvision
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from PIL import Image
import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from PIL import ImageColor
from MyApp import fish_parts
import math
from MyApp import specie
from MyApp import health_detection

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
#cfg.DATASETS.TRAIN = ("straindata",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 1000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
cfg.MODEL.WEIGHTS = "trained_models\\fish_detectionv4.pth"  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9   # set a custom testing threshold
cfg.MODEL.DEVICE = 'cpu'
# cfg.DATASETS.TEST=("straindata49",)
predictor = DefaultPredictor(cfg)
# cfg

from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances
register_coco_instances("fish",{},"\\detectFish_v3\\trainval.json","\\detectFish_v3\\images")
sample_metaadata=MetadataCatalog.get("fish")   


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

#Fish detection using detectron2
def predict_frame(frame,one_unit_cm, fish_status):
    ftype_arr=[]
    orgImg=frame.copy()
    my_dic={}
    failed=0
    body_bleed=0
    frame_height, frame_width, _ = frame.shape
    total_weight=0
    sum_of_length=0
    sum_of_height=0
    size = (frame_width, frame_height)
    frame2 = frame.copy()
    outputs = predictor(frame2)
    v = Visualizer(frame2[:, :, ::-1],
		metadata=sample_metaadata, 
		scale=0.8, 
		instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
		)
    frame_process=cv2.imwrite("output\\Fish_actual.png",orgImg)    
    #TODO only draw ploygon and rec
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    myresult=out.get_image()[:, :, ::-1]
    myresult=cv2.resize(myresult,size)
    outputnump=outputs["instances"].pred_boxes.tensor.numpy()
    #Bounding boxes array 
    coord_array = []
    for j in range(len(outputnump)):
        x_text=((outputnump[j][0]+outputnump[j][2])/2)
        x_text=round(x_text)
        y_text=(outputnump[j][1]+outputnump[j][3])/2
        y_text=round(y_text)
        coord_array.append([x_text, y_text])
    print("coord_array",coord_array)    
    mask = outputs['instances'].pred_masks.to('cpu').numpy()
    mask = mask.astype(np.uint8)
    mask[mask > 0] = 255
    (N, H, W) = mask.shape
    increase_count=0
    for i in range(N):
        success_fail=0
        print("FISH ID:", i)
        mask1 = np.zeros((H, W), dtype=np.uint8)
        mask1[mask[i, :] > 0] = 255
        cnts = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        min_rect = cv2.minAreaRect(c)  # min_area_rectangle
        min_rect = np.int0(cv2.boxPoints(min_rect))
        mybox = cv2.minAreaRect(c)
        rect = cv2.minAreaRect(c) # Get the minimum bounding rectangle (center (x, y), (width, height), rotation angle)
        center_of_min_rect=rect[0]
        width_of_min_rect=round(rect[1][1])
        height_of_min_rect=round(rect[1][0])
        angle_of_fish_rect=round(rect[2])
        #AREA of FISH detected 
        fish_mask = cv2.bitwise_and(frame2, frame2, mask=mask1)
        #Checking the rotation of object 
        if (width_of_min_rect>height_of_min_rect):
            length=width_of_min_rect
            height=height_of_min_rect
            after_rotation=mask1
            actual_angle_of_fish=0
        else:
            actual_angle_of_fish=angle_of_fish_rect
            after_rotation = imutils.rotate(mask1, angle=angle_of_fish_rect)
            length=height_of_min_rect
            height=width_of_min_rect
        cv2.imwrite("output\\Fish_rotated.png",after_rotation)
        read_img = after_rotation
        #read_img = cv2.cvtColor(read_img, cv2.COLOR_BGR2GRAY)
        cnts_after_rotation, hierarchy = cv2.findContours(read_img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        read_img=cv2.drawContours(read_img, cnts_after_rotation, -1, (0, 0, 0), 3)
        #Here we are finding the height of fish by using extreme points
        for c in cnts_after_rotation:
            rot_x,rot_y,rot_w,rot_h = cv2.boundingRect(c)
            x1=rot_x
            y1=rot_y
            x2=rot_x+rot_w
            y2=rot_y+rot_h
            width_value=x2-x1
            x_percent0=int(0.34*width_value)
            x_percent1=int(0.40*width_value)
            x_percent2=int(0.60*width_value)
            x_percent3=int(0.70*width_value)
            roi_crop=read_img[y1:y2,x1+x_percent0:x1+x_percent1]
            roi_crop2=read_img[y1:y2,x1+x_percent2:x1+x_percent3]
            #roi_crop = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB)
            im2 = Image.fromarray(roi_crop)
            #roi_crop2 = cv2.cvtColor(roi_crop2, cv2.COLOR_BGR2RGB)
            im3 = Image.fromarray(roi_crop2)

            v=after_rotation.shape
            my_img_1 = np.zeros((v[0], v[1], 1), dtype = "uint8")
            #Copy and paste cropped 38% fish on black image
            my_img_1 = cv2.cvtColor(my_img_1, cv2.COLOR_BGR2RGB)
            im1 = Image.fromarray(my_img_1)
            back_im = im1.copy()
            back_im2 = im1.copy()
            back_im.paste(im2, (x1+x_percent0, y1))
            back_im2.paste(im3, (x1+x_percent2, y1))
            
            pillowImage = np.array(back_im)
            pillowImage = cv2.cvtColor(pillowImage, cv2.COLOR_RGB2BGR)
            
            pillowImage2 = np.array(back_im2)
            pillowImage2 = cv2.cvtColor(pillowImage2, cv2.COLOR_RGB2BGR)

            pil_gray = cv2.cvtColor(pillowImage, cv2.COLOR_BGR2GRAY)
            pil_gray2 = cv2.cvtColor(pillowImage2, cv2.COLOR_BGR2GRAY)
            # Find Canny edges
            #edged = cv2.Canny(pil_gray, 30, 200)
            cnts = cv2.findContours(pil_gray, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            if (len(cnts)>0):
                c = max(cnts, key=cv2.contourArea)
                extTop = tuple(c[c[:, :, 1].argmin()][0])
                extBot = tuple(c[c[:, :, 1].argmax()][0])
                #cv2.rectangle(pillowImage, (x, y), (x+w, y+h), (0, 0, 255), 1)
                orange_line_length1=math.dist(extTop,extBot)

                cnts = cv2.findContours(pil_gray2, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                c = max(cnts, key=cv2.contourArea)
                extTop = tuple(c[c[:, :, 1].argmin()][0])
                extBot = tuple(c[c[:, :, 1].argmax()][0])
                #cv2.rectangle(pillowImage, (x, y), (x+w, y+h), (0, 0, 255), 1)
                orange_line_length2=math.dist(extTop,extBot)
                if orange_line_length2>orange_line_length1:
                    fish_width_38=orange_line_length2
                    fish_width_64=orange_line_length1
                else:
                    fish_width_38=orange_line_length1
                    fish_width_64=orange_line_length2
                masked = cv2.bitwise_and(frame2, frame2, mask=mask1)
            else:
                #print("Failed here ")
                failed=1
                fish_width_38%=height
                #print("Checking height", fish_width_updated4)
        new_mask=masked
        cv2.imwrite("output\\Maskeddd_image.png", masked)
                  
        if failed==0:
            fish_id = i+1
            print("fish_id",fish_id)
        
        frame_process=cv2.imread("output\\Fish_actual.png")      
        #Body part detection
        front_height,parts_detected,tail_bld,array_bleed_fin= fish_parts.fish_part(masked, length, frame_process,fish_mask,fish_status)    
        fish_parts_detected_img=parts_detected
        #print("Is tail bleeding?",tail_bld)
        cv2.imwrite("output\\Fish_actual.png",fish_parts_detected_img )

        # Detect red spots on the body other than body parts 
        contour_found=0
        if fish_status=="unhealthy":
            #print("Classification results", fish_status)
            red_spots=cv2.imread("output\\tail_mask_updated.png")
            hsv = cv2.cvtColor(red_spots, cv2.COLOR_BGR2HSV)
            lower_red = np.array([0,50,50])
            upper_red = np.array([10,255,255])
            mask0 = cv2.inRange(hsv, lower_red, upper_red)
            # upper mask (170-180)
            lower_red = np.array([170,50,50])
            upper_red = np.array([180,255,255])
            mask1 = cv2.inRange(hsv, lower_red, upper_red)
            # join my masks
            mask_final = mask0+mask1
            # Find contours from the mask
            contours, hierarchy = cv2.findContours(mask_final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
           
            for contour in contours:
                #print("cv2.contourArea(contour)",cv2.contourArea(contour))
                if cv2.contourArea(contour)>=500:
                    contour_found=1
                    fish_parts_detected_img = cv2.drawContours(fish_parts_detected_img, contour, -1, (0, 0, 255), 2)  
                    
                    #output = cv2.putText(output, 'Unhealthy fish', (300,300), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,0), 2, cv2.LINE_AA)
                    #resized_fish_parts = cv2.resize(fish_parts_detected_img, (300, 900))
                    #cv2.imshow("fish_parts_detected_img",resized_fish_parts)
                    #cv2.waitKey(0)
            if (contour_found==1):
                body_bleed="Yes"
            else:
                body_bleed="No"        

        length_of_fish_cm=length*one_unit_cm
        height_of_fish_cm=height*one_unit_cm
        fish_width_updated_38=fish_width_38*one_unit_cm #Height 38%
        fish_width_updated_64=fish_width_64*one_unit_cm
        
        
        #Fish specie classifier 
        img_path="source_img.png"
        fish_type= specie.finder(img_path)
        print("Checking fish type here", fish_type)
        ftype_arr.append(fish_type)  
        increase_count=increase_count+1
        Weight1=0
        

        if (fish_type=='char'):
            fish_width_updated_cm4=fish_width_updated_38
            Weight1=((411.48*(math.log(length_of_fish_cm*length_of_fish_cm*fish_width_updated_38)))-3292.4)
        elif(fish_type=='tilapia'):
            fish_width_updated_cm4=fish_width_updated_38
            Weight1=(0.0543*(length_of_fish_cm*length_of_fish_cm*fish_width_updated_38))+24.616
        elif(fish_type=='pikeperch'):
            fish_width_updated_cm4=fish_width_updated_64
            Weight1=(0.0501*(length_of_fish_cm*length_of_fish_cm*fish_width_updated_64))-130.63
        elif(fish_type=='trout'):
            fish_width_updated_cm4=fish_width_updated_38
            if (length_of_fish_cm>30):
                #Trout is salmon trout
                Weight1=(1630.5*(math.log(length_of_fish_cm*length_of_fish_cm*fish_width_updated_38)))-15076 
            else:
                x_variable=(length_of_fish_cm*length_of_fish_cm*fish_width_updated_38)
                #print("x_variable",x_variable)
                power_val=pow(x_variable, 1.1856)
                #print("Power",power_val)
                ans=0.0123*power_val
                #print("ans",ans)
                Weight1=(ans)                
        elif(fish_type=='perch'):
            fish_width_updated_cm4=fish_width_updated_64
            Weight1=(0.0675*(length_of_fish_cm*length_of_fish_cm*fish_width_updated_64))-4.4942       
        Weight1=Weight1+38
        total_weight=total_weight+Weight1
        sum_of_length=sum_of_length+length_of_fish_cm
        sum_of_height=sum_of_height+fish_width_updated_cm4
        
    print("i hereeeeeeeeeeeeeeeee" , i )            
    my_dic[i+1] = (round(length_of_fish_cm, 1),round(Weight1,1), round(fish_width_updated_cm4,1), fish_type,array_bleed_fin,tail_bld,body_bleed )
    print("my_dic", my_dic)
    #sum_of_length=(sum_of_length/increase_count)
    sum_of_length=round(sum_of_length,1)
    #sum_of_height=(sum_of_height/increase_count)
    #sum_of_height=round(sum_of_height,1)
    total_weight = round(total_weight, 1)
    #average_biomass=total_weight//increase_count
    #print("Hello heelo",array_bleed_fin)
    results = {
        "fish_details" : my_dic,
        "total_weight" : total_weight,
        "total_count"  : increase_count,
        "fish_status":fish_status,
    }
    return fish_parts_detected_img, results
        

def result(data):
    source_path = data['source_img']
    matched_path = "static\\matched.png"
    
    image_src = cv2.imread(cv2.samples.findFile(source_path))    
    filename, file_extension = os.path.splitext(source_path)
    localfilename = filename + '_processed' + file_extension
    
    image_width = data['image_width']
    image_height = data['image_height']
    rois = data['rois']
    image_path = source_path
    filename, file_extension = os.path.splitext(source_path)
    localfilename = filename + '_processed' + file_extension
    
    img = cv2.imread(source_path)
    img = cv2.resize(img, (image_width, image_height))
    orig_img = img.copy()    

    if source_path is None:
      return "Error : Image Path '" + image_path + "' not found"
    
    fish_status= health_detection.finder(source_path)

    #Detect yellow ruler
    img=image_src  
    kernel = np.ones((5,5),np.float32)/25
    img = cv2.filter2D(img,-1,kernel)
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hsv = cv2.medianBlur(hsv, 5)
    lower_range=np.array([22,93,0])
    upper_range=np.array([33,255,255])
    mask=cv2.inRange(hsv,lower_range,upper_range)
    after_mask=cv2.bitwise_and(img,img,mask=mask)
    #cv2.imwrite("Ruler.png",after_mask)
    img_gray = cv2.cvtColor(after_mask, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 40, 255, cv2.THRESH_BINARY)
    contours, hierarchy= cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
    largest_item= sorted_contours[0]
    c = largest_item
    min_rect_s = cv2.minAreaRect(c)  # min_area_rectangle
    min_rect_s = np.int0(cv2.boxPoints(min_rect_s))
    #cv2.drawContours(image_src, [min_rect_s], -1, (0, 255, 0), 3)
    rect_s = cv2.minAreaRect(c)
    width_of_ruler=round(rect_s[1][0])
    height_of_ruler=round(rect_s[1][1])
    if width_of_ruler>height_of_ruler:
        side_of_tray=width_of_ruler
    else:
        side_of_tray=height_of_ruler
    lenth_of_tray_cm=16.6
    length_of_tray_pixels=side_of_tray
    one_unit_cm=lenth_of_tray_cm/length_of_tray_pixels
    frame, results= predict_frame(image_src,one_unit_cm, fish_status)
    cv2.waitKey(0)

    cv2.imwrite(matched_path,frame)  
    return results
