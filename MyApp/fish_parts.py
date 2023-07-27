# import libraries
import math
from PIL import Image, ImageDraw, ImageFilter
import torch, torchvision
import os, json, cv2, random
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from imutils.perspective import four_point_transform
from imutils import contours
from imutils import perspective
import imutils
from collections import namedtuple
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025 
cfg.SOLVER.MAX_ITER = 500    
cfg.SOLVER.STEPS = []        
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5 
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
cfg.MODEL.WEIGHTS = "trained_models\\fish_parts4.pth"  
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.DEVICE = 'cpu'
predictor = DefaultPredictor(cfg)
register_coco_instances("fish_parts4",{},".\\detection_labels\\trainval.json",".\\\\updated_biomass_v3\\images")
sample_metaadata=MetadataCatalog.get("fish_parts4")
MetadataCatalog.get("fish_parts4").set(thing_classes=["Fish", "Tail", "Head", "Eyes", "Fin"])
sample_metaadata=MetadataCatalog.get("fish_parts4") 

# Method to find the mid point of a line detection_labels
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

center_line_coord=[]
def fish_part(myimg,total_len_fish,orgImg, fish_mask,fish_status):  #myimg image has object detected and rest is blacked out    
    dic_status=[]
    #Fin_average=[] #fin_extLeft, fin_extRight, b_middle_fin
    frame2=myimg
    org_img=orgImg
    fish_width_fin=0
    Fin_average=[] 
    fish_width=0  #front height
    Image_height, Image_width, _ = org_img.shape
    size = (Image_width//3, Image_height//3)
    orig_size = (Image_width, Image_height)
    org_img=cv2.resize(org_img,size)
    #Status of body parts 
    head_arr=[ ] #LXW
    head_length=0
    head_width=0
    head_status=0
    tail_status=0
    tail_bleed=0
    tail_bleed_ans=0
    eye_status=0
    fin_count=0
    fin_bleed=0
    fin_bleed_array=[]

    outputs = predictor(frame2)
    v = Visualizer(orgImg[:, :, ::-1],
        metadata=sample_metaadata, 
        scale=0.8, 
        #instance_mode=ColorMode.IMAGE_BW   
        )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    myresult=out.get_image()[:, :, ::-1]
    myresult=cv2.resize(myresult,size)
    outputnump=outputs["instances"].pred_boxes.tensor.numpy()
    output_length = len(outputnump)
    #print("No: of fish parts detected are:"+str(output_length))
    check_detection=(output_length)
    class_name=outputs["instances"].pred_classes
    coord_array = []
    for j in range(len(outputnump)):
        x_text=((outputnump[j][0]+outputnump[j][2])/2)
        x_text=round(x_text)
        y_text=(outputnump[j][1]+outputnump[j][3])/2
        y_text=round(y_text)
        coord_array.append([x_text, y_text])
    mask = outputs['instances'].pred_masks.to('cpu').numpy()
    mask = mask.astype(np.uint8)
    mask[mask > 0] = 255
    (N, H, W) = mask.shape
    fish_count=0
    for i in range(N):


        """
        if (class_name[i]==0):
            fish_count=fish_count+1
            list_obj = outputnump[i].tolist()
            org=(int(list_obj[0]),int(list_obj[1]))
            myresult = cv2.putText(myresult.astype(np.uint8).copy(), str(fish_count), org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        """    
        mask1 = np.zeros((H, W), dtype=np.uint8)
        mask1[mask[i, :] > 0] = 255
        #cv2.imwrite("img.png",img)
        #cv2.waitKey(0)
        fin_img = cv2.bitwise_and(frame2, frame2, mask=mask1)
        #cv2.imwrite('fin_output{0}.png'.format(i+1),fin_img)

        cnts = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        min_rect = cv2.minAreaRect(c)  # min_area_rectangle
        min_rect = np.int0(cv2.boxPoints(min_rect))
        mybox = cv2.minAreaRect(c)
        rect = cv2.minAreaRect(c)
        width_of_min_rect=round(rect[1][0])
        height_of_min_rect=round(rect[1][1])
        mybox = cv2.cv.BoxPoints(mybox) if imutils.is_cv2() else cv2.boxPoints(mybox)
        mybox = np.array(mybox, dtype="int")
        mybox = perspective.order_points(mybox)
        
        """
        (tl, tr, br, bl) = mybox
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        myresult=cv2.resize(myresult,orig_size)
        if (class_name[i]==0):
            dist1=math.dist( (int(tl[0]),(int(tl[1]))),(int(tr[0]),(int(tr[1]))) )
            dist2=math.dist( (int(tl[0]),(int(tl[1]))),(int(bl[0]),(int(bl[1]))) )
            if (dist1>dist2):
                p=(int(tltrX), int(tltrY))
                q=(int(blbrX), int(blbrY))
                fish_width_fin=math.dist(p, q)
                fish_width=fish_width_fin
                A_point=(int(tltrX), int(tltrY))
                B_point=(int(blbrX), int(blbrY))
                center_line_coord.append(A_point)
                center_line_coord.append(B_point) 
            elif (dist2>dist1):
                p=(int(tlblX), int(tlblY))
                q=(int(trbrX), int(trbrY))
                fish_width_fin=math.dist(p, q)
                fish_width=fish_width_fin
                A_point=(int(tlblX), int(tlblY))
                B_point=(int(trbrX), int(trbrY))
                center_line_coord.append(A_point)
                center_line_coord.append(B_point)
        """
        if width_of_min_rect>height_of_min_rect:
            #print("rect",rect)
            length=width_of_min_rect
            height=height_of_min_rect
        else:
            length=height_of_min_rect
            height=width_of_min_rect

        if(class_name[i]==1):
            final_mask = 255 - mask1
            tail_mask = cv2.bitwise_not(final_mask)
            tcontours, thierarchy = cv2.findContours(tail_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in tcontours:
                tail_mask_updated=cv2.fillPoly(fish_mask, pts=[c], color=(0,0,0))
                cv2.imwrite("output\\tail_mask_updated.png",tail_mask_updated)
            tail_status=1
            masked = cv2.bitwise_and(frame2, frame2, mask=mask1)
            if fish_status=="unhealthy":
                hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
                lower_red = np.array([0,100,10])
                upper_red = np.array([9,230,230])
                mask0 = cv2.inRange(hsv, lower_red, upper_red)
                # upper mask (170-180)
                lower_red = np.array([169,100,10])
                upper_red = np.array([180,230,230])
                mask1 = cv2.inRange(hsv, lower_red, upper_red)
                mask_final = mask0+mask1
                contours, hierarchy = cv2.findContours(mask_final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour)>=100:
                        myresult = cv2.drawContours(myresult, contours, -1, (0, 0, 255), 3)
                        tail_bleed=1
               
            if (length<=height):
                tail_len=length
            else:
                tail_len=height 


        if (class_name[i]==2):
            head_status=1
            if width_of_min_rect>height_of_min_rect:
                head_length=width_of_min_rect
                head_width=height_of_min_rect
            else:
                head_length=height_of_min_rect
                head_width=width_of_min_rect 

        if(class_name[i]==3):
            eye_status=1    
        
        if (class_name[i]==4):
            fin_count=fin_count+1
            print(fin_count,"fin_count")
            cnts = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            Fin_average.append([extLeft,extRight])
            contour_found=0
            #list_obj = outputnump[i].tolist()
            #org=(int(list_obj[0]+35),int(list_obj[1]+15))
            #myresult = cv2.putText(myresult.astype(np.uint8).copy(), str(i+1), org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            mask1 = np.zeros((H, W), dtype=np.uint8)
            mask1[mask[i, :] > 0] = 255
            fin_img = cv2.bitwise_and(frame2, frame2, mask=mask1)
            #cv2.imwrite('fin{0}.png'.format(i+1),fin_img)
            masked = cv2.bitwise_and(frame2, frame2, mask=mask1)
            final_fin = 255 - mask1
            fish_fin_mask = cv2.bitwise_not(final_fin)
            fincontours, thierarchy = cv2.findContours(fish_fin_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if(class_name[i]==1):
                final_mask = 255 - mask1
                tail_mask = cv2.bitwise_not(final_mask)
                tcontours, thierarchy = cv2.findContours(tail_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for c in tcontours:
                    tail_mask_updated=cv2.fillPoly(fish_mask, pts=[c], color=(0,0,0))
                    cv2.imwrite("tail_mask_updated.png",tail_mask_updated)

                tail_status=1
            
            masked = cv2.bitwise_and(frame2, frame2, mask=mask1)
            if fish_status=="unhealthy":
                hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
                lower_red = np.array([0,100,10])
                upper_red = np.array([9,230,230])
                mask0 = cv2.inRange(hsv, lower_red, upper_red)
                # upper mask (170-180)
                lower_red = np.array([169,100,10])
                upper_red = np.array([180,230,230])
                mask1 = cv2.inRange(hsv, lower_red, upper_red)
                mask_final = mask0+mask1
                contours, hierarchy = cv2.findContours(mask_final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour)>=100:
                        myresult = cv2.drawContours(myresult, contours, -1, (0, 0, 255), 3)
                        tail_bleed=1
               
            if (length<=height):
                tail_len=length
            else:
                tail_len=height 
            
            if fish_status=="unhealthy":
                # Detect red spots on fish
                hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
                # lower mask (0-10)
                lower_red = np.array([0,100,10])
                upper_red = np.array([9,230,230])
                mask0 = cv2.inRange(hsv, lower_red, upper_red)
                # upper mask (170-180)
                lower_red = np.array([169,100,10])
                upper_red = np.array([180,230,230])
                mask1 = cv2.inRange(hsv, lower_red, upper_red)
                # join my masks
                mask_final = mask0+mask1
                contours, hierarchy = cv2.findContours(mask_final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contourlength=len(contours)
                if contourlength>10:
                    myresult = cv2.drawContours(myresult, contours, -1, (0, 0, 255), 3)
                    contour_found=1
                print("contourlength",contourlength)
            if (contour_found==1):
                fin_bleed="Yes"
            else:
                fin_bleed="No"        
            
            fin_bleed_array.append(fin_bleed)
            print("Here is the fin_bleed status ", fin_bleed_array)
    
    parts_detected=myresult
    dic_status.append(tail_status)
    dic_status.append(head_status)
    dic_status.append(eye_status)
    head_arr.append(head_length)
    head_arr.append(head_width)
    if tail_bleed==0:
        tail_bleed_ans="No"
    elif tail_bleed==1:
        tail_bleed_ans="Yes" 

    return fish_width,parts_detected,tail_bleed_ans,fin_bleed_array