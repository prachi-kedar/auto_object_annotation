from cmath import cos
import cv2
import numpy as np
import time

from fontTools.misc.arrayTools import pointInRect
from openvino.inference_engine import IECore
from math import atan
import openvino_models as models
from datetime import datetime
from shapely.geometry import Polygon
from shapely.geometry import LineString
import tensorflow as tf
from tensorflow.keras.models import load_model
from torchvision import datasets, models, transforms
import sys
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
xmin_10, ymin_10, xmax_10, ymax_10, xmin_9, ymin_9, xmax_9, ymax_9=0,0,0,0,0,0,0,0
default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))

colors = (
        (255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85),
        (255, 0, 170), (85, 255, 0), (255, 170, 0), (0, 255, 0),
        (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
        (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255),
        (0, 170, 255))
def getrect(actual_line):
    
    dis=np.linalg.norm(np.array(actual_line[1])-np.array(actual_line[0]))
    print(dis)
    cd_length= int(dis)
    if cd_length > 200:
        cd_length=200
    ab = LineString([actual_line[1],actual_line[0]])
    left = ab.parallel_offset(cd_length / 2, 'left')
    right = ab.parallel_offset(cd_length / 2, 'right')
    c = left.boundary[1]
    d = right.boundary[0]  # note the different orientation for right offset
    cd = LineString([c, d])
    ef = LineString([(int(c.x),int(c.y)), (int(d.x),int(d.y))])
    left = ef.parallel_offset(cd_length , 'left')
    right = ef.parallel_offset(50 , 'right')
    e = left.boundary[1]
    f = right.boundary[0]
    pol = Polygon([(int(c.x),int(c.y)), (int(d.x),int(d.y)),(int(e.x),int(e.y))])
    xmin,ymin,xmax,ymax = pol.bounds
    return int(xmin),int(ymin),int(xmax),int(ymax)
    
def perpendicular(point1,point2):
    actual_line = (point1,point2)
    try:
        xmin,ymin,xmax,ymax = getrect(actual_line)
    except Exception as e:
        print(e)
        xmin,ymin,xmax,ymax=0,0,0,0
    return xmin, ymin, xmax, ymax

def rect(point,buffer=150):
    x=point[0]
    y=point[1]
    left, top, right, bottom = max(x - buffer, 0), max(y - buffer, 0), max(x + buffer, 0), max(y + buffer, 0)
    return left, top, right, bottom
def draw(img,poses, point_score_threshold,output_transform,count,collect,person_details,tracks_draw,skeleton=default_skeleton,draw_ellipses=False):
    #img = output_transform.resize(img)
    #if poses.size == 0:
    #    return img
    stick_width = 4

    img_limbs = np.copy(img)
    all_cod_10=[]
    all_cod_9=[]
    for pose in poses:
        xmin_10, ymin_10, xmax_10, ymax_10, xmin_9, ymin_9, xmax_9, ymax_9=0,0,0,0,0,0,0,0

        points = pose[:,:2].astype(np.int32)
        points = output_transform.scale(points)
        points_scores = pose[:,2]
        point_0 = points[5]
        cv2.circle(img,point_0,10,(0,0,0),-1)
        for id,bbox in tracks_draw.items():
            flag = pointInRect(point_0,bbox)
            if flag==1:
                xmin_10, ymin_10, xmax_10, ymax_10=perpendicular(points[10],points[8])
                xmin_9, ymin_9, xmax_9, ymax_9=perpendicular(points[9],points[7])
                #xmin_10, ymin_10, xmax_10, ymax_10=rect(points[10])
                #xmin_9, ymin_9, xmax_9, ymax_9=rect(points[9])
                person_details[id]=[[xmin_10, ymin_10, xmax_10, ymax_10,points[10]],[xmin_9, ymin_9, xmax_9, ymax_9,points[9]]]
        #for i,(p,v) in enumerate(zip(points, points_scores)):
        #    if v > point_score_threshold:
               
        #xmin_10, ymin_10, xmax_10, ymax_10=perpendicular(points[10],points[8])
        #xmin_9, ymin_9, xmax_9, ymax_9=perpendicular(points[9],points[7])

        if collect:
            img_right = img[ymin_10:ymax_10,xmin_10:xmax_10]
            img_left = img[ymin_9:ymax_9,xmin_9:xmax_9]
            cv2.imwrite("new_dataset/right/"+str(count)+"r"+".jpg", img_right)
            cv2.imwrite("new_dataset/left/"+str(count)+"l"+".jpg", img_left)
        #cv2.rectangle(img,(int(xmin_10),int(ymin_10)),(int(xmax_10),int(ymax_10)), (0, 255, 0), 4)
        #cv2.rectangle(img,(int(xmin_9),int(ymin_9)),(int(xmax_9),int(ymax_9)), (0, 255, 0), 4)
        
        all_cod_10.append([xmin_10, ymin_10, xmax_10, ymax_10])
        all_cod_9.append([xmin_9, ymin_9, xmax_9, ymax_9])
        for i, j in skeleton:
            if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                if draw_ellipses:
                    middle = (points[i] + points[j]) // 2
                    vec = points[i] - points[j]
                    length = np.sqrt((vec * vec).sum())
                    angle = int(np.arctan2(vec[1], vec[0]) * 180 / np.pi)
                    polygon = cv2.ellipse2Poly(tuple(middle), (int(length / 2), min(int(length / 50), stick_width)),
                                               angle, 0, 360, 1)
                    cv2.fillConvexPoly(img_limbs, polygon, colors[j])
                else:
                    #pass
                    cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=colors[j], thickness=stick_width)
    cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
    return img,person_details,count

def preprocess(frame,size):
    n,c,h,w = size
    input_image = cv2.resize(frame, (w,h))
    input_image = input_image.transpose((2,0,1))
    input_image.reshape((n,c,h,w))
    return input_image

def classifier(img,input_layer_c,output_layer_c,exec_net_c):
    infer_req_ = exec_net_c.start_async(request_id=0,inputs={input_layer_c:img})
    status=infer_req_.wait()
    flag = infer_req_.outputs[output_layer_c]
    return flag

def preprocess_tensorflow(img):
    img = img/255
    img = cv2.resize(img, (224, 224))
    img = img.reshape((1,224,224,3))
    return img

def classifier_tensorflow(img,model):
    flag = model.predict(img)
    return flag

def pickup(frame_dummy,rights,lefts,model_c):
    for right,left in zip(rights,lefts):
        img_right = frame_dummy[right[1]:right[3],right[0]:right[2]]
        img_left = frame_dummy[left[1]:left[3],left[0]:left[2]]  
        print(img_right.shape)

        #img_right = preprocess(img_right,size_c)
        #img_left = preprocess(img_left,size_c)
        img_right = preprocess_tensorflow(img_right)
        img_left = preprocess_tensorflow(img_left)
        flag_right = classifier_tensorflow(img_right,model_c)
        flag_left = classifier_tensorflow(img_left,model_c)

def preprocess_pytorch(img):
    to_classify = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    to_classify = cv2.resize(img, (224, 224))
    MEAN = 255 * np.array([0.485, 0.456, 0.406])
    STD = 255 * np.array([0.229, 0.224, 0.225])
    to_classify = ((to_classify - MEAN) / STD).astype("float32")

    transform = transforms.ToTensor()
    tensor = transform(to_classify)
    tensor = tensor.unsqueeze(0).to(device)
    return tensor

def classifier_pytorch(img,model_ft):
    print(img.shape)
    if img.shape[0] !=0 and img.shape[1] !=0:
        to_classify = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        to_classify = cv2.resize(to_classify, (224, 224))
        MEAN = 255 * np.array([0.485, 0.456, 0.406])
        STD = 255 * np.array([0.229, 0.224, 0.225])
        to_classify = ((to_classify - MEAN) / STD).astype("float32")

        transform = transforms.ToTensor()
        tensor = transform(to_classify)
        tensor = tensor.unsqueeze(0).cuda()

        outputs = model_ft(tensor)
        _, preds = torch.max(outputs, 1)
    else:
        preds = torch.tensor([6], dtype=torch.int64,device=device)
    return preds

def image_collect(img,poses, point_score_threshold,output_transform,count,collect,skeleton=default_skeleton,draw_ellipses=False):

    stick_width = 4

    img_limbs = np.copy(img)
    for pose in poses:
        xmin_10, ymin_10, xmax_10, ymax_10, xmin_9, ymin_9, xmax_9, ymax_9=0,0,0,0,0,0,0,0

        points = pose[:,:2].astype(np.int32)
        points = output_transform.scale(points)
        points_scores = pose[:,2]
       
        xmin_10, ymin_10, xmax_10, ymax_10=perpendicular(points[10],points[8])
        xmin_9, ymin_9, xmax_9, ymax_9=perpendicular(points[9],points[7])

        if collect:
            img_right = img[ymin_10:ymax_10,xmin_10:xmax_10]
            img_left = img[ymin_9:ymax_9,xmin_9:xmax_9]
            try:
                cv2.imwrite("new_dataset/right/"+str(count)+"r"+".jpg", img_right)
            except Exception as e:
                print(e)
            try:
                cv2.imwrite("new_dataset/left/"+str(count)+"l"+".jpg", img_left)
            except Exception as e:
                print(e)
        cv2.rectangle(img,(int(xmin_10),int(ymin_10)),(int(xmax_10),int(ymax_10)), (0, 255, 0), 4)
        cv2.rectangle(img,(int(xmin_9),int(ymin_9)),(int(xmax_9),int(ymax_9)), (0, 255, 0), 4)
        
        for i, j in skeleton:
            if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                if draw_ellipses:
                    middle = (points[i] + points[j]) // 2
                    vec = points[i] - points[j]
                    length = np.sqrt((vec * vec).sum())
                    angle = int(np.arctan2(vec[1], vec[0]) * 180 / np.pi)
                    polygon = cv2.ellipse2Poly(tuple(middle), (int(length / 2), min(int(length / 50), stick_width)),
                                               angle, 0, 360, 1)
                    cv2.fillConvexPoly(img_limbs, polygon, colors[j])
                else:
                    #pass
                    cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=colors[j], thickness=stick_width)
    cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
    return img,count
