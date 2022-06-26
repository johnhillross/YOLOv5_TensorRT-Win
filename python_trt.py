from ctypes import *
import cv2
import numpy as np
import numpy.ctypeslib as npct
import random

class Detector():
    def __init__(self,model_path,dll_path):
        self.yolov5 = CDLL(dll_path)
        self.yolov5.Detect.argtypes = [c_void_p,c_int,c_int,POINTER(c_ubyte),npct.ndpointer(dtype = np.float32, ndim = 2, shape = (50, 6), flags="C_CONTIGUOUS")]
        self.yolov5.Init.restype = c_void_p
        self.yolov5.Init.argtypes = [c_void_p]
        self.yolov5.cuda_free.argtypes = [c_void_p]
        self.c_point = self.yolov5.Init(model_path)

    def predict(self,img):
        rows, cols = img.shape[0], img.shape[1]
        res_arr = np.zeros((50,6),dtype=np.float32)
        self.yolov5.Detect(self.c_point,c_int(rows), c_int(cols), img.ctypes.data_as(POINTER(c_ubyte)),res_arr)
        self.bbox_array = res_arr[~(res_arr==0).all(1)]
        return self.bbox_array

    def free(self):
        self.yolov5.cuda_free(self.c_point)

def getColor(classes):
    color=[]
    for i in range(len(classes)):
        color.append([random.randint(0, 255) for _ in range(3)])
    
    return color

def visualize(img,bbox_array,classes,color):
    for temp in bbox_array:
        xywh = [temp[0],temp[1],temp[2],temp[3]]    # xywh
        clas = classes[int(temp[4])]
        score = temp[5]
        cv2.rectangle(img,(int(xywh[0]),int(xywh[1])),(int(xywh[0]+xywh[2]),int(xywh[1]+xywh[3])), color[int(temp[4])], 2)
        img = cv2.putText(img, clas+" "+str(round(score,2)), (int(xywh[0]),int(xywh[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[int(temp[4])], 2)
   
    return img