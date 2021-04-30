#-*- coding: utf-8 -*-
#obstacle detection OK
import cv2
import sys
import gc
import time
import sys
import numpy as np
import sys,os,time
import Jetson.GPIO as GPIO
from skimage import data, exposure, img_as_float
# GPIO  set
GPIO.setmode(GPIO.BOARD)
#check mode
mode = GPIO.getmode()
#no warning
GPIO.setwarnings(False)

# set initial value
straight=37
left=33
right=29
detect=40
reset=31

#setup
GPIO.setup(detect, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(straight, GPIO.OUT,initial=GPIO.LOW)
GPIO.setup(left, GPIO.OUT,initial=GPIO.LOW)
GPIO.setup(right, GPIO.OUT,initial=GPIO.LOW)
GPIO.setup(reset, GPIO.OUT,initial=GPIO.LOW)
#set default as low voltage
# define Python user-defined exceptions
class Error(Exception):
   """Base class for other exceptions"""
   pass

class OBSTACLE_DETECTION(Error):
   """Raised when the input value is too small"""
   pass

#class ValueTooLargeError(Error):
#   """Raised when the input value is too large"""
#   pass

# our main program
# user guesses a number until he/she gets it right

# you need to guess this number

if __name__ == '__main__':
    while True:
            #顏色設定      
        color = (0, 0, 0)
    
            #攝影機設定
        cap = cv2.VideoCapture(r"/dev/video2")
        a=0
        _, frame = cap.read()   #读取一帧视频
        try:
            if GPIO.input(40)==1:
                raise OBSTACLE_DETECTION
            
            GPIO.output(straight, GPIO.HIGH)
            GPIO.output(left, GPIO.HIGH)
            GPIO.output(right, GPIO.HIGH)
            print('go.....')
           
            #人臉偵測的路徑
            cascade_path = r'haarcascade_frontalface_alt2.xml' 
            image= exposure.adjust_gamma(frame, 0.5)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

                #使用人脸识别分类器，读入分类器
            cascade = cv2.CascadeClassifier(cascade_path)
                                                              
                #利用分类器识别出哪个区域为人脸
            faceRects = cascade.detectMultiScale(image, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))        
            if len(faceRects) > 0:                 
                for faceRect in faceRects: 
                    x, y, w, h = faceRect
                    print('x,y:',x+int((w)/2), y+int((h)/2)) #x y 的中心點
                        #擷取臉部原始圖像大小
                    image = frame[y : y + h , x : x + w ]
                    if h>40 and w>40:
                        print('h',h)
                        cv2.rectangle(frame, (x , y ), (x + w , y + h ), color, thickness = 3)
                        print('image size:',np.array(image).shape)
                        center_y=y+int((h)/2)
                        center_x=x+int((w)/2)
                        if(400>center_x>280):
                            print("go straight")
                            GPIO.output(right, GPIO.HIGH)
                            GPIO.output(straight, GPIO.HIGH)
                            GPIO.output(left, GPIO.HIGH)
                            time.sleep(0.3)
                        elif(center_x>400):
                            print("turn right")
                            GPIO.output(right, GPIO.LOW)
                            GPIO.output(straight, GPIO.HIGH)
                            GPIO.output(left, GPIO.HIGH)
                            time.sleep(0.05)
                        else:
                            print("turn left")
                            GPIO.output(left, GPIO.LOW)
                            GPIO.output(straight, GPIO.HIGH)
                            GPIO.output(right, GPIO.HIGH)
                            time.sleep(0.05)
                cv2.putText(frame,'.', 
                                (x+int((w)/2), y+int((h)/2)),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                2,                                     #字号
                                (0,0,255),                           #颜色
                                5)
            cv2.imshow("Recognise myself", frame)
        
            #等待10毫秒看是否有按键输入
            k = cv2.waitKey(10)
            #如果输入q则退出循环
            if k & 0xFF == ord('q'):
                break
        
   
    #释放摄像头并销毁所有窗口
        
        except OBSTACLE_DETECTION:
            print('here come obstacle')
            GPIO.output(reset,GPIO.HIGH) #reset
            GPIO.output(reset,GPIO.LOW)  
            print('reset finish..')
            GPIO.output(left, GPIO.LOW)
            GPIO.output(right, GPIO.LOW)
            print('go back ready')
            time.sleep(0.5)
            GPIO.output(right, GPIO.HIGH)#往右
            print('turn_right ready')
            time.sleep(0.5)  
            print('no obstacle and go')
        except KeyboardInterrupt: #強制中斷
            GPIO.cleanup() 
            import sys
            sys.exit(0)  
        cap.release()
        #cv2.destroyAllWindows()
        

