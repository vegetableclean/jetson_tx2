#demo on tx2
#coding=UTF-8
import sys
import argparse
import cv2
import time
import scipy.misc
import matplotlib.pyplot as plt
# 1.讀取圖檔
man = cv2.imread('man.jpeg')
woman = cv2.imread('woman.jpeg')
from revise import Model
WINDOW_NAME = 'CameraDemo'
from load_face_dataset import preprocessing


def parse_args():

    # Parse input arguments

    desc = 'Capture and display live camera video on Jetson TX2/TX1'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--rtsp', dest='use_rtsp',
                        help='use IP CAM (remember to also set --uri)',
                        action='store_true')
    parser.add_argument('--uri', dest='rtsp_uri',
                        help='RTSP URI, e.g. rtsp://192.168.1.64:554',
                        default=None, type=str)
    parser.add_argument('--latency', dest='rtsp_latency',
                        help='latency in ms for RTSP [200]',
                        default=200, type=int)
    parser.add_argument('--usb', dest='use_usb',
                        help='use USB webcam (remember to also set --vid)',
                        action='store_true')
    parser.add_argument('--vid', dest='video_dev',
                        help='device # of USB webcam (/dev/video?) [1]',
                        default=1, type=int)
    parser.add_argument('--width', dest='image_width',
                        help='image width [1920]',
                        default=1920, type=int)
    parser.add_argument('--height', dest='image_height',
                        help='image height [1080]',
                        default=1080, type=int)
    args = parser.parse_args()
    return args

def open_cam_rtsp(uri, width, height, latency):
    gst_str = ('rtspsrc location={} latency={} ! '
               'rtph264depay ! h264parse ! omxh264dec ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(uri, latency, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def open_cam_usb(dev, width, height):

    # We want to set width and height here, otherwise we could just do:

    #     return cv2.VideoCapture(dev)
    gst_str = ('v4l2src device=/dev/video{} ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)RGB ! '
               'videoconvert ! appsink').format(dev, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
	
def open_cam_onboard(width, height):
    # On versions of L4T prior to 28.1, add 'flip-method=2' into gst_str
    gst_str = ('nvcamerasrc ! '
               'video/x-raw(memory:NVMM), '
               'width=(int)2592, height=(int)1458, '
               'format=(string)I420, framerate=(fraction)30/1 ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_window(width, height):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, 'Camera Demo for Jetson TX2/TX1')

def read_cam(cap):
    show_help = True
    full_scrn = False
    help_text = '"Esc" to Quit, "H" for Help, "F" to Toggle Fullscreen'
    font = cv2.FONT_HERSHEY_PLAIN
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            # Check to see if the user has closed the window
            # If yes, terminate the program
            break
        _, img = cap.read() # grab the next image frame from camera
        if show_help:
            cv2.putText(img, help_text, (11, 20), font,
                        1.0, (32, 32, 32), 4, cv2.LINE_AA)
            cv2.putText(img, help_text, (10, 20), font,
                        1.0, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, img)
        key = cv2.waitKey(10)
        if key == 27: # ESC key: quit program
            break
        elif key == ord('H') or key == ord('h'): # toggle help message
            show_help = not show_help
        elif key == ord('F') or key == ord('f'): # toggle fullscreen
            full_scrn = not full_scrn
            if full_scrn:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)

            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)
def main():
    args = parse_args()
    print('Called with args:')
    print(args)
    print('OpenCV version: {}'.format(cv2.__version__))
    if args.use_rtsp:
        cap = open_cam_rtsp(args.rtsp_uri,
                            args.image_width,
                            args.image_height,
                            args.rtsp_latency)
    elif args.use_usb:
        cap = open_cam_usb(args.video_dev,
                           args.image_width,
                           args.image_height)
    else: # by default, use the Jetson onboard camera
        cap = open_cam_onboard(args.image_width,
                               args.image_height)

    if not cap.isOpened():
        sys.exit('Failed to open camera!')
    open_window(args.image_width, args.image_height)
    read_cam(cap)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args=parse_args()
    #define FPS
    counter=0
    #輸入模型
    model = Model()
    model.load_model(r'./model/gender_training.model.h5')
    #框顏色
    color_w = (0, 0, 255)
    color_m = (255, 0, 0)
    #捕捉相機
    cap = cv2.VideoCapture("/dev/video1")    
                     
    #haarlike
    cascade_path = './haarcascade_frontalface_alt2.xml'
    #計數
    counter_man=0
    counter_woman=0   
    #迴圈設定   
    loop=0
 
    while loop==0:
        _, frame = cap.read()   #读取一帧视频
        #FPS
        counter=0
        x=1
        start_time=time.time()
        cv2.putText(frame,'FPS:', 
                                (10, 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (0,255,0),                           #颜色
                                2) 
        cv2.putText(frame,'Hello welcome~', 
                                (200, 100),                      #坐标

                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (0,255,0),                           #颜色
                                2)   
        
            #图像灰化，降低计算复杂度
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #使用人脸识别分类器，读入分类器
        cascade = cv2.CascadeClassifier(cascade_path)                

            #利用分类器识别出哪个区域为人脸
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))        
        if len(faceRects) > 0:                 
            for faceRect in faceRects: 
                x, y, w, h = faceRect

                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                print("image size:",image.shape)
                #將原始圖像預處理
                image = preprocessing(image)
                print("image resize:",image.shape)
                 #轉換為灰階
                
                faceID = model.face_predict(image)   
                print('predict label:',faceID)
                counter+=1
                    #如果是“我”
                
                if faceID == 0:  
                    counter_man=counter_man+1				
                  
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color_m, thickness = 2)

                        #文字提示是谁
                    cv2.putText(frame,'man', 
                                (x + 30, y + 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                color_m,                           #颜色
                                2)
                   
                    if counter_man >=10:
                       cv2.putText(frame,'man_check', 
                               (x + 30, y + 30),                      #坐标
                               cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (0,0,255),                           #颜色
                                2)
                       print('recognize as man')
                       plt.imshow(man) #原始照片
                       plt.show()	
                       break
                        
                else:
                    counter_woman=counter_woman+1                   					
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color_w, thickness = 2)

                        #文字提示是谁
                    cv2.putText(frame,'woman', 
                                (x + 30, y + 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                color_w,                           #颜色
                                2)
                    
                   # print('counter_man=', end='')
                   # print(counter_man)
                    if counter_woman >=10:
                        cv2.putText(frame,'woman_check', 
                                (x - 30, y - 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                               1,                                  #字号
                                (255,0,0),                           #颜色
                                2)
                        print('recognize as woman')
                        plt.imshow(woman) #原始照片
                        plt.show()
                        break
                        #cv2.imwrite(r"C:\Users\vegetableclean\Anaconda3\Lib\site-packages\conda\practice\outcome_pic\1.jpg", frame)						
                        #time.sleep(1)
                        #loop=1
                        #b=man()
                if (time.time() - start_time) > 5 :
                    
                    cv2.putText(frame,'%s'%int(counter / (time.time() - start_time)), 
                                (90, 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (0,255,0),                           #颜色
                                2)  
                    print("******---------------FPS-----------------*****:%s frame per second ", time.time() - start_time)
                    
                    counter = 0
                    start_time = time.time()
                
                else:
                    print((time.time() - start_time))
                    cv2.putText(frame,'%s'%int(counter / (time.time() - start_time)), 
                                (90, 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (0,255,0),                           #颜色
                                2)     
               	
        cv2.imshow("Recognise your gender", frame)

            #等待10毫秒看是否有按键输入
        k = cv2.waitKey(10)
            #如果输入q则退出循环
        if loop==1:
            break

        #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()
