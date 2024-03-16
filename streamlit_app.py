import streamlit as st
import dlib
import cv2
from imutils import face_utils
import pygame
from scipy.spatial import distance
import warnings

warnings.filterwarnings("ignore")

st.sidebar.title("RAJESH")
st.sidebar.subheader("""Click on start button to launch camera.\nClick on stop button to stop the camera""")

st.title("Eye Drowsiness Detection System")


frame_placeholder=st.empty()

face_detect=dlib.get_frontal_face_detector()
pred=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def calculate(a,b):

    dist=distance.euclidean(a,b)
    return dist



def blink(a,b,c,d,e,f):
    
    up=calculate(b,f)+calculate(c,e)
    d=calculate(a,d)
    
    res=up/(2*d)    
    return res


def draw(a,b,img):
    cv2.line(img,a,b,(255,0,0))

def capture():
    
    cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
    
    if cap is None or not cap.isOpened():
        st.error("Camera Not Found..!")
    
    cap.set(cv2.CAP_PROP_FPS,14)
    timer=0
    eye_close=False
    b=0
    m=0
    mouth_close=False
            
    while cap.isOpened():
        
        ret,img=cap.read()
        img=cv2.flip(img,1)
        img=cv2.copyMakeBorder(img,50,50,10,10,cv2.BORDER_CONSTANT)
        
        
        if ret:
            
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            
            face=face_detect(gray)
            
            for i in face:
                
                landmark=pred(gray,i)
                mark=face_utils.shape_to_np(landmark)
                
                left=blink(mark[36],mark[37],mark[38],mark[39],mark[40],mark[41])
                right=blink(mark[42],mark[43],mark[44],mark[45],mark[46],mark[47])
                mouth1=calculate(mark[62],mark[66])
                mouth2=calculate(mark[63],mark[65])
                mouth3=calculate(mark[61],mark[67])
                mouth4=calculate(mark[60],mark[64])
                
                
                avg=(left+right)/2.0
                avg_mouth=(mouth1+mouth2+mouth3)/(3.0*mouth4)
                
                
                if avg<0.2:
                    eye_close=True
                elif avg>=0.2 and eye_close:
                    b+=1
                    eye_close=False
                    
                if avg_mouth<55:
                    mouth_close=True
                elif avg_mouth>=50 and mouth_close:
                    m+=1
                    mouth_close=False
                    
                

                
                for i in range(36,41):
                    draw(mark[i],mark[i+1],img)

                
                for i in range(42,47):
                    draw(mark[i],mark[i+1],img)
                    
                draw(mark[41],mark[36],img)
                draw(mark[47],mark[42],img)
                
                
                for i  in range(48,59):
                    draw(mark[i], mark[i+1], img)
                draw(mark[48],mark[59],img)
                
                
                for i  in range(60,67):
                    draw(mark[i], mark[i+1], img)
                draw(mark[60],mark[67],img)
                
                
                if avg_mouth>0.50:
                    cv2.putText(img,"Drowsiness Alert !",(220,565),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
                    
                    
                
                if avg>=0.22:
                
                    cv2.putText(img,"Open",(20,45),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
                    timer-=0.5
                    if timer<0:
                        timer=0
                        cv2.putText(img,f'Timer : {timer}',(150,45),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
                        
                        
                    else:
                        cv2.putText(img,f'Timer : {timer}',(150,45),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
                    
                    
                    
                elif avg<0.2:
                    
                    cv2.putText(img,"Closed",(20,45),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
                    timer+=0.5
                    
                    if timer>10:
                    
                        timer=10
                        
                        cv2.putText(img,f'Timer : {timer}',(150,45),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
                        
                        cv2.putText(img,'ALERT !',(400,45),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
                        mixer.music.play()
                        
                            
                       
                    else:
                        cv2.putText(img,f'Timer : {timer}',(150,45),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)

                              
        
        frame=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame,channels="RGB",width=400)
        
        if cv2.waitKey(1)==ord('q'):
            cap.release()
            cv2.destroyAllWindows()


col=st.columns([1,1,1,1,1])
start=col[0].button("Start")
stop=col[1].button("Stop")
if start:
    pygame.mixer.init()
    pygame.mixer.music.load('alarm.mp3')
    capture()
    
if stop:
    cv2.destroyAllWindows()
    
    
    
    
