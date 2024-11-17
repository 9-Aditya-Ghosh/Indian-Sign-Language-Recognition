import cv2
import mediapipe as mp 
import time 

cap = cv2.VideoCapture(0) 

mpHands = mp.solutions.hands
hands = mpHands.Hands()  #parameters of this function are set to default
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0 

while True :
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
        
    #print(results.multi_hand_landmarks)
    
    if results.multi_hand_landmarks: 
      for handLms in results.multi_hand_landmarks: 
        for id, lm in enumerate (handLms.landmark): 
          #print(id , lm)
          h , w , c = img.shape 
          cx, cy = int(lm.x*w), int(lm.y*h)
          print(id ,cx, cy)
        mpDraw.draw_landmarks(img, handLms , mpHands.HAND_CONNECTIONS)
    
    #calc fps :    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime 
    
    cv2.putText(img,"FPS : "+str(int(fps)) , (10,50) ,
                cv2.FONT_HERSHEY_SIMPLEX, 
                1 , (0,0,0),3)

    cv2.imshow("Image" , img) 
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()