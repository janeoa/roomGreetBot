import cv2
import numpy as np
import time
vid = cv2.VideoCapture(0)
import vlc


lastFindTime = time.time()
p = vlc.MediaPlayer("saepal.wav")

while(True):
    ret, img = vid.read()
    scale_percent = 50 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_image, 130, 255, cv2.THRESH_BINARY)
    # Creating kernel
    kernel = np.ones((10, 10), np.uint8)
    kerne2 = np.ones((15, 15), np.uint8)
    # Using cv2.erode() method 
    image = cv2.erode(thresh, kernel) 
    image = cv2.dilate(image, kerne2)
    
    contours, hierarchy = cv2.findContours(image=image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    image_copy = frame.copy()
    boxes = frame.copy()
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    #contours = contours[0] if len(contours) == 2 else contours[1]
    # print(len(contours))

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        a4woh = 29.7/21.0
        if(w/h > a4woh * 0.92 and w/h < a4woh *1.08):
            cv2.rectangle(boxes, (x, y), (x + w, y + h), (0,255,0), 2)
            lastFindTime = time.time()

    #print(time.time() - lastFindTime)       
    if(time.time() - lastFindTime > 1):
        print("hehe epta", time.time())
        p.stop()
        p = vlc.MediaPlayer("saepal.wav")
        p.play()
        lastFindTime = time.time()+10


    cv2.imshow('frame', frame)
    cv2.imshow('gray_image', gray_image)
    cv2.imshow('thresh', thresh)
    cv2.imshow('image', image)
    cv2.imshow('image_copy', image_copy)
    cv2.imshow('boxes', boxes)
    
    
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



vid.release()
cv2.destroyAllWindows()
