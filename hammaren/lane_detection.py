import cv2
import numpy as np

canny_min_val = 200
canny_max_val = 400

def change_canny_min(val):
    global canny_min_val
    print("Changing ", val)
    canny_min_val = val

def change_canny_max(val):
    global canny_max_val
    print("Changing max to ", val)
    canny_max_val = val

window_name = "lane detection"
cv2.namedWindow(window_name)
cv2.createTrackbar('Canny min', window_name, canny_min_val, 1000, change_canny_min)
cv2.createTrackbar('Canny max', window_name, canny_max_val, 1000, change_canny_max)

def detect_lanes(image):
    """
    image: OpenCV Mat frame.
    """
    global canny_min_val
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.Canny(gray_image, canny_min_val, canny_max_val)
    lines = cv2.HoughLines(gray_image,1,np.pi/180,200)
    print("Num lines in first: ", len(lines))
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)

    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(gray_image,1,np.pi/180,100,minLineLength,maxLineGap)
    print("Num lines in second: ", len(lines))
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)


    cv2.imshow(window_name, image)