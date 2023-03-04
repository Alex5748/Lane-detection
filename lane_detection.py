import cv2
import numpy as np
import matplotlib.pyplot as plt
import os



min_width_rect = 80
min_height_rect =80

# initialize algorithm
algo = cv2.createBackgroundSubtractorMOG2(detectShadows=False,varThreshold=900)



def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), 0)
    canny = cv2.Canny(blur, 5, 150)
    return canny




def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(127,458 ), (135, 17), (575, 15), (580, 465)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    
    return masked_image
    


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 4)
    return line_image




cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, frame = cap.read()
    cropped_image = region_of_interest(frame)
    canny_image = canny(cropped_image)
    lines = cv2.HoughLinesP(canny_image, 2, np.pi/180, 100,
                        np.array([]), minLineLength=40, maxLineGap=50)
    line_image = display_lines(frame, lines)
    combo_image = cv2.addWeighted(frame, 1, line_image, 1, 1)
    
    
                
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(frame, (3, 3), 5)
    

    # apply on all frame
    image_sub = algo.apply(cropped_image)
    dilat = cv2.dilate(image_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    countour_shape,h = cv2.findContours(
        dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("combo_image",image_sub)

    for (i, c) in enumerate(countour_shape):
        (x, y, w, h) = cv2.boundingRect(c)
        valid_vech = (w >= min_width_rect) and (h >= min_height_rect)
        if not valid_vech:
            continue
        
        rectangle_draw = cv2.rectangle(combo_image, (x,y), (x+h, y+h), (0,255,0), 2)
        
        #roi= (x, y, x+h, y+h)
        print(x+h)
        
        #rectangle box formed right side line
        line1=cv2.line(combo_image, (x+h,y),(x+h,y+h), (255,0,0))
        x5=432
        y5=442
        x2=445
        y2=28
        #fixed lane on the road(coordinated manually given)
        draw_line1=cv2.line(combo_image,(x5,y5),(x2,y2),(255,0,0,),1)
        
        
        
        if(x+h>x5):
            print("vehicle crossed lane")
            vehicle_image = combo_image[x:w, y:h]
            #cv2.imwrite('.jpg',vehicle_image)
            #vehicle_image.save("lanecrossed.jpg")
            
            #save path for cropped image
            save_dir = r'D:\New folder\minor project\Lane Detection'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(os.path.join(save_dir, 'cropped_image.jpg'), vehicle_image)
            cv2.imshow("cropped_image",vehicle_image)
        else:
            print("lane is not crossed")
            
      
    cv2.imshow("result", combo_image)
    
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
