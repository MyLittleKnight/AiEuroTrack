
import cv2
import numpy as np
from matplotlib import pyplot as plt

def select_white_yellow(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    # white color mask
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(hls, lower, upper)
    # yellow color mask
    low_yellow = np.uint8([10, 0, 100])
    up_yellow = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(hls, low_yellow, up_yellow)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    return mask

def findVector(image):
    w,h = image.shape[:2]
    region_of_interest_vertices = [
        (150,w),
        (453,196),
        (627,196),
        (939,w)
    ]
    points = np.array([region_of_interest_vertices],dtype = np.int32)
    return points

def selectRegion(image,vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(image)
    
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        # in case, the input image has a channel dimension        
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) 
    
    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def  average_slope_interceptaverage(lines):
    left_lines    = [] # (slope, intercept)
    left_weights  = [] # (length,)
    right_lines   = [] # (slope, intercept)
    right_weights = [] # (length,)
    
    #for ex y = mx + c
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2==x1:
                continue # ignore a vertical line
            slope = (y2-y1)/(x2-x1) # finding m
            intercept = y1 - slope*x1 #finding c
            length = np.sqrt((y2-y1)**2+(x2-x1)**2) # finding length
            if slope < 0: # y is reversed in image
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    
    #Ağırlıklarını vektörlerin uzunluğu kabul edecek şekilde m ve c' nin ortalamasını bulalım.
    left_avr = np.dot(left_weights,left_lines)/np.sum(left_weights) if len(left_weights) >0 else None
    right_avr= np.dot(right_weights,right_lines)/np.sum(right_weights) if len(right_weights) >0 else None
    return left_avr,right_avr

def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None
    
    slope, intercept = line
    
    # make sure everything is integer as cv2.line requires it
    if  slope != 0:
        x1 = int((y1 - intercept)/slope)
    else:
        x1 = None
    if slope != 0:
        x2 = int((y2 - intercept)/slope)
    else:
        x2 = None

    y1 = int(y1)
    y2 = int(y2)
    
    if (x1 or x2) is None:
        return None
    else:
        return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    left_lane, right_lane = average_slope_interceptaverage(lines)
    
    y1 = image.shape[0] # bottom of the image
    y2 = y1*0.6         # slightly lower than the middle

    left_line  = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)
    
    return left_line, right_line

def draw_lane_lines(image,lines, color=[255, 255, 0], thickness=20):
    # make a separate image to draw lines and combine with the orignal later
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    return cv2.addWeighted(image, 1.0, line_image, 1, 0.0)



def capture_process(img):
    cropped_img = selectRegion(img,findVector(img))
    mask = select_white_yellow(cropped_img)
    blur = cv2.GaussianBlur(mask, (5,5), 0)
    edges = cv2.Canny(blur,50,150)

    return edges,cropped_img