import cv2
import numpy as np


def roi(img):
    
    #blank mask:
    mask = np.zeros_like(img)   


    vertices = np.array([[0,383],[0,330],[200,210],[312,210],[512,330],[512,383],
                         ], np.int32)

    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, 255)
    
    #returning the image only where mask pixels are nonzero
    masked = cv2.bitwise_and(img, mask)
    return masked


def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    isX = True if orient == 'x' else False
    sobel = cv2.Sobel(gray, cv2.CV_64F, isX, not isX)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    return mag_binary



def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(grad_dir)
    dir_binary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1

    return dir_binary

def apply_thresholds(image, ksize=3):
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    return combined



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


def combine_threshold(s_binary, combined):
    combined_binary = np.zeros_like(combined)
    combined_binary[(s_binary == 1) | (combined == 1)] = 1

    return combined_binary


def findVector(image):
    w,h = image.shape[:2]
    region_of_interest_vertices = [
        (h*0.1,w),
        (h*0.4,w/2),
        (h*0.6,w/2),
        (h*0.9,w)
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

def drowLines(edges,img):
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)

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
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    
    return ((x1, y1), (x2, y2))


def lane_lines(image, lines):
    left_lane, right_lane = average_slope_interceptaverage(lines)
    
    y1 = image.shape[0] # bottom of the image
    y2 = y1*0.6         # slightly lower than the middle

    left_line  = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)
    
    return left_line, right_line


def draw_lane_lines(image,lines, color=[255, 0, 0], thickness=20):
    # make a separate image to draw lines and combine with the orignal later
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    return cv2.addWeighted(image, 1.0, line_image, 1, 0.0)



def process_img(img):
    cropped_img = roi(img)
    mask = select_white_yellow(cropped_img)
    blur = cv2.GaussianBlur(mask, (5,5), 0)
    edges = cv2.Canny(blur,50,150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=50)
    left_line,right_line = lane_lines(img, lines)
    result = draw_lane_lines(img, lane_lines(img, lines))

    return result


























