#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np #importing numpy to make use of arrays
import cv2 #importing cv2 to make use of reading and writing images
import math #importing math to do basic math calculation

#reading image using imread by giving it the path to our image and also mentioning that it is a grayscale image
img = cv2.imread("C:/Users/anvit/OneDrive/Desktop/CV PROJECT/test/Zebra-crossing-1.bmp", cv2.IMREAD_GRAYSCALE)

#converting our image into an array using numpy
im = np.array(img, dtype=float)

#storing the height and weight of the image for further use
im_height = im.shape[0]
im_width = im.shape[1]

#creating few arrays for later use in our algorithm
output = np.full((im_height, im_width), -1)

new_img_array = np.full((im_height,im_width), -1)
final_array = np.full((im_height,im_width), -1)
new_image = np.full((im_height, im_width), -1)

g_mag = np.full((im_height, im_width), -1)
G_X = np.full((im_height, im_width), -1)
G_Y = np.full((im_height, im_width), -1)

sector = np.full((im_height, im_width), -1)
nms_output = np.full((im_height, im_width), -1)

double_threshold_output = np.full((im_height, im_width), -1)

## Guassian Smoothing

#our given mask
guassian_mask = np.array([[1,1,2,2,2,1,1],
                          [1,2,2,4,2,2,1],
                          [2,2,4,8,4,2,2],
                          [2,4,8,16,8,4,2],
                          [2,2,4,8,4,2,2],
                          [1,2,2,4,2,2,1],
                          [1,1,2,2,2,1,1]])

# normalizing with the sum of the mask i.e., 140
guassian_mask_normalized = (1.0/140)*np.array([[1,1,2,2,2,1,1],
                                               [1,2,2,4,2,2,1],
                                               [2,2,4,8,4,2,2],
                                               [2,4,8,16,8,4,2],
                                               [2,2,4,8,4,2,2],
                                               [1,2,2,4,2,2,1],
                                               [1,1,2,2,2,1,1]])

#calculating the sum of it to do further operations - 1D guassian mask
tot = sum(guassian_mask_normalized)

#guassian smoothing function, implemented 2D by performing convolution with a 1D mask in x direction
#and then with the resultant 1D mask in y direction(applied 1D twice)
def guassian_filter(im, tot):
    #our boundary is 3 because our filter is a 7x7 array, so 3 above, below , and to the sides and a center pixel
    boundary = 3
    #convolution with 1D guassian mask in x direction
    for row in range(0, im_height):
        for col in range(boundary, im_width - boundary):
            val = im[row][col - 3] * tot[0]
            val = val + im[row][col - 2] * tot[1]
            val = val + im[row][col - 1] * tot[2]
            val = val + im[row][col] * tot[3]
            val = val + im[row][col + 1] * tot[4]
            val = val + im[row][col + 2] * tot[5]
            val = val + im[row][col + 3] * tot[6]
            new_img_array[row,col] = val
    #convolution with the resultant 1D mask in y direction
    for row in range(boundary, im_height - boundary):
        for col in range(0, im_width):
            val = new_img_array[row - 3][col] * tot[0]
            val = val + new_img_array[row - 2][col] * tot[1]
            val = val + new_img_array[row - 1][col] * tot[2]
            val = val + new_img_array[row][col] * tot[3]
            val = val + new_img_array[row + 1][col] * tot[4]
            val = val + new_img_array[row + 2][col] * tot[5]
            val = val + new_img_array[row + 3][col ] * tot[6]
            final_array[row,col] = val
    cv2.imwrite('C:/Users/anvit/OneDrive/Desktop/CV PROJECT/test/Zebra-crossing-1-1-GAUSSIAN.bmp', final_array)
    return final_array

#calling guassian smoothing on the given image
# gaussian_output2 = guassian_filter(im, tot)

## Sobels Operator

#sobels operator
def sobels_operator(inp):
    g_x = np.array([[-1,0,1],
                   [-2,0,2],
                   [-1,0,1]])
    g_y = np.array([[1,2,1],
                   [0,0,0],
                   [-1,-2,-1]])
    
    #Horizontal Gradient 
    G_X = conv_sobel(inp,g_x)
    #Saving the image
    cv2.imwrite('C:/Users/anvit/OneDrive/Desktop/CV PROJECT/test/Zebra-crossing-1-2-NORM_sobelGX.bmp', G_X)
       
    #Vertical Gradient
    G_Y = conv_sobel(inp,g_y)
    #saving the image
    cv2.imwrite('C:/Users/anvit/OneDrive/Desktop/CV PROJECT/test/Zebra-crossing-1-3-NORM_sobelGY.bmp', G_Y)
    
    #Gradient Magnitude
    g_mag = np.sqrt(np.square(G_X) + np.square(G_Y))
    #saving the image
    cv2.imwrite('C:/Users/anvit/OneDrive/Desktop/CV PROJECT/test/Zebra-crossing-1-4-_sobel.bmp', g_mag)    
    return G_X,G_Y, g_mag

#convolution done by implementing using 2D directly
def conv_sobel(x,y):
    
    #height and width of the image
    xH = x.shape[0]
    xW = x.shape[1]
    #a new array to store the output of convolution
    sobel_image = np.zeros((xH, xW))
    
    #height and width of the mask
    yH = y.shape[0]
    yW = y.shape[1]  
    
    #for loopinng inside the boundary, we divide by 2
    yH_new = int((yH - 1) // 2)
    yW_new = int((yW - 1) // 2)
    
    #selecting the range so the we only consider pixels that are inside the boundary/border
    for i in range(yH_new, xH-yH_new):
        for j in range(yW, xW-yW_new):
            
            #consider a variable to calculate the summation over the mask
            summation = 0
            
            #to check the corresponding mask value , we have to set ranges
            for k in np.arange(-yH_new, yH_new+1):
                for l in np.arange(-yW_new, yW_new+1):
                    
                    #m is the value of the corresponding image pixel
                    m = x[i+k, j+l]
                    #n is the value of the corresponding mask pixel
                    n = y[yH_new+k, yW_new+l]
                    #summation is the sum of their products over the range
                    summation += (m * n)
            
            #To normalize our image to contain values from 0 to 255 we divide the summation by 4
            sobel_image[i,j] = summation / 4 
            
    return sobel_image

## Non-Maxima Supression

#Non-Maxima Supression
def nms(theta,sobel_output):
    
    #for every pixel in our array we have to figure out it's sector 
    for i in range(im_height):
        for j in range(im_width):
            
            #Assigning sectors based on theta value
            if((theta[i][j] >= 0 and theta[i][j] < 22.5) or (theta[i][j] >= 157.5 and theta[i][j] < 202.5) or (theta[i][j] >= 337.5 and theta[i][j] <= 360)):
                sector[i][j] = 0
            if((theta[i][j] >= 22.5 and theta[i][j] < 67.5) or (theta[i][j] >= 202.5 and theta[i][j] < 247.5)):
                sector[i][j] = 1
            if((theta[i][j] >= 67.5 and theta[i][j] < 112.5) or (theta[i][j] >= 247.5 and theta[i][j] < 292.5)):
                sector[i][j] = 2
            if((theta[i][j] >= 112.5 and theta[i][j] < 157.5) or (theta[i][j] >= 292.5 and theta[i][j] < 337.5)):
                sector[i][j] = 3
    
    #Here we do -5 to ignore the boundary pixels
    for i in range(5, im_height-5):
        for j in range(5, im_width-5):
            
            #We check for each sector, if the pixel is non maxima then we supress it
            if(sector[i][j] == 0):
                if(sobel_output[i][j] > sobel_output[i][j+1] and sobel_output[i][j] > sobel_output[i][j-1]):
                    nms_output[i][j] = sobel_output[i][j]
                else:
                    nms_output[i][j] = 0
                    
            if(sector[i][j] == 1):
                if(sobel_output[i][j] > sobel_output[i-1][j+1] and sobel_output[i][j] > sobel_output[i+1][j-1]):
                    nms_output[i][j] = sobel_output[i][j]
                else:
                    nms_output[i][j] = 0
            
            if(sector[i][j] == 2):
                if(sobel_output[i][j] > sobel_output[i-1][j] and sobel_output[i][j] > sobel_output[i+1][j]):
                    nms_output[i][j] = sobel_output[i][j]
                else:
                    nms_output[i][j] = 0
            
            if(sector[i][j] == 3):
                if(sobel_output[i][j] > sobel_output[i+1][j+1] and sobel_output[i][j] > sobel_output[i-1][j-1]):
                    nms_output[i][j] = sobel_output[i][j]
                else:
                    nms_output[i][j] = 0
    
    #saving the image
    cv2.imwrite('C:/Users/anvit/OneDrive/Desktop/CV PROJECT/test/Zebra-crossing-1-5-_NMS.bmp', nms_output)         
    return nms_output               

## Double Thresholding

#Double Thresholding
#t1 is low threshold, t2 = 2*t1 is the high threshold
t1 = 15
t2 = 30
def double_threshold(theta,sobel_output,nms_result,t1,t2,double_threshold_output):
    
    #check if pixel is <low threshold, >high threshold, equal or in between both
    for i in range(1,im_height-1):
        for j in range(1,im_width-1):
            
            #if <low threshold, assign it a 0
            if(nms_result[i][j] < t1 and double_threshold_output[i][j] == -1):
                double_threshold_output[i][j] = 0
            #if >high thresold, assign it a 255
            if(nms_result[i][j] > t2 and double_threshold_output[i][j] == -1):
                double_threshold_output[i][j] = 255
            
            #if equal to or in-between low and high threshold, perform 'check' function as defined below
            if(nms_result[i][j] >= t1 and nms_result[i][j] <= t2 and double_threshold_output[i][j] == -1):
                check(i,j,double_threshold_output,t2,theta,sobel_output)
                
    #saving the image
    cv2.imwrite('C:/Users/anvit/OneDrive/Desktop/CV PROJECT/test/Zebra-crossing-1-6-_DOUBLETHRESOLDING.bmp', double_threshold_output)

#To check if the 8 connected neighbour has gradient magnitude >high threshold and the absolute difference between them is 45 degrees
def check(i,j,double_threshold_output,t2,theta,sobel_output):    
    
    if((sobel_output[i-1][j-1] > t2) and (np.absolute(theta[i][j] - theta[i-1][j-1]) <= 45)):
        double_threshold_output = 255
    elif((sobel_output[i-1][j] > t2) and (np.absolute(theta[i][j] - theta[i-1][j]) <= 45)):
        double_threshold_output= 255
    elif((sobel_output[i-1][j+1] > t2) and (np.absolute(theta[i][j] - theta[i-1][j+1]) <= 45)):
        double_threshold_output = 255
    elif((sobel_output[i][j-1] > t2) and (np.absolute(theta[i][j] - theta[i][j-1]) <= 45)):
        double_threshold_output = 255
    elif((sobel_output[i][j+1] > t2) and (np.absolute(theta[i][j] - theta[i][j+1]) <= 45)):
        double_threshold_output = 255
    elif((sobel_output[i+1][j-1] > t2) and (np.absolute(theta[i][j] - theta[i+1][j-1]) <= 45)):
        double_threshold_output = 255  
    elif((sobel_output[i+1][j] > t2) and (np.absolute(theta[i][j] - theta[i+1][j]) <= 45)):
        double_threshold_output= 255
    elif((sobel_output[i+1][j+1] > t2) and (np.absolute(theta[i][j] - theta[i+1][j+1]) <= 45)):
        double_threshold_output = 255
    else:
        double_threshold_output = 0
        
    return double_threshold_output


#calling guassian smoothing on the given image
gaussian_output2 = guassian_filter(im, tot)

#calling sobels opertor
g_x,g_y,sobel_output = sobels_operator(gaussian_output2)

#Finding the Gradient Angles i.e., theta
#arctan2 is tan inverse, which gives us results in radians with [-pi to pi]
theta = np.arctan2(g_y,g_x)
#Converting radians to degrees
theta = np.rad2deg(theta)
#converting all the negatives into positives by adding 360 so the range is now [0 to 360]
for i in range(3, im_height - 3):
        for j in range(3, im_width - 3):
            if(theta[i][j] < 0):
                theta[i][j] += 360

#calling non maxima supression
nms_result = nms(theta,sobel_output)

#calling double thresholding
dtt_output = double_threshold(theta,sobel_output,nms_result,t1,t2,double_threshold_output)


# In[ ]:




