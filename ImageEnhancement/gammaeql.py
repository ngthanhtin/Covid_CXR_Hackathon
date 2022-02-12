import numpy as np
import cv2

def gamma_correction_auto(RGBimage, equalizeHist = True):
    originalFile = RGBimage.copy()
    red = RGBimage[:,:,2]
    green = RGBimage[:,:,1]
    blue = RGBimage[:,:,0]

    vidsize = (600, 1000)
    forLuminance = cv2.cvtColor(originalFile,cv2.COLOR_BGR2YUV)
    Y = forLuminance[:,:,0]
    totalPix = vidsize[0]* vidsize[1]
    summ = np.sum(Y[:,:])
    Yaverage = np.divide(totalPix,summ)

    epsilon = 1.19209e-007
    correct_param = np.divide(-0.3,np.log10([Yaverage + epsilon]))
    correct_param = 0.7 - correct_param 

    red = red/255.0
    red = cv2.pow(red, correct_param)
    red = np.uint8(red*255)
    if equalizeHist:
        red = cv2.equalizeHist(red)
    
    green = green/255.0
    green = cv2.pow(green, correct_param)
    green = np.uint8(green*255)
    if equalizeHist:
        green = cv2.equalizeHist(green)
        
    blue = blue/255.0
    blue = cv2.pow(blue, correct_param)
    blue = np.uint8(blue*255)
    if equalizeHist:
        blue = cv2.equalizeHist(blue)
    
    output = cv2.merge((blue,green,red))
    #print(correct_param)
    return output

if __name__ == '__main__':
    image_path = "../TrainSet/TrainSet/P_1_1.png"
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gammaequal = gamma_correction_auto(img, equalizeHist = True)

    cv2.imshow('Gamma + EqualizeHistogram', gammaequal)
    cv2.imwrite("a.png", gammaequal)
    cv2.waitKey(0)