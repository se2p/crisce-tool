import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import imutils
import cv2


### constants
RED_CAR_BOUNDARY = np.array([[0, 190, 210],
                            [179, 255, 255]])

BLUE_CAR_BOUNDARY = np.array([[85, 30, 40],
                            [160, 255, 255]])

class Pre_Processing():
    
    def __init__(self, red_car_boundary=RED_CAR_BOUNDARY, blue_car_boundary=BLUE_CAR_BOUNDARY):
        self.red_car_boundary = red_car_boundary
        self.blue_car_boundary = blue_car_boundary


    def readImage(self, image_path):
        """ Read the image and create a blank mask"""
        image = cv2.imread(image_path)
        return image

    def changeColorSpace(self, image, color_code):
        """
        Input: 
            image 
            color_code = cv2.COLOR_BGR2HSV , COLOR_BGR2GRAY, COLOR_BGR2HLS, COLOR_BGR2RGB 
        output: 
            image
        """
        image = cv2.cvtColor(image, color_code)
        return image

    def resize(self, image):
        if image.shape[0] >= 1000:
            image = imutils.resize(image=image, height=960)
        elif image.shape[1] >= 500:
            image = imutils.resize(image=image, width=400)
        return image

    def getMask(self, image):
        h, w = image.shape[:2]
        mask = np.zeros((h, w, 3), np.uint8)
        return mask

    def getMaskWithRange(self, image):
        """
        Input: hsv(or hsl) and image 
        Output: tuple consisting of mask_red, mask_blue
        """
        mask_red = cv2.inRange(
            image, self.red_car_boundary[0], self.red_car_boundary[1])
        mask_blue = cv2.inRange(
            image, self.blue_car_boundary[0], self.blue_car_boundary[1])
        return (mask_red, mask_blue)

    def bitwiseAndOperation(self, image, mask):
        result = cv2.bitwise_and(image, image, mask=mask)
        return result

    def bitwiseOrOperation(self, image_1, image_2):
        result = cv2.bitwise_or(image_1, image_2)
        return result
    
    def threshold(self, image, lower_limit=50, upper_limit=255, type_of_thresh=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU):
        """
        Threshold the image
        """
        _, thresh = cv2.threshold(image, lower_limit, upper_limit, type_of_thresh)
        return thresh

    def blurImage(self, image, kernel_size=(3, 3), sigmaX=0):
        """ 
        Applying Gaussian Bluring on the image 
        
        Input: Image, kernel_size 
        Output: Blured Image
        """
        blur = cv2.GaussianBlur(image, kernel_size, sigmaX)
        return blur

    def applyMorphologicalOperation(self, image, kernel_window=(5, 5), morph_operation=cv2.MORPH_OPEN):
        """ Morphological Operations """
        # rect_kernel = cv2.getStructuringElement( cv2.MORPH_RECT,(10,10))
        # ellipse_kernel = cv2.getStructuringElement( cv2.MORPH_ELLIPSE,(15,15))
        # morph_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, ellipse_kernel)
        kernel = np.ones(kernel_window, np.uint8)
        morph_image = cv2.morphologyEx(image, morph_operation, kernel)
        return morph_image

    def dilate(self, image, kernel_window=(10, 10), iterations=1):
        """
        Input: image, kernel_size, Viz 
        Output: dilated image
        """
        # Increasing the pixel intesities along X and Y-axis of Horizontal and Vertical Morphological Kernal 
        kernel  = np.ones(kernel_window, np.uint8)  # note this is a horizontal kernel
        dilate_img  = cv2.dilate(image, kernel, iterations=iterations)
        return dilate_img
    
    def erode(self, image, kernel_window=(10, 10), iterations=1):
        """
        Input: image, kernel_size, Viz 
        Output: eroded image
        """
        # Increasing the pixel intesities along X and Y-axis of Horizontal and Vertical Morphological Kernal
        kernel  = np.ones(kernel_window, np.uint8)  # note this is a rectangular kernel
        erode_img = cv2.erode(image, kernel, iterations=iterations)
        return erode_img

    def plotFigure(self, image, cmap="brg", title="Figure", figsize=(20, 20)):
        """ Plotting the image 
        Input: image, cmap, title, figsize
        """
        fig = plt.figure(figsize=figsize)
        ax1 = fig.subplots(1, sharey=True, sharex=True)
        # plt.imshow(gray, cmap='gray')
        ax1.imshow(image, cmap=cmap)
        ax1.set_title(title)
        plt.show()

    def showImage(self, title, image, time=0):
        """ Bluring the image 
        Input: title, image
        """
        cv2.imshow(title, image)
        cv2.waitKey(time)
        cv2.destroyAllWindows()
        
    def saveFigure(self, image_name, dpi=300):
        plt.savefig(image_name + '.jpg', dpi=dpi)
        

