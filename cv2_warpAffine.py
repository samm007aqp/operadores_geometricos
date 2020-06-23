import cv2
import numpy as np 
from matplotlib import pyplot as plt 
import math as ma 
# https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html
# fuente general
# Progama que traslada, scala, rota, imagenes usando la libreria opencv

img = cv2.imread("julio.jpg")
rows, cols, channels = img.shape
#=---------------------- Traslation move the entire image in the x and y axis
traslation = img.copy()
M = np.float32([[1,0,50],[0,1,70]])
traslation = cv2.warpAffine(traslation,M,(cols,rows))
#cv2.imwrite("salida.jpg")
#cv2.imshow("translation",traslation)
#-------------- ----------Scaling /=-=====---
scaling = img.copy()
#cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])
# scr input image
# dsize desired size for the output imge
# fx scale factor along the horizontal axis
# fy scale factor aling the vertical axis
#interpolation INTER_NEAREST INTER_LINEAR INTER_AREA INTER_NEAREST INTER_CUBIC 
# https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/

#scaling = cv2.resize(scaling,None,fx=1.5,fy=1.5,interpolation=cv2.INTER_CUBIC)
#scaling = cv2.resize(scaling,(3*cols, 2*rows), interpolation = cv2.INTER_CUBIC)
#cv2.imshow("scaling",scaling)
M = np.float32([[1,0,0],[0,1,0]])
scaling = cv2.warpAffine(scaling ,M,(int(cols*1.5),int(rows*1.5)),flags = cv2.INTER_LINEAR  )
#cv2.imshow("scaling",scaling)
#======================= Rotacion ===========
rotate = img.copy()
M2 = cv2.getRotationMatrix2D((cols/2,rows/2),22,1)
#v2.getRotationMatrix2D(center, angle, scale) j
'''Parameters:	
	center – Center of the rotation in the source image.
	angle – Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
	scale – Isotropic scale factor.
	map_matrix – The output affine transformation, 2x3 floating-point matrix.
'''
print(M2)
Bx = (1-ma.cos(ma.pi/8))*(cols/2) - ma.sin(ma.pi/8)*(rows/2)
By = ma.sin(ma.pi/8)*(cols/2) + (1-ma.cos(ma.pi/8))*(rows/2)
M = np.float32([[ma.cos(ma.pi/8),ma.sin(ma.pi/8),Bx],[-ma.sin(ma.pi/8),ma.cos(ma.pi/8),By]])
print(M)
rotate = cv2.warpAffine(img,M,(cols,rows))
cv2.imshow("rotate",rotate)
#===========================Shear==========
shear = img.copy()
M = np.float32([[1,0.2,0],[0.1,1,0]])
shear = cv2.warpAffine(shear,M,(cols,rows))
#cv2.imshow("shear",shear)

cv2.waitKey(0)
cv2.destroyAllWindows()

