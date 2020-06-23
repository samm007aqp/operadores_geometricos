import cv2
import numpy as np 
from matplotlib import pyplot as plt 

def median(i,j,row,col,origin):
	lin = 3
	arriba =  i-lin
	abajo =   i+lin
	isq =	j-lin
	der = 	j+lin
	if arriba < 0:
		arriba = 0
	if abajo > row:
		abajo = row-1
	if isq < 0:
		isq = 0
	if der > col:
		der = col-1
	x = origin[arriba:abajo,isq:der]
	x1 = np.median(x[:,:,0])
	x2 = np.median(x[:,:,1])
	x3 = np.median(x[:,:,2])
	return [int(x1),int(x2),int(x3)]

def mean(i,j,row,col,origin):
	lin = 1
	arriba =  i-lin
	abajo =   i+lin
	isq =	j-lin
	der = 	j+lin
	if arriba < 0:
		arriba = 0
		
	if abajo > row:
		abajo = row-1
	if isq < 0:
		isq = 0
	if der > col:
		der = col-1
	total = 8
	subimg = origin[arriba:abajo,isq:der]
	r = np.sum(subimg[:,:,0])/total
	g = np.sum(subimg[:,:,1])/total
	b = np.sum(subimg[:,:,2])/total
	print([r,g,b])
	return [r,g,b]

def median_filter(inpu,rows,cols):
	origin = inpu.copy()
	for i in range (rows):
		for j in range(cols):
			if inpu[i,j,0] < 1 and inpu[i,j,1] < 1 and inpu[i,j,2] < 1:
				inpu[i,j] = median(i,j,rows,cols,origin)



img = cv2.imread("my_war_salida-angle.jpg")
rows, cols , channels = img.shape
x = img[300:304,200:204]
print(x)
print(np.median(x, axis = 0)) 
print(np.median(x, axis = 1)) 
print(np.median(x, axis = 2)) 
median_filter(img,rows,cols)
cv2.imwrite("medial_salida.jpg",img)

