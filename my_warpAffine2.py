import cv2
import numpy as np 
from matplotlib import pyplot as plt 
import math as ma 
from decimal import Decimal, ROUND_HALF_UP



def size_angle(teta,h,w):
	if teta< ma.pi/2:
		new_w = ( (w)*ma.cos(teta)+( (h)*ma.sin(teta)))
		new_h = ( (w)*ma.sin(teta)+( (h)*ma.cos(teta)))
	else :
		h_ = w
		w_ = h
		teta = teta-ma.pi/2
		new_w = ( (w_)*ma.cos(teta)+( (h_)*ma.sin(teta)))
		new_h = ( (w_)*ma.sin(teta)+( (h_)*ma.cos(teta)))

	return abs(int(new_h)), abs(int(new_w))

def median(i,j,row,col):
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
	#total = 4*lin*lin
	return np.median(tmp[arriba:abajo,isq:der].reshape(-1,3), axis=0)

def median_filter(input,rows,cols):
	origin = input.clone()
	for i in range (rows):
		for j in range(cols):
			if input[i,j,0] < 1 and input[i,j,1] < 1 and input[i,j,2] < 1:
				input[i,j] = median(i,j,rows,cols)

def my_warpAffine(input,matrix,rows,cols):
	#A = np.float32([[1,0],[0,1]])
	#input = input.astype(int)
	out = np.zeros([rows+ int(matrix[1:2,2:3]),cols+int(matrix[0:1,2:3]), 3], dtype=np.uint8)
	print(out.shape)
	#out = out.astype(np.uint8)
	for y in range(rows):
		for x in range(cols):
			aux = np.dot(matrix[:,0:2],[[y],[x]]) + matrix[:,2:3] 
			out[int(aux[1:2,:]),int(aux[0:1,:])] = input[x,y]
	#out = out.astype(np.uint8)
	cv2.imshow("salida",out)
	cv2.imwrite("my_war_salida.jpg",out[0:rows,0:cols])
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def my_warpAffine2(input,matrix,rows,cols):
	if rows:
		nr,nc = size_angle(Angle,rows,cols)
		nr = round(rows/nr,1)
		nc = round(cols/nc,1)
		input = cv2.resize(input,None,fx=nc,fy=nr,interpolation=cv2.INTER_CUBIC)
	out = np.zeros([rows ,cols, 3], dtype=np.uint8)
	print(out.shape)
	rr,cc,cha = input.shape
	print([rr,cc])
	for y in range(rr):
		for x in range(cc):
			aux = np.dot(matrix[:,0:2],[[y],[x]]) + matrix[:,2:3] 
			ty = float(aux[1,0] + ((cols-cc)//2) )
			tx = float( aux[0,0]+((rows-rr)//2) )
			iy = Decimal(ty).quantize(0,ROUND_HALF_UP)
			ix = Decimal(tx).quantize(0,ROUND_HALF_UP)
			if iy>0 and iy<rows and ix>0 and ix<cols :
				#print(str(iy)+"_"+str(ty)+";"+str(ix)+"_"+str(tx))
				out[int(iy),int(ix)] = input[x,y]

	cv2.imshow("salida",out)
	cv2.imwrite("my_war_salida-angle.jpg",out)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def my_warpAffine3(input,matrix,rows,cols):
	if rows:
		nr,nc = size_angle(Angle,rows,cols)
	print([nr,nc])
	out = np.zeros([nr ,nc, 3], dtype=np.uint8)
	for y in range(rows):
		for x in range(cols):
			aux = np.dot(matrix[:,0:2],[[y],[x]]) + matrix[:,2:3] 
			ty = float(aux[1,0]+ ((nc-cols)//2) ) 
			tx = float( aux[0,0]+((nr-rows)//2) )
			iy = Decimal(ty).quantize(0,ROUND_HALF_UP)
			ix = Decimal(tx).quantize(0,ROUND_HALF_UP)
			if iy>0 and iy<nr and ix>0 and ix<nc :
				#print(str(iy)+"_"+str(ty)+";"+str(ix)+"_"+str(tx))
				out[int(iy),int(ix)] = input[x,y]

	cv2.imshow("salida",out)
	cv2.imwrite("my_war_salida-angle.jpg",out)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def make_matrix_angle(angle):
	Bx = (1-ma.cos(angle))*(cols//2) - ma.sin(angle)*(rows//2)
	By = ma.sin(angle)*(cols//2) + (1-ma.cos(angle))*(rows//2)
	M = np.float32([[ma.cos(angle),ma.sin(angle),Bx],[-ma.sin(angle),ma.cos(angle),By]])
	return M

img = cv2.imread("julio.jpg")
rows, cols , channels = img.shape
Angle = ma.pi*(90/180)
#M = np.int32([[1,0,100],[0,1,20]])
M = make_matrix_angle(Angle)
#M = np.float32([[1,0.2,0],[0.1,1,0]])
my_warpAffine3(img,M,rows,cols)


