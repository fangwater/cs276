import tkinter as tk
from scipy.stats import norm
from PIL import ImageTk 
from PIL import Image 
import os
import imageio
import numpy as np
import math
#data_load
path = 'LFdata/toyLF/'
Temp = [] 
for i in range(1,257):
    if i<10:
        name = "00"+str(i)
    elif i<100:
        name = "0"+str(i)
    else:
        name = str(i)
    Temp.append(imageio.imread(os.path.join(path,'lowtoys'+name+'.bmp')))


#get the photo [320*240] as a vector with the uv coord(x,y)
def get_photo(x,y):
    index = 255-x-16*y
    return Data[index]

def matrix_2_img(matrix):
    return matrix.reshape(240,320,3)
def img_2_matrix(img):
    return img.reshape(320*240,3)
def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm
def show_img(img):
    im=Image.fromarray(img) 
    im.show()
#Input:photos like as(wh,3)*n,and the weight
#Output:photos*weight
def Blend_by_weight(P,w):
    Image = np.zeros((76800,3),dtype=np.uint8)
    for i in range(len(P)):
        Temp = (np.floor(P[i]*w[i])).astype(np.uint8)
        Image = Image + Temp
    return matrix_2_img(Image)

#x(0,15),y(0,15)
def Q_interpolator(coords):
    x,y = (coords[0],coords[1])
    x_ = int(x)
    y_ = math.ceil(y)
    d1 = x-x_
    d2 = y_-y
    w = [(1-d1)*(1-d2),d1*(1-d2),(1-d1)*d2,d1*d2]
    p1 = get_photo(x_,y_)
    p2 = get_photo(x_+1,y_)
    p3 = get_photo(x_,y_-1)
    p4 = get_photo(x_+1,y_-1)
    return Blend_by_weight([p1,p2,p3,p4],normalize(w))

def disparity_matrix(disparity,h,w):
    #(u,v) = (u_,v_) + disparity(bias_x,bias_y)
    x_bias,y_bias = disparity
    White_image = np.zeros((h,w,2),dtype=np.uint16)
    for x in range(w):
        for y in range(h):
            Coord = np.array([x,y])+disparity
            if Coord[1]<0 :
                Coord[1] = 0
            if Coord[0]<0:
                Coord[0] = 0
            if Coord[1]>h-1 :
                Coord[1] = h-1
            if Coord[0]>w-1:
                Coord[0] = w-1
            White_image[y][x] = np.floor(Coord)
    return White_image
#repreduce the image by adjust(u,v) to new (u,v)
def re_image(Matrix,Bias_map,w,h):
    #Image base
    Image = matrix_2_img(Matrix)
    #New image
    White_image = np.zeros((h,w,3),dtype=np.uint8)
    #Based on the new,coords,get the pixels from the orginal photo
    for x in range(w):
        for y in range(h):
            u,v = Bias_map[y][x]
            White_image[y][x] = Image[v][u]
    return White_image
def get_bias(center,point_set,x_bias,y_bias):
    #Based on the center get the bias of all arround points
    Bias_all = [] 
    for i in range(len(point_set)):
        Bias = (np.array(point_set[i])-center)*[x_bias,y_bias]
        Bias_all.append(Bias)
    return Bias_all
def Q_interpolator2(coords,x_bias,y_bias):
    x,y = (coords[0],coords[1])
    x_ = int(x)
    y_ = math.ceil(y)
    Square = [[x_,y_],[x_+1,y_],[x_,y_-1],[x_+1,y_-1]]
    Bias_all = get_bias(coords,Square,x_bias,y_bias)
    #The bais for 4 photo
    d1 = x-x_
    d2 = y_-y
    w = [(1-d1)*(1-d2),d1*(1-d2),(1-d1)*d2,d1*d2]
    P = [get_photo(Square[i][0],Square[i][1]) for i in range(len(Square))]
    #All bias matrix
    Disparity_matrix = [disparity_matrix(B,240,320) for B in Bias_all]
    #All redo image
    Re_images = [re_image(P[i],Disparity_matrix[i],320,240) for i in range(len(Bias_all))]
    #Blend redo image
    R = [img_2_matrix(pp) for pp in Re_images]
    return Blend_by_weight(R,normalize(w))

def aperture_size(coord,r):
    Max = np.array(coord)+r
    Min = np.array(coord)-r
    return print(Max)
def scale_by_z(Z_,image):
    W = 320
    H = 240
    Range = np.array([0.5-Z_/2,0.5+Z_/2])
    y = (H*Range).astype(int) 
    x = (W*Range).astype(int) 
    image = image[y[0]:y[1],x[0]:x[1]]
    I = Image.fromarray(image).resize((320,240))
    return np.array(I)
#Input the Coord of the adperture,get arround
def Get_arround_set(Coord,r):
    #In circle
    Inner = []
    #Out of circle
    Outer = []
    x,y = Coord
    x1y1 = [math.floor(x-r),math.ceil(y+r)]
    x4y4 = [math.ceil(x+r),math.floor(y-r)]
    Up = [[i,x1y1[1]] for i in range(x1y1[0],x4y4[0]+1)]
    Down =  [[i,x4y4[1]] for i in range(x1y1[0],x4y4[0]+1)]
    Left = [[x1y1[0],i+1] for i in range(x4y4[1],x1y1[1]-1)]
    Right = [[x4y4[0],i+1] for i in range(x4y4[1],x1y1[1]-1)]
    Outer = Up+Down+Left+Right
    for i in range(x1y1[0]+1,x4y4[0]):
        Temp =[]
        for j in range(x4y4[1]+1,x1y1[1]):
            Temp.append([i,j])
        Inner = Inner + Temp
    return Inner,Outer
def Get_distence(A,center):
    A = np.array(A)
    return (np.sum((A-center)**2,axis=1))**0.5
#get weight
def make_weight(Inner_distence,Outer_distence,size,lightplus,gauss_scale):
    Out_= size/Outer_distence
    Inner_ = Inner_distence/Inner_distence
    Weight_Out = norm.pdf(Outer_distence,scale=size*gauss_scale)
    Weight_inner = norm.pdf(Inner_distence,scale=size*gauss_scale)
    ALL_weight = np.hstack((Weight_inner*Inner_,Weight_Out*Out_))
    return ALL_weight*lightplus
def make_bias_images(x,y,ALL_set,xbias,ybias):
    P = [get_photo(i[0],i[1]) for i in ALL_set]
    Bias_all = get_bias([x,y],ALL_set,xbias,ybias)
    Disparity_matrix = [disparity_matrix(B,240,320) for B in Bias_all]
    Re_images = [re_image(P[i],Disparity_matrix[i],320,240) for i in range(len(Bias_all))]
    R = [img_2_matrix(pp) for pp in Re_images]
    return R
def get_aperture_image(x,y,size,xbias,ybias,light,gauss_scale):
    #Get the points
    center = [x,y]
    A,B = Get_arround_set([x,y],size)
    Inner_distence = Get_distence(A,center)
    Outer_distence = Get_distence(B,center)
    Weight = make_weight(Inner_distence,Outer_distence,size,light,gauss_scale)
    ALL_set = A+B
    R = make_bias_images(x,y,ALL_set,xbias,ybias)
    W = normalize(Weight)
    return Blend_by_weight(R,W)
Data = []
for i in range(len(Temp)):
    Data.append(img_2_matrix(Temp[i]))
Data = np.array(Data)

root = tk.Tk()


root.title("Assignment1")
Matrix = matrix_2_img(Data[0])
imgtk = ImageTk.PhotoImage(image=Image.fromarray(Matrix))
    
Y = tk.Scale(root, from_=16, to=0 ,length=240,resolution=0.1,\
             orient="vertical",label="Y")
Y.grid(column = 1, row = 1)

W = tk.Label(root,image = imgtk,height = 240,width = 320)
W.grid(column = 2, row = 1)

X = tk.Scale(root, from_=0, to=16, length=320, \
      resolution=0.1,orient="horizontal",label = "X")
X.grid(column = 2, row = 2)
Z = tk.Scale(root, from_=1, to=0, length=320, \
      resolution=0.05,orient="horizontal",label = "Z")
Z.grid(column = 2, row = 3)
D_x = tk.Scale(root, from_=-10, to=10, length=200, \
      resolution=0.1,orient="vertical",label = "D_x")
D_x.grid(column = 3, row = 1)
D_y = tk.Scale(root, from_=-10, to=10, length=200, \
      resolution=0.1,orient="vertical",label = "D_y")
D_y.grid(column = 4, row = 1)
Aperture_size = tk.Scale(root, from_=1, to=5, length=200, \
      resolution=0.1,orient="vertical",label = "Aperture_size")
Aperture_size.grid(column = 5, row = 1)
Lightplus = tk.Scale(root, from_=1, to=2, length=200, \
      resolution=0.1,orient="vertical",label = "lightplus")
Lightplus.grid(column = 6, row = 1)
Scaletimes = tk.Scale(root, from_=0.3, to=3, length=200, \
      resolution=0.1,orient="vertical",label = "scaletimes ")
Scaletimes.grid(column = 7, row = 1)


def Get_image_pinhole():
    y = Y.get()
    x = X.get()
    z = Z.get()
    xbias = D_x.get()
    ybias = D_y.get()
    Matrix = scale_by_z(z,Q_interpolator2([x,y],xbias,ybias))
    imgtk = ImageTk.PhotoImage(image=Image.fromarray(Matrix))
    W.configure(image=imgtk)
    W.image = imgtk

def Get_image_aperture():
    y = Y.get()
    x = X.get()
    z = Z.get()
    xbias = D_x.get()
    ybias = D_y.get()
    size = Aperture_size.get()
    light = Lightplus.get()
    gauss_scale = Scaletimes.get()
    Matrix = scale_by_z(z,get_aperture_image(x,y,size,xbias,ybias,light,gauss_scale))
    imgtk = ImageTk.PhotoImage(image=Image.fromarray(Matrix))
    W.configure(image=imgtk)
    W.image = imgtk



root.bind("<Return>", Get_image_pinhole)
root.bind("<Return>", Get_image_aperture)

Pinhole = tk.Button(root,text="Make Pinhole Image",command=Get_image_pinhole).grid(column = 3,row = 3)
Aperture = tk.Button(root,text="Make Aperture Image",command=Get_image_aperture).grid(column=3,row = 2)

root.mainloop()

