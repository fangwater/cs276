import random
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes
from matplotlib.font_manager import FontProperties
 
def draw():
    data = np.zeros((6,6))
    #B_ = C
    for b in range(len(B)):
        data[B[b][0]][B[b][1]]= B_[b]
    for a in range(len(A)):
        data[A[a][0]][A[a][1]]= A_[a]
    xLabel = ['0','1','2','3','4',"5"]
    yLabel = ['0','1','2','3','4',"5"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yticks(range(len(yLabel)))
    ax.set_yticklabels(yLabel)
    ax.set_xticks(range(len(xLabel)))
    ax.set_xticklabels(xLabel)
    im = ax.imshow(data, cmap=plt.cm.hot_r)
    plt.colorbar(im)
    plt.title("Weight in Aperture Without Guassion((s,t)=(2.1,2.6),size=2)")
    #show
    plt.show()
d = draw()