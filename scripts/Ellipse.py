import numpy as np
import matplotlib.pylab as plt

def fit_ellipse(data): # data is m*2 format
    covariance = np.cov(data.T)
    eigval, eigvec = np.linalg.eig(covariance)
    minIdx, maxIdx = (0,1) if(eigval[0]<eigval[1]) else (1,0)
    minVal, maxVal = eigval[minIdx],eigval[maxIdx]
    minVec, maxVec = eigvec[:,minIdx],eigvec[:,maxIdx]
    angle = np.arctan2(maxVec[1],maxVec[0])
    angle = angle+2*np.pi if(angle<0) else angle
    x0,y0 = np.mean(data,axis=0)
    chisquare_val = 2.4477
    a = chisquare_val*np.sqrt(maxVal)
    b = chisquare_val*np.sqrt(minVal)
    phi = angle
    return x0,y0,a,b,phi


def calSize(ellipse): # x0,y0,a,b,phi
    return 1000*np.pi*ellipse[2]*ellipse[3]


def plot_ellipse(x0,y0,a,b,phi,figNew=False,ax=None,**kargs): # if figNew=False,should specify ax
    theta_grid = np.linspace(0,2*np.pi) #num=50
    ellipse_x_r = a*np.cos(theta_grid)
    ellipse_y_r = b*np.sin(theta_grid)
    R = np.array([[np.cos(phi),np.sin(phi)],[-np.sin(phi),np.cos(phi)]])
    r_ellipse = np.dot(np.array([ellipse_x_r,ellipse_y_r]).T,R)
    if figNew:
        fig,ax = plt.subplots()
        ax.plot(r_ellipse[:,0]+x0,r_ellipse[:,1]+y0,**kargs)
    else:
        ax.plot(r_ellipse[:,0]+x0,r_ellipse[:,1]+y0,**kargs)

def containInEllipse(x,y,x0,y0,a,b,phi):
    # x,y can be either np.array or scalar
    ox = x - x0
    oy = y - y0
    rotx = ox * np.cos(phi) + oy * np.sin(phi)
    roty = -ox * np.sin(phi) + oy * np.cos(phi)
    dist_x = rotx / a
    dist_y = roty / b
    return (np.hypot(dist_x,dist_y)<=1)