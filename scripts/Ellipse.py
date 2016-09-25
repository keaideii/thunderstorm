import numpy as np

class Ellipse(object):

    @staticmethod
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

    @staticmethod
    def calSize(elli):
        return 1000*np.pi*elli[2]*elli[3]