# encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import hickle
from ellipse import plot_ellipse

def plotTracks(tracks,MP,SP,figsize=[15,10]):
    ## tracks dataFrame：trackID/x/y/size  MP:merge point pair  SP:split point pair
    minN,maxN = np.min(tracks.frameID),np.max(tracks.frameID)
    tracksN = len(np.unique(tracks.trackID))
    c = tracks.frameID - min(tracks.frameID)
    s = tracks['size']
    fig,ax = plt.subplots(figsize=figsize)
    ax.scatter(tracks.x,tracks.y,c=c,s=s*2)
    grouped = tracks.groupby(tracks.trackID)
    grouped.apply(lambda x:ax.plot(x.x,x.y))
    # fig,ax = plt.subplots(figsize=[15,10])
    for i in MP:
        ax.plot(i[0],i[1],'r--')
    for i in SP:
        ax.plot(i[0],i[1],'b--')
    ax.set_title("FrameID: "+str(minN)+"~"+str(maxN)+"   Track Num: "+str(tracksN))
    plt.show()

def plotEllipseOnCluster(ellipse,clusterData,minFID,maxFID,figsize=[15,15]): ## center,ellipse,clusterData are DFs
    ell = ellipse.loc[(ellipse.frameID>=minFID)&(ellipse.frameID<=maxFID),:]
    clu = clusterData.loc[(clusterData.frameID>=minFID)&(clusterData.frameID<=maxFID),:]
    minN,maxN = np.min(ell.frameID),np.max(ell.frameID)
    psize1 = np.where(clu.cluID!=-1,20,5)
    psize2 = np.where(clu.if_core,30,0)
    psize = psize1+psize2
    # clu.plot(x='x',y='y',kind='scatter',c=clu.cluID,size=pSize,figsize=[15,10])
    fig,ax = plt.subplots(figsize=figsize)
    ax.scatter(clu.x,clu.y,s=psize,c=clu.cluID)
    ellN = len(ell)
    for i in range(ellN):
        if ell.iloc[i,0]!=-1:
            plot_ellipse(*ell.iloc[i,0:5],linestyle='-',linewidth=3,ax=ax)
    ax.grid(True)
    ax.set_title("FrameID: "+str(minN)+"~"+str(maxN))
    plt.show()



## plot
if __name__=="__main__":
    dir = r"F:\gitInStormor\thunderstorm\data"
    os.chdir(dir)
    tracks = pd.read_csv("tracks.csv")
    MP = hickle.load('mergeCell.hkl')
    SP = hickle.load('splitCell.hkl')
    # plotTracks(tracks,MP,SP)

    ## ellipse plot
    center = pd.read_csv('center.csv')
    clusterData = pd.read_csv("clusterData.csv")
    ellipse = pd.read_csv("ellipse.csv")
    plotEllipseOnCluster(center,ellipse,clusterData,15,40)