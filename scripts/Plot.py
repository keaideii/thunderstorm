# encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import hickle

def plotTracks(tracks,MP,SP,figsize=[15,10]):
    ## tracks dataFrame：trackID/x/y/size  MP:merge point pair  SP:split point pair
    minN,maxN = np.min(tracks.frameID),np.max(tracks.frameID)
    tracksN = len(np.unique(tracks.trackID))
    grouped = tracks.groupby(tracks.trackID)
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



## plot
if __name__=="__main__":
    dir = r"F:\gitInStormor\thunderstorm\data"
    os.chdir(dir)
    tracks = pd.read_csv("tracks.csv")
    MP = hickle.load('mergeCell.hkl')
    SP = hickle.load('splitCell.hkl')
    plotTracks(tracks,MP,SP)