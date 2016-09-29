# encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.animation as animation

class AnimatedScatterCluster(object):

    def __init__(self,df,frameSta=None,frameEnd=None,interval=200,x='x',y='y'): ## specify frameID col in DF
        self.df = df
        self.x = x
        self.y = y
        self.fig, self.ax = plt.subplots()
        self.frameSta = df.frameID[0] if frameSta is None else frameSta
        self.frameEnd = df.frameID[len(df.frameID)-1] if frameEnd is None else frameEnd
        self.ani = animation.FuncAnimation(self.fig, self.update, interval = interval,
                                           init_func=self.setup_plot,frames=self.frameEnd-self.frameSta+1,blit=True,repeat=False)

    def setup_plot(self):

        self.xlim = [self.df[self.x].min(),self.df[self.x].max()]
        self.ylim = [self.df[self.y].min(),self.df[self.y].max()]
        self.text = self.ax.text(self.xlim[0]+0.3,self.ylim[1]-0.3,"")
        self.scat = self.ax.scatter([],[], c=[],s=20, animated=True,alpha=0.9)
        self.ax.axis(self.xlim+self.ylim)
        self.scat.set_clim(0,100)
        #         plt.colorbar(self.scat)
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,self.text

    def update(self, i):
        ii = i + self.frameSta
          ## timeSeg may not start from 0
        data = self.df.loc[self.df.frameID==i,]
        nclu = len(set(data.cluID)) - (1 if -1 in list(data.cluID) else 0)
        text = "FrameID: "+str(ii)+" "+"    Cluster Num: "+str(nclu)
        self.text = self.ax.text(self.xlim[0]+0.1,self.ylim[1]-0.1,text)
        self.scat.set_clim(0,100)
        colors = list(np.linspace(0,100,nclu+2))
        colors = np.array(colors[2:]+[0])
        c = np.array(colors[data.cluID])
        s = np.where(data.if_core,80,10) ## 核心点大小为14，非核心点为10
        self.scat.set_array(c)
        self.scat._sizes = s
        self.scat.set_offsets(np.array(data.loc[:,[self.x,self.y]]))
        return self.scat,self.text

    def show(self):
        plt.show()


if __name__ == '__main__':
    import os
    os.chdir(r"F:\gitInStormor\thunderstorm\data")
    df = pd.read_csv("clusterData.csv")
    a = AnimatedScatterCluster(df,0,300,interval=200)
    a.show()
    # mywriter = animation.FFMpegWriter()
    # a.ani.save('storm1.mp4',writer=mywriter)