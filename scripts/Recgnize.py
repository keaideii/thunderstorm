# encoding=utf-8

import numpy as np
import pandas as pd
from pandas import Series
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from datetime import datetime,timedelta
from Ellipse import Ellipse
import os

centerDtype=[("x","f4"),("y","f4")]

class Recognize(object):
    def __init__(self,dataDir,timeSpan=10,eps=0.3,min_samples=5):
        self.data0,self.timeLen = Recognize.preProcess(dataDir,timeSpan) #df,timeSeq,timeLen
        self.data_tran = StandardScaler().fit_transform(np.array(self.data0[["x","y"]]))
        self.span = timeSpan
        self.eps = eps
        self.min_samples = min_samples
        self.clusterData = np.array([np.nan]*4) #"tIdx":[data(exclude -1 label),label]
        self.center = np.array([np.nan]*5) # x,y,size,ID,frameID
        self.ellipse = np.array([np.nan]*7) # a is the longer axis and b is the shorter

    def seqCluster(self):
        tl = self.timeLen
        ts = list(self.data0["timeSeg"])
        for i in range(tl):
            idx = [ii for ii,seg in enumerate(ts) if seg==i] #generate idx
            data = np.array(self.data_tran[idx,:])
            if(len(data) > 1.5*self.min_samples):
                center = [] # x,y,size,ID,frameID
                ellipse = []
                cluData = []
                db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(data)
                labels = db.labels_
                core_label = labels[labels!=-1] # store
                data_ori = self.data0.loc[idx,["x","y"]].loc[labels!=-1,:] # store
                grouped = data_ori.groupby(core_label)
                xy = grouped.mean()
                ell = grouped.apply(lambda x:Ellipse.fit_ellipse(np.array(x)))
                for i in xy.index:
                    center.append([xy.iloc[i,0],xy.iloc[i,1],Ellipse.calSize(ell[i])])
                    ellipse.append(list(ell[i]))
                    cluData.append(np.array(data_ori.loc[core_label==i]))
                    self.test = np.vstack((self.test,np.array(data_ori)))
            else:
                center,ellipse,cluData = 0,0,0
            self.clusterData.append(cluData)
            self.center.append(center)
            self.ellipse.append(ellipse)

    def saveData(self):
        import hickle as hkl
        # hkl.dump(self.clusterData,"clusterData.hkl")
        # hkl.dump(self.center,"center.hkl")
        # hkl.dump(self.ellipse,"ellipse.hkl")
        hkl.dump(self.test,"test.hkl")

    @staticmethod
    def preProcess(dir,span):

        def genTimeSeg(time,idx):
            if idx == timeLen-1: return timeLen-1
            if time>= timeSeq[idx] and time<timeSeq[idx+1]:
                return idx
            else:
                return genTimeSeg(time,idx+1)

        def genTimeSeq(df): # unit is minite
            first = df['time'][0]
            last = df['time'][len(df.time)-1]
            delta = timedelta(minutes=span)
            time = first
            timeSeq = [time]
            while timeSeq[-1] < last:  ## 5min 共818个时间段  10min 共410个时间段
                time += delta
                timeSeq.append(time)
            return timeSeq

        df =  pd.read_csv(dir,header=None,names=['time','y','x','cur'])
        format = '%Y/%m/%d %H:%M'
        df['time'] = df[['time']].applymap(lambda x:datetime.strptime(x,format))
        # df['strength'] = np.abs(df.cur)
        timeSeq = genTimeSeq(df) # gen time seq to seperate time col
        timeLen = len(timeSeq)
        df['timeSeg'] = Series([genTimeSeg(t,0) for t in df.time]) # reflect time to different tag
        df = df.drop(["time","cur"],axis=1)
        return df,timeLen





############################################################
if __name__=="__main__":
    dir = r'C:\Users\Administrator\desktop\StormCloud\nanrui_root.csv'
    os.chdir(r'C:\Users\Administrator\desktop\StormCloud')
    a = Recognize(dir,10)
    a.seqCluster()
    print a.test
    a.saveData()


