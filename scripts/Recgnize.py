# encoding=utf-8

import pandas as pd
from pandas import Series
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from datetime import datetime,timedelta
from ellipse import *
from pandas import DataFrame


class Recognize(object):
    def __init__(self,dataDir,timeSpan=10,eps=0.1,min_samples=50):
        self.data0,self.timeLen = Recognize.preProcess(dataDir,timeSpan) #df,timeSeq,timeLen
        self.data_tran = StandardScaler().fit_transform(np.array(self.data0[["x","y"]]))
        self.span = timeSpan
        self.eps = eps
        self.min_samples = min_samples
        self.clusterData = DataFrame() #"tIdx":[data(exclude -1 label),label]
        self.center = [] # x,y,size,clusterID,frameID
        self.ellipse = [] # a is the longer axis and b is the shorter

    def seqCluster(self):
        tl = self.timeLen
        ts = list(self.data0["timeSeg"])
        for i in range(tl):
            idx = [ii for ii,seg in enumerate(ts) if seg==i] #generate idx
            data = np.array(self.data_tran[idx,:])
            data_temp = self.data0.loc[idx,["x","y"]]
            if(len(data) > 1.0*self.min_samples):
                db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(data)
                labels = db.labels_
                core_label = labels[labels!=-1] # store
                core_samples = np.zeros_like(labels,dtype=bool)
                core_samples[db.core_sample_indices_] = True
                data_ori = data_temp.loc[labels!=-1,:] # store
                grouped = data_ori.groupby(core_label)
                xy = grouped.mean()
                data_temp['cluID'] = labels ## also include -1 clu
                data_temp['if_core'] = core_samples
                data_temp['frameID'] = i
                ell = grouped.apply(lambda x:fit_ellipse(np.array(x)))
                for cluIdx in xy.index:
                    self.center.append([xy.iloc[cluIdx,0],xy.iloc[cluIdx,1],calSize(ell[cluIdx]),cluIdx,i])  # x,y,size,cluID,frameID
                    self.ellipse.append(list(ell[cluIdx])+[cluIdx,i])
                    self.clusterData = pd.concat([self.clusterData,data_temp])
            else:
                self.center.append([-1,-1,-1,-1,i])  # x,y,size,ID,frameID
                self.ellipse.append([-1,-1,-1,-1,-1,-1,i])
                data_temp['cluID'] = -1
                data_temp['if_core'] = False
                data_temp['frameID'] = i
                self.clusterData = pd.concat([self.clusterData,data_temp])


    def saveData(self,dir):
        DataFrame(self.center,columns=['x0','y0','size','cluID','frameID']).to_csv(dir+"center.csv",index=False)
        DataFrame(self.ellipse,columns=['x0','y0','a','b','phi','cluID','frameID']).to_csv(dir+"ellipse.csv",index=False)
        DataFrame(self.clusterData).to_csv(dir+"clusterData.csv",index=False,columns=None)


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
    # dir = r'C:\Users\76999\Desktop\StormCloud\\'
    dir = r'C:\Users\Administrator\desktop\StormCloud\\'
    # os.chdir(r'C:\Users\Administrator\desktop\StormCloud')
    a = Recognize(dir+"nanrui_root.csv",10)
    a.seqCluster()
    # dir2 = r"D:\workspace\idea\thunderstorm\data\\"
    dir2 = r'F:\gitInStormor\thunderstorm\data\\'
    a.saveData(dir2)



