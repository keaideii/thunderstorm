# encoding=utf-8

from Recgnize import *
import os
from Hungarian import *


storm_dtype = [('x','f4'),('y','f4'),('size','f4'),('a','f4'),('b','f4'),('phi','f4'),('cluID','i4'),('frameID','i4'),('trackID','i4'),('f_x','f4'),('f_y','f4'),('f_size','f4')]
len_storm_dtype = len(storm_dtype)
end_storm = np.array([tuple([-1]*len_storm_dtype)],dtype=storm_dtype)[0]

class Track(object):
    ## profit fun C uses Dist only
    ## c_thres
    def __init__(self,c_thres=0.3,xy_weight=0.995,backN=3):

        self.c_thres = c_thres
        self.backN = backN
        self.xy_weight = xy_weight
        self.tracks = []
        self.currStorms = [] # x,y,size,cluID,frameID,ellipse,trackID,  ## now
        self.prevStorms = []  # a Storm is labeled by (cluID & frameID)  ## last # x,y,size,cluID,frameID,ellipse,trackID ## 直接根据
        self.merge = [[]] #(from,to)
        self.split = [[]] #(from,to)
        self.mergeCell = [[]]
        self.splitCell = [[]]

    def track_step(self,stormCells): ## stormCells is np.array with dtype
        if(not self.prevStorms):
            for i in range(len(stormCells)):
                # stormCells[i] = np.insert(stormCells[i],len(stormCells[i]),i)
                stormCells[i]['trackID'] = i
                self.tracks.append(np.array([stormCells[i]]))
            self.currStorms.append(stormCells)
            self.prevStorms.append(stormCells) #[np.array(),]

            return
        else:
            self.currStorms.append(stormCells)
            prevStorms = self.prevStorms[-1]
            currStorms = self.currStorms[-1]
            nPrev = len(prevStorms)
            nCurr = len(currStorms)
            C = self.cal_cost(prevStorms,currStorms) #cost function
            # matching
            matchPair0 = hungarian(C)
            # print matchPair0
            matchPair = self._rejectLongMatch(matchPair0,C)
            # print matchPair
            keep,end,start = self.handle_match(matchPair,nPrev,nCurr,prevStorms,currStorms) # dict can be changed through function
            self._merge(keep,end,currStorms,prevStorms)
            self._split(keep,end,start,currStorms,prevStorms)
            self._update(keep,end,start,currStorms) #更新tracks
            # print C
            self.prevStorms.append(currStorms)

    def _rejectLongMatch(self,matchPair,C):
        return [[i,j] for i,j in matchPair if C[i][j]<self.c_thres]

    def _update(self,keep,end,start,currStorms):
        ## add zero_point to end track
        for i,trackID in end.items():
            self.tracks[trackID] = np.hstack((self.tracks[trackID],end_storm))
        for i,trackID in keep.items(): ## only need to update keep tracks's preds
            currStorms[i]['trackID'] = trackID
            self.tracks[trackID] = np.hstack((self.tracks[trackID],currStorms[i]))
            track = self.tracks[trackID]
            preds = self.predict_cells(track)
            self.tracks[trackID][-1]['f_x'] = preds[0]
            self.tracks[trackID][-1]['f_y'] = preds[1]
            self.tracks[trackID][-1]['f_size'] = preds[2]
            currStorms[i]['f_x'] = preds[0]
            currStorms[i]['f_y'] = preds[1]
            currStorms[i]['f_size'] = preds[2]

        # for i,_ in start.items():
        #     trackID = len(self.tracks)
        #     currStorms[i]['trackID'] = trackID
        #     self.tracks.append(np.array([currStorms[i]])) ## !!!! np.array([tuple()])

    # def _updatePredicts(self,currStorms): ## update f_x,f_y,f_size
    #     for i,cell in enumerate(currStorms):
    #         track = self.tracks[cell['trackID']]
    #         preds = self.predict_cells(track)
    #         currStorms['f_x'][i] = preds[0]
    #         currStorms['f_y'][i] = preds[1]
    #         currStorms['f_size'][i] = preds[2]



    def _split(self,keep,end,start,currStorms,prevStorms):
        ## 上一时刻keep的trackID到这一时刻的start
        split = []
        splitCell = []
        fromIDs = prevStorms['trackID'] # prev trackIDs
        ell = prevStorms[['a','b','phi','size']]
        predictxys = [self.predict_cells(self.tracks[i]) for i in fromIDs] #preX,preY,preSize
        predictEll = self.predict_ell(ell,predictxys)
        startCells = [currStorms[["x","y"]][i] for i in start.keys()]
        startTracks = [id for id in start.values()]
        for i,xy in enumerate(startCells): #def containInEllipse(x,y,x0,y0,a,b,phi):
            matchIdx = [idx for idx,preEll in enumerate(predictEll) if containInEllipse(xy[0],xy[1],*preEll)]
            if len(matchIdx)>0:
                matchCellxy = np.array([[predictxys[idx][0],predictxys[idx][1]] for idx in matchIdx])
                dist = np.hypot(matchCellxy[:,0]-xy[0],matchCellxy[:,1]-xy[1])
                minIdx = dist.argmin()
                fromID = fromIDs[matchIdx[minIdx]]
                fromCell = prevStorms[['x','y']][matchIdx[minIdx]]
                toID = startTracks[i]
                toCell = startCells[i]
                split.append([fromID,toID])
                splitCell.append([fromCell,toCell])
        self.split.append(split)
        self.splitCell.append(splitCell)


    def predict_ell(self,ell,predictxys):
        pre_sizes = np.array([i[2] if i[2]>0 else 0 for i in predictxys])
        pre_xy = [[i[0],i[1]] for i in predictxys]
        ori_size = ell['size']
        ratio = np.sqrt(pre_sizes/ori_size)
        preAB = [[r*i for i in e] for e in ell[['a','b']] for r in ratio]
        zipRes = zip(pre_xy,preAB,ell['phi'])
        return [i[0]+i[1]+[i[2]] for i in zipRes] #[[],[]]


    def _merge(self,keep,end,currStorms,prevStorms):
        ellParams = ["x","y","a","b","phi"]
        merge = []
        mergeCell = []
        for idx,fromID in end.items():  # trackID of end ## 注意此处不能写i 被后面循环的i遮盖
            track = self.tracks[fromID]
            predictxys = self.predict_cells(track)
            preX = predictxys[0]
            preY = predictxys[1]
            ellipses = [currStorms[ellParams][trackID] for trackID in keep.keys()] # trackID of start the last cell in the track ##先筛选变量名 再选择idx
            inEllIdx = [i for i,ell in enumerate(ellipses) if containInEllipse(preX,preY,*ell)] # which ell contains predict xy ##注意列表推导式内的i不能与外层idx相同
            if len(inEllIdx)>0:
                fromCell = prevStorms[['x','y']][idx]
                xy = np.array([currStorms[["x","y"]][keep.keys()[idx]] for idx in inEllIdx])# find the min dist ell
                dists = np.hypot(xy["x"]-preX,xy["y"]-preY)
                toID = keep[keep.keys()[inEllIdx[dists.argmin()]]]
                toCell = currStorms[['x','y']][keep.keys()[inEllIdx[dists.argmin()]]]
                merge.append([fromID,toID]) # (from,to)
                mergeCell.append([fromCell,toCell])
        self.merge.append(merge)
        self.mergeCell.append(mergeCell)



    def predict_cells(self,track,deltaT=1):
        params = ["x","y","size"]
        seqs = track[params][-self.backN:]
        if len(seqs)>1:
            At = []
            Bt = []
            for param in params:
                seq = seqs[param]
                at,bt = self._exp_smooth(seq)
                At.append(at)
                Bt.append(bt)
            res = np.array(At)+np.array(Bt)*deltaT
        else:
            res = seqs[-1]
        return res

    @staticmethod
    def _exp_smooth(seq,alpha=0.9):
        # seq should be longer than 1
        s1 = [seq[0],alpha*seq[1]+(1-alpha)*seq[0]]
        s2 = [s1[-1]]
        ll = len(seq)
        for i in range(2,ll):
            s1.append(alpha*seq[i]+(1-alpha)*s1[-1])
            s2.append(alpha*s1[-1]+(1-alpha)*s2[-1])
        at = 2*s1[-1] - s2[-1]
        bt = alpha/(1-alpha)*(s1[-1]-s2[-1])
        return at,bt


    def handle_match(self,matchPair,nPrev,nCurr,prevStorms,currStorms):
        keep = {}
        end = {}
        start = {}
        nMatch = len(matchPair)
        matchedPrev = [i for i,j in matchPair]
        matchedCurr = [j for i,j in matchPair]
        for pre,cur in matchPair:
            keep[cur] = prevStorms[pre]['trackID']
        if(nPrev > nMatch): # more prev cells
            unmatchedPrev =  [i for i in range(nPrev) if i not in matchedPrev]
            for i in unmatchedPrev:
                end[i] = prevStorms[i]['trackID']
        if(nCurr > nMatch): # more curr cells
            unmatchedCurr = [i for i in range(nCurr) if i not in matchedCurr]
            for i in unmatchedCurr:
                trackID = len(self.tracks)
                currStorms[i]['trackID'] = trackID
                self.tracks.append(np.array([currStorms[i]]))## !!!! np.array([tuple()])
                start[i] = trackID # update the trackID once
        return keep,end,start



    def cal_cost(self,prev,curr):
        xloc0 = prev['x']
        xloc1 = curr['x']
        yloc0 = prev['y']
        yloc1 = curr['y']
        size0 = prev['size']
        size1 = curr['size']
        # print "size0",size0
        # print "size1",size1
        xy = self.xy_weight*np.hypot(xloc0[:,np.newaxis]-xloc1[np.newaxis,:], yloc0[:,np.newaxis]-yloc1[np.newaxis,:])
        size = (1-self.xy_weight)*np.abs(size0[:,np.newaxis] - size1[np.newaxis,:])
        # print "xy",xy
        # print "size",size
        # print "total",xy+size
        return xy+size


    @staticmethod
    def con2Tuple(cen,ell):
        temp = [tuple(list(cen[i])+list(ell[i])+[-1,]+list(cen[i])) for i in range(len(cen))] #initial trackID is -1 the initial predicts is their x,y,size
        res = np.array(temp,dtype=storm_dtype)
        return res

    def save(self):
        res = DataFrame(self.flatTracks())
        res = res.loc[res.x!=-1,:]
        res.to_csv("tracks.csv",index=False)

    def flatTracks(self):
        return np.array([i for track in self.tracks for i in track])


    def flat_cell(self,cell,file):
        import hickle
        res = [[[j[0][0],j[1][0]],[j[0][1],j[1][1]]] for i in cell if len(i)>0 for j in i]
        hickle.dump(res,file)
        return res




if __name__=="__main__":
    os.chdir(r"F:\gitInStormor\thunderstorm\data")
    center = pd.read_csv('center.csv')
    ellipse = pd.read_csv("ellipse.csv")
    center.columns = ['x0','y0','size','cluID','frameID']
    ellipse.columns = ['x0','y0','a','b','phi','cluID','frameID']
    staId = 47
    endId = 70
    center = center.loc[(center.frameID>=staId)&(center.frameID<=endId),:]
    ellipse = ellipse.loc[(ellipse.frameID>=staId)&(ellipse.frameID<=endId),:]
    t = Track()
    frames = range(staId,endId+1)
    for i in frames:
        cen = np.array(center.loc[center.frameID==i,['x0','y0','size']])
        ell = np.array(ellipse.loc[ellipse.frameID==i,['a','b','phi','cluID','frameID']])
        corner = t.con2Tuple(cen,ell)
        t.track_step(corner)
    # print t.merge
    # print t.split
    t.save()
    t.flat_cell(t.mergeCell,'mergeCell.hkl')
    t.flat_cell(t.splitCell,'splitCell.hkl')
    print t.merge
    print t.mergeCell
    print t.split
    print t.splitCell










