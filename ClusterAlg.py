#!/usr/bin/env python
# from types import TupleType
'''
This module implement a typical K-means Clustering. The original author is Jing Wang (hbhzwj@gmail.com);
cf. https://github.com/hbhzwj/GAD/blob/eadf9fe5c8749bb7c091c41a6ab0e8488a7f8619/gad/Detector/ClusterAlg.py
'''
from __future__ import print_function, division, absolute_import
from random import sample

try:
    import Queue as queue # replace with 'import queue' if using Python 3
except ImportError:
    import queue
    
## {{{ http://code.activestate.com/recipes/577661/ (r2)
# from sadit.util import queue
class MedianHeap:
    def __init__(self, numbers = None):
        self.median = None
        self.left = queue.PriorityQueue() # max-heap
        self.right = queue.PriorityQueue() # min-heap
        self.diff = 0 # difference in size between left and right

        if numbers:
            for n in numbers:
                self.put(n)

    def top(self):
        return self.median

    def put(self, n):
        if not self.median:
            self.median = n
        elif n <= self.median:
            self.left.put(-n)
            self.diff += 1
        else:
            self.right.put(n)
            self.diff -= 1

        if self.diff > 1:
            self.right.put(self.median)
            self.median = -self.left.get()
            self.diff = 0
        elif self.diff < -1:
            self.left.put(-self.median)
            self.median = self.right.get()
            self.diff = 0

    def get(self):
        median = self.median

        if self.diff > 0:
            self.median = -self.left.get()
            self.diff -= 1
        elif self.diff < 0:
            self.median = self.right.get()
            self.diff += 1
        elif not self.left.empty():
            self.median = -self.left.get()
            self.diff -= 1
        elif not self.right.empty():
            self.median = self.right.get()
            self.diff += 1
        else: # median was the only element
            self.median = None

        return median

try:
    import numpy as np
    NUMPY = True
except:
    NUMPY = False

def GetMedian(numbers):
    if NUMPY:
        return float(np.median(numbers))
    else:
        m = MedianHeap(numbers)
        return m.get()

def ReClustering(data, centerPt, DF):
    cluster = []
    for pt in data:
        dv = []
        for c in centerPt:
            d = DF(pt, c)
            dv.append(d)
        md = min(dv)
        cluster.append(dv.index(md))
    return cluster

def Add(x, y):
    res = []
    for i in xrange(len(x)):
        res.append(x[i]+y[i])

    return res

def NDivide(cSum, cLen):
    res = []
    for i in xrange(len(cSum)):
        x = []
        for j in xrange(len(cSum[0])):
            x.append( cSum[i][j] * 1.0 / cLen[i] )

        res.append(x)
    return res

def CalClusteringCenterKMeans(data, cluster):
    ucLen = max(cluster) + 1
    cSum = [ [0] * len(data[0]) for i in xrange(ucLen)]
    cLen = [0] * ucLen
    i = -1
    for ele in data:
        i += 1
        cl = cluster[i]
        cSum[cl] = Add(cSum[cl], ele)
        cLen[cl] += 1

    try:
        cAve = NDivide(cSum, cLen)
    except:
        import pdb; pdb.set_trace()
    return cAve

def CalClusteringCenterKMedians(data, cluster):
    ucLen = max(cluster) + 1
    cMed = []
    for cl in xrange(ucLen):
        clusterData = [d for i, d in zip(cluster, data) if i == cl]
        med = [GetMedian(vals) for vals in zip(*clusterData)]
        cMed.append(med)
    return cMed

def Equal(x, y):
    if len(x) != len(y):
        return False
    for i in range(len(x)):
        if x[i] != y[i]:
            return False
    return True

def KMeans(data, k, distFunc):
# def KMedians(data, k, distFunc):
    print('start KMeans...')
    # Initialize
    centerPt = sample(data, k)
    while True:
        cluster = ReClustering(data, centerPt, distFunc)
        NewCenterPt = CalClusteringCenterKMeans(data, cluster)
        if Equal(NewCenterPt, centerPt):
            break
        centerPt = NewCenterPt

    return cluster, centerPt

def KMedians_(data, k, distFunc):
# def KMeans(data, k, distFunc):
    print('start KMedians ...')
    # Initialize
    centerPt = sample(data, k)
    while True:
        cluster = ReClustering(data, centerPt, distFunc)
        NewCenterPt = CalClusteringCenterKMedians(data, cluster)
        if Equal(NewCenterPt, centerPt):
            break
        centerPt = NewCenterPt
    distToClusterCenter = []
    i = 0
    for pt in data:
        distToClusterCenter.append(distFunc(pt, centerPt[cluster[i]]))
        i += 1
    optDist = sum(distToClusterCenter)
    return cluster, distToClusterCenter, optDist

def KMedians(data, k, n, distFunc):
    '''
    To determine a relatively stable clustering result, we need to run the algorithm several times, 
    and n is the number of running times. 
    ----
    Output also the distance to cluster centers
    '''
    cluster_list = []
    distToClusterCenter_list = []
    optDist_list = []
    
    for i in range(n):
        results = KMedians_(data, k, distFunc)
        cluster_list.append(results[0])
        distToClusterCenter_list.append(results[1])
        optDist_list.append(results[2])
    idx = optDist_list.index(min(optDist_list))
    return cluster_list[idx], distToClusterCenter_list[idx]

##### Define the distance between Jam A and Jam B; added by Jing Zhang (jingzbu@gmail.com)

from geopy.distance import vincenty

def DF(A_, B_):
    '''
    The argument A_ and B_ should be two lists
    '''
    A_ = np.array(A_)
    B_ = np.array(B_)
    A = np.reshape(A_, (int(np.size(A_) / 2), 2))
    B = np.reshape(B_, (int(np.size(B_) / 2), 2))
    len_A = np.size(A, 0)
    len_B = np.size(B, 0)

    dist_AB_ = []
    for i in range(len_A):
        for j in range(len_B):
            jam1 = (A[i,1], A[i,0])
            jam2 = (B[j,1], B[j,0])
            dist_AB_.append(vincenty(jam1, jam2).meters)
    if (len(dist_AB_) > 0):
        dist_AB = min(dist_AB_)
    else:
        dist_AB = 1e8
    return dist_AB

def test():
#     data = [(5, 6, 8, 9),
#             (10, 0, 4, 1),
#             (10, 0, 2, 1),
#             (10, 0, 5, 1),
#             (10, 20, 30, 1),
#             (10, 20, 30, 2),
#             (10, 20, 30, 3),
#             (10, 20, 30, 4),
#             (1, 1, 1, 1),
#             (10, 20, 30, 6),
#             (10, 20, 30, 7),
#             (10, 20, 30, 8),
#             (10, 200, 1, 1),
#             (10, 20, 30, 9),
#             (10, 0, 3, 1),
#             (2, 3, 5, 6),
#             (10, 20, 30, 5)
#             ]
##### Modified by Jing Zhang (jingzbu@gmail.com)
    AA = np.array([[-71.09702, 42.344783],
      [-71.095651, 42.345145],
      [-71.095584, 42.345162],
      [-71.095326, 42.345231],
      [-71.095052, 42.345308],
      [-71.094946, 42.345339],
      [-71.094644, 42.345437],
      [-71.094461, 42.345497],
      [-71.094333, 42.345564],
      [-71.09428, 42.345602],
      [-71.093212, 42.346572],
      [-71.09309, 42.346661],
      [-71.092813, 42.346817]])

    BB = np.array([[-71.093802, 42.367956],
      [-71.093229, 42.367484],
      [-71.092506, 42.366904],
      [-71.091803, 42.366315]])
    
    CC = np.array([[-71.19702, 42.344783],
      [-71.195651, 42.345145],
      [-71.195584, 42.345162],
      [-71.195326, 42.345231],
      [-71.195052, 42.345308],
      [-71.194946, 42.345339]])
    A_ = list(np.reshape(AA, np.size(AA)))
    B_ = list(np.reshape(BB, np.size(BB)))
    C_ = list(np.reshape(CC, np.size(CC)))
    data = [A_, A_, B_, B_, B_, C_, A_, B_, A_, B_, B_, A_, A_, C_, C_]
#     print(data)
    k = 3
    n = 10
    
#     DF = lambda x,y:abs(x[0]-y[0]) * (256**3) + abs(x[1]-y[1]) * (256 **2) + abs(x[2]-y[2]) * 256 + abs(x[3]-y[3])
    # DF = lambda x,y:abs(x[0]-y[0]) * 2 + abs(x[1]-y[1]) * 2 + abs(x[2]-y[2]) * 2+ abs(x[3]-y[3]) * 2
    # DF = lambda x,y: ((x[0]-y[0]) ** 2) * (256 ** 3) + ((x[1]-y[1]) ** 2) * (256 **2) + ((x[2]-y[2]) ** 2) * (256)+ (x[3]-y[3]) ** 2
    # DF = lambda x,y:(x[0]-y[0]) ** 2  + (x[1]-y[1]) ** 2  + (x[2]-y[2]) ** 2 + (x[3]-y[3]) ** 2

    cluster, distToClusterCenter = KMedians(data, k, n, DF)
    print('cluster, ', cluster)
    print('distToClusterCenter, ', distToClusterCenter)
