from util import *

data_folder = '/home/jzh/Waze/'

import json

# Loading JSON data
with open(data_folder + 'jam_ref_data.json', 'r') as json_file:
    jam_ref_data = json.load(json_file)
    
with open(data_folder + 'jam_test_data.json', 'r') as json_file:
    jam_test_data = json.load(json_file)

### Quantize the features by k-means clustering 

import numpy as np

loc_list = []
length_list = []
numPts_list = []
speed_list = []

ref_loc_list = []
ref_length_list = []
ref_numPts_list = []
ref_speed_list = []
for key in jam_ref_data.keys():
    ref_loc_list.append(jam_ref_data[key]['longitude_latitude'])
    loc_list.append(jam_ref_data[key]['longitude_latitude'])
    
    ref_length_list.append(jam_ref_data[key]['length'])
    length_list.append(jam_ref_data[key]['length'])
    
    ref_numPts_list.append(float(jam_ref_data[key]['numPts']))
    numPts_list.append(float(jam_ref_data[key]['numPts']))
    
    ref_speed_list.append(jam_ref_data[key]['speed'])
    speed_list.append(jam_ref_data[key]['speed'])


test_loc_list = []
test_length_list = []
test_numPts_list = []
test_speed_list = []
for key in jam_test_data.keys():
    test_loc_list.append(jam_test_data[key]['longitude_latitude'])
    loc_list.append(jam_test_data[key]['longitude_latitude'])
    
    test_length_list.append(jam_test_data[key]['length'])
    length_list.append(jam_test_data[key]['length'])
    
    test_numPts_list.append(float(jam_test_data[key]['numPts']))
    numPts_list.append(float(jam_test_data[key]['numPts']))
    
    test_speed_list.append(jam_test_data[key]['speed'])
    speed_list.append(jam_test_data[key]['speed'])

loc_data = loc_list
length_data = np.array(length_list)
numPts_data = np.array(numPts_list)
speed_data = np.array(speed_list)

len(loc_data), len(ref_loc_list), len(test_loc_list)

### Quantization levels: loc_k = 3, length_k = 2, numPts_k = 1, speed_k = 1

loc_data = [list(np.reshape(loc_data[i], np.size(loc_data[i]))) for i in range(len(loc_data))]

from scipy.cluster.vq import kmeans2
from ClusterAlg import DF, KMedians, KMedians_
from multiprocessing import Process

loc_k = 3

# create 4 sub-processes to do the work
p1 = Process(target=KMedians_, args=(loc_data, loc_k, DF,))
p1.start()
p1.join()

p2 = Process(target=KMedians_, args=(loc_data, loc_k, DF,))
p2.start()
p2.join()

p3 = Process(target=KMedians_, args=(loc_data, loc_k, DF,))
p3.start()
p3.join()

p4 = Process(target=KMedians_, args=(loc_data, loc_k, DF,))
p4.start()
p4.join()


# loc_k = 3
# loc_label, loc_centroid, distToClusterCenter, optDist = KMedians_(loc_data, loc_k, DF)
