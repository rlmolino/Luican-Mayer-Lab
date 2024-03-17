"""
RHK Curve Processor 2.0
Laurent Molino
Code last modified: 2022-09-30
Comments last modified: 2024-03-17

IDs and removes bad curves from STS data set
accepts RHK type data file
removes curves that are blank, saturate, of loose surface
averages curves, all together or in bins
"""
import numpy as np
import math
import argparse

#unpacks RHK data, part 1
def make_data_list(address):
    data_string = open(address).read() 
    data_list = []
    lines = data_string.split('\n')
    for i in range(0,len(lines)):
            line = lines[i].split(' ')
            clean_line = ' '.join(line).split() #get rid of empty string elements
            data_list.append(clean_line)
    return data_list

#unpacks RHK data, part 2
def filter_rhk_file(data_list):
    position_data = np.array(np.array(data_list[19:21])[:,1:],dtype=float)
    sts_data = np.array(data_list[21:-2],dtype=float)
    return position_data, sts_data

#based on current data, IDs good curves
#rejects curves that saturate (min/max over +/- 10nA)
#rejects curves without VB on CB DOS (min/max under +/- 0.1 nA)
#rejects curves with big discontinuities (over disc_thresh)
def get_curve_list(current_data, disc_thresh=1e-9):
    curve_list = []
    rev9_cruve_list = []
    n_curves = np.shape(current_data)[1] - 1
    x_data = current_data[:,0]
    y_data = current_data[:,1:]
    for i in range(0, n_curves): #curve_number
        good_curve = True
        y_min = min(y_data[:,i])
        y_max = max(y_data[:,i])
        if y_min >= -1e-10 or y_max <= 1e-10 or y_min <= -9.99e-9 or y_max >= 9.99e-9:
            good_curve = False
        if disc_thresh > 0:
            for j in range(1, len(y_data[:,i])):
                if abs(y_data[j-1,i] - y_data[j,i]) >= disc_thresh:
                    good_curve = False
        if good_curve == True:
            curve_list.append(i)
            rev9_cruve_list.append(i+1)
    print("{0} good curves".format(len(curve_list)))
    print(rev9_cruve_list)
    return curve_list

#bins curves and averages them
def bin_ave(bin_size, sts_data, ave=False): #bin_size doesn't matter in ave==True
    n_curves = np.shape(sts_data)[1] - 1
    x_data = sts_data[:,0]
    y_data = sts_data[:,1:]
    if ave == True:
        bin_size = n_curves
    n_bins = math.floor(n_curves/bin_size)
    bin_data = np.zeros(shape=(np.shape(sts_data)[0],1 + n_bins))
    bin_data[:,0] = x_data
    for i in range(0, n_bins):
        for j in range(0, bin_size):
            bin_data[:,i + 1] += y_data[:,i * bin_size + j] / bin_size
    return bin_data

#makes array of good curves based on curve_list
def select_data(sts_data, curve_list):
    x_data = sts_data[:,0]
    y_data = sts_data[:,1:]
    good_data = np.zeros(shape=(np.shape(sts_data)[0],1 + len(curve_list)))
    good_data[:,0] = x_data
    good_data[:,1:] = y_data[:,curve_list]  
    return good_data

#main loop, calls above functions to unpack files, remove bad curves, do averaging
def main(i_files, didv_files, ave, bin_list, names):
    if len(i_files) != len(didv_files):
        print('ERROR: NUMBER OF FILES DO NOT MATCH')
    if len(i_files) != len(names):
        print('ERROR: NUMBER OF NAMES DOES NOT MATCH')
    
    
    for i in range(0,len(i_files)):
        i_file = i_files[i]
        i_list = make_data_list(i_file)
        i_data = filter_rhk_file(i_list)[1]
        didv_file = didv_files[i]
        didv_list = make_data_list(didv_file) 
        didv_data = filter_rhk_file(didv_list)[1]
        
        curve_list = get_curve_list(i_data)
        didv_good_data = select_data(didv_data, curve_list)
        np.savetxt('{0}_dIdV.txt'.format(names[i]), didv_good_data)
        if ave == True:
            ave_data = bin_ave(0, didv_good_data, ave=True)
            np.savetxt('{0}_dIdV_ave.txt'.format(names[i]), ave_data)
        for j in range(0,len(bin_list)):
            bin_data = bin_ave(bin_list[j], didv_good_data)
            np.savetxt('{0}_dIdV_bin{1}.txt'.format(names[i],bin_list[j]), bin_data)

#%%
#Edit Here:
i_files = ['2022-09-30/TaS2_0753_i.txt']
didv_files = ['2022-09-30/TaS2_0753_didv.txt']
ave = True #averages all
curves_per_bin = 1 #average bin size
names = ['2021_06_30_09_32'] #out file name base

#%%
#Run Here:
i_data = main(i_files, didv_files, ave, curves_per_bin, names)