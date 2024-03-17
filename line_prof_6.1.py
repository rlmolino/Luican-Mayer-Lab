"""
STS Line Profile Plotter 6.1
Laurent Molino
Code last modified: 2023-01-16
Comments last modified: 2024-03-17

Makes STS line profile plot from RHK-type STS data
Can remove bad curves and replace with average of neighbours
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import regex as re

#address is path to RHK type STS file
#return list of V, dos data array
def read_txt(address):
    sts_string = open(address).read()
    sts_list = []
    lines = sts_string.split('\n')
    for i in range(0,len(lines)):
            line = lines[i].split(' ')
            clean_line = ' '.join(line).split() #get rid of empty string elements
            sts_list.append(clean_line)
    data = np.array(sts_list[21:-2], dtype=float) #cuts off header and footer
    V = data[:,0]
    dos = np.matrix.transpose(data[:,1:])
    return V, dos

#input list of good curves as we usually record them (format '1,3-7,29,30-64')
#reutnres list of good curve values
def curve_list_filter(curve_string):
    filtered_list = []
    curve_list = curve_string.split(',')
    for i in range(0, len(curve_list)):
        if re.match('^\ ?[0-9]+\ ?$', curve_list[i]):
            filtered_list.append(int(curve_list[i]))
        elif re.match('^\ ?[0-9]+\-[0-9]+\ ?$', curve_list[i]):
            start = int(re.findall('^\ ?[0-9]+\ ?', curve_list[i])[0])
            end = int(re.findall('\ ?[0-9]+\ ?$', curve_list[i])[0])
            filtered_list += list(range(start,end + 1))
        else:
            print('ERROR: NO MATCH IN CURVELIST')
    filtered_list = [x - 1 for x in filtered_list]
    return filtered_list

#finds index of ellement nearest value within array
#need below
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

#reads data, does averaging, cuts off top and bottom
def process_data(address, curve_string, top=0, bottom=0):
    curve_list = curve_list_filter(curve_string)
    V, raw_dos = read_txt(address)
    n_curves = np.shape(raw_dos)[0]
    dos = np.copy(raw_dos)
    curve_list_index = 0
    #replace bad curves with averages of adgacent curves
    for i in range(0, n_curves):
        if i in curve_list:
            curve_list_index += 1
        else:
            if curve_list_index == 0: #if no previous good curves
                print('opening curve bad')
                dos[i,:] = dos[curve_list[0],:]#take first good curve
            elif curve_list_index == len(curvelist) - 1: #if last good curve is past
                print('closing curve bad')
                dos[i,:] = dos[curve_list[-1],:]#take last good curve is past
            else:
                dos[i,:] = np.mean([dos[curve_list[curve_list_index - 1],:], [dos[curve_list[curve_list_index],:]]], axis=0)
                #average of last good curve and next good curve
    #cut off top and bottom of range
    if top == 0:
        top_index = 0
    else:
        top_index = find_nearest(V, top)
    if bottom == 0:
        bottom_index = -1
    else:
        bottom_index = find_nearest(V, bottom)
    cut_dos = dos[:,top_index:bottom_index]    

    return V, cut_dos, top_index, bottom_index

#not used, from old version, keep here for good luck
def plot(data, size, n_fig):
    plot_extent = (0, size, 0, size)
    plt.figure(n_fig)
    plt.contourf(data, cmap='magma', levels = 100, extent=plot_extent)
    plt.axis('scaled')
    cbar = plt.colorbar()
    plt.xlabel('')
    plt.ylabel('')
    cbar.set_label('', rotation=270)
    cbar.set_ticks([])
 #   plt.set_xticks(np.arange(0, 20, 2).astype(int))
    plt.autoscale(enable=True)

#does plotting, see work area bwlow for argument meaning
def plot_2D(dos, V, top_ind, bottom_ind, dist, x_step, V_step, filename, flip, title=''):
    data = np.rot90(np.fliplr(dos))
    fig, ax = plt.subplots()
    plot_extent = (0, 10, 0, 10)
    if flip == True:
        data = np.flip(data, axis=1)
    plt.contourf(data, cmap='terrain', levels = 100, extent=plot_extent)

    nV = (V[bottom_ind] - V[top_ind])/V_step
    Vticks = np.round(np.arange(V[top_ind], V[bottom_ind] + (V[bottom_ind] - V[top_ind])/nV, (V[bottom_ind] - V[top_ind])/nV),1)
    ax.set_yticks(np.round((Vticks - V[top_ind])*10/(V[bottom_ind] - V[top_ind]), 4))
    ax.set_yticklabels(Vticks, fontdict= {'fontsize': 12, 'fontweight': 'bold'})

    end_tick = math.floor(dist)
    xticks = np.arange(0, end_tick + x_step, x_step)
    ax.set_xticks(np.round(xticks * 10 /dist,4))
    ax.set_xticklabels(xticks, fontdict= {'fontsize': 12, 'fontweight': 'bold'})

    cbar = plt.colorbar(pad=-0.11)
    cbar.set_label('log(dI/dV) (a.u.)', font='Arial', rotation=270, labelpad=12, fontsize=12.0, fontweight='bold')
    d_max = np.amax(dos)
    d_min = np.amin(dos)
    #cbar.set_ticks(np.arange(d_min, d_max + (d_max-d_min)/7, (d_max-d_min)/5).astype(int))
    cbar.set_ticks([])

    plt.axis('scaled')
    plt.xlabel('Position (nm)', font='Arial', fontsize=12.0, fontweight='bold')
    plt.ylabel('Bias Voltage (V)', font='Arial', fontsize=12.0, fontweight='bold')
    plt.autoscale(enable=True)
    plt.title(title)
    plt.savefig('{0}.png'.format(filename), format='png', dpi=1200)

#Makes colour bar saturate below min and above max value
def cut_ends(dos, min_val, max_val):
    new_dos = np.copy(dos)
    new_dos[new_dos<min_val] = min_val
    new_dos[new_dos>max_val] = max_val
    return new_dos

#smooths data with rolling average, within a cruve and between adjacent curves
def get_smooth_dos(dos, in_curve_ave, out_curve_ave):
    n_curves, n_points = np.shape(dos)
    smooth_dos = np.zeros(shape=(n_curves - 2* out_curve_ave, n_points - 2*in_curve_ave))
    for i in range(0, n_curves - 2*out_curve_ave):
        for j in range(0, n_points - 2*in_curve_ave):
            smooth_dos[i,j] = np.average(dos[i:i+2*out_curve_ave+1, j:j+2*in_curve_ave+1])
    return smooth_dos

#takes log of STS data
#ensures that everything is above zero by applying unifrom shift
#anything below zero is an artifact anyway
def make_log(dos):
    y_min = np.amin(dos)
    if y_min <= 0:
        dos = dos - 1.1 * y_min
    log_dos = np.zeros_like(dos)
    log_dos = np.log10(dos)
    return log_dos

#%%
#WORK AREA

#Main Setting:
data = '2023_01_13_16_allcurves1-22_24-32.txt' #path to data saved as asscii, including all curves (even bad)
curvelist = '1-59,62-64'  #stinrg of good curve list
line_length = 25.6  #nm
flip = True #Set True if first point is on the right

#Make it pretty:
CB_bound, VB_bound = 0, 0 #edges, default zero
x_step = 5 #x_ticks
V_step = 0.5 #V_ticks
in_curve_ave = 1 #smooting within curve
out_curve_ave = 2 #smooting between curves
bottom = -13.2 #bottom cutoff in log
top = -10.3 #top cutoff in log

#%%
#RUN EVERYTHING
V, dos, top_ind, bottom_ind  = process_data(data, curvelist, top=VB_bound, bottom=CB_bound)#input all curves including bad ones
s_dos = get_smooth_dos(dos, in_curve_ave, out_curve_ave)
log_dos = make_log(s_dos)
log_dos = cut_ends(log_dos, bottom, top)
plot_2D(log_dos, V, top_ind, bottom_ind, line_length, x_step, V_step, data[0:-4], flip, title='')