"""
Band-Gap Fitter 3.0
Laurent Molino
Code last modified: 2023-01-16
Comments last modified: 2024-03-17

Used to determine band-gap from STS data
Acounts for rounding of di/dv corners due to band-bending
Works by fitting a line to the log of the DOS on the VB, CB and gap, and finding intersections
"""

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

#unpacks STS data from RHK file-type
#returns di/dV as a function of V
def read_txt(address):
    sts_string = open(address).read() 
    sts_list = []
    lines = sts_string.split('\n')
    for i in range(0,len(lines)):
            line = lines[i].split(' ')
            clean_line = ' '.join(line).split() #get rid of empty string elements
            sts_list.append(clean_line)
    sts_data_array = np.array(sts_list[21:-2], dtype=float) #cuts off header and footer
    return sts_data_array

#take log of sts data
def make_log(linear_sts):
    y_min = min(linear_sts[:,1])
    log_sts = np.zeros(shape=(len(linear_sts[:,0]),2))
    log_sts[:,0] = linear_sts[:,0]
    if y_min < 0:
        log_sts[:,1] = np.log10(linear_sts[:,1] - 2 * y_min)
    else:
        log_sts[:,1] = np.log10(linear_sts[:,1])
    return log_sts

#fits a line to the data within the bounds
#bounds formate in [upper_bound, lower_bound]
#if testing == True, returns line just in fit interval for better visualization
def fit_line(sts, bounds, testing):
    fit_interval_list = []
    for i in range(0,len(sts[:,0])):
        if sts[i,0] >= bounds[0] and sts[i,0] <= bounds[1]:
            fit_interval_list.append(sts[i,:])
    fit_interval = np.array(fit_interval_list, dtype=float)
    fit = ss.linregress(fit_interval)
    m = fit.slope
    b = fit.intercept
    m_error = fit.stderr
    b_error = fit.intercept_stderr
    if testing == True:
        line = np.zeros(shape=(len(fit_interval[:,0]),2))
        line[:,0] = fit_interval[:,0]
        line[:,1] = m * fit_interval[:,0] + b
    elif testing == False:
        line = np.zeros(shape=(len(sts[:,0]),2))
        line[:,0] = sts[:,0]
        line[:,1] = m * sts[:,0] + b
    return [m, b, m_error, b_error], line

#finds interection between two lines
def intersection(fit_1, fit_2):
    m1 = fit_1[0]
    b1 = fit_1[1]
    m1_error = fit_1[2]
    b1_error = fit_1[3]
    m2 = fit_2[0]
    b2 = fit_2[1]
    m2_error = fit_2[2]
    b2_error = fit_2[3]
    x = (b2 - b1) / (m1 - m2)
    x_error = np.sqrt((1/((m1 - m2)**2)) * (b1_error**2 + b2_error**2) + \
            ((b2 - b1)**2/((m1 - m2)**4)) * (m1_error**2 + m2_error**2))
    return x, x_error

#retuens value with correct number of sig-figs based on error
def round_to_error(x, x_error):
    x_error_dec = -1 * int(np.floor(np.log10(abs(x_error))))
    x_rounded = round(x, x_error_dec)
    x_error_rounded = round(x_error, x_error_dec)
    return x_rounded, x_error_rounded

#plots STS data with linear fits for VB, CB and gap
#saves at addresss
#If testing == True, plots in format that is less pretty but easier to see what is happening
def plot_fits(sts, vb_line, gap_line, cb_line, address, testing):
    if testing == True:
        sts_range = abs(max(sts[:,1]) - min(sts[:,1]))
        fig, ax = plt.subplots(figsize=(12, 6), dpi=600)
        ax.plot(sts[:,0], sts[:,1], color='red', label = 'log(STS) [arb. units]', linewidth=0.5)
        ax.plot(vb_line[:,0], vb_line[:,1], '--', color='black', label = 'VB Fit', linewidth=0.5)
        ax.plot(gap_line[:,0], gap_line[:,1], '--', color='black', label = 'Gap Fit', linewidth=0.5)
        ax.plot(cb_line[:,0], cb_line[:,1], '--', color='black', label = 'CB Fit', linewidth=0.5)
        ax.set_xlabel('Energy [V]', fontdict=dict(weight='bold'), fontsize=4)
        ax.set_ylabel('log(STS) [arb. units]', fontdict=dict(weight='bold'), fontsize=4)
        ax.set_ylim([min(sts[:,1]) - 0.025 * sts_range, max(sts[:,1]) + 0.025 * sts_range])
        fig.savefig('{0}.png'.format(address[0:-4]), transparent=True)
        ax.axes.yaxis.set_ticks([])
        #ax.set_title(address)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='minor', length=2, width = 1, fontsize=4)
        ax.grid(which='both')
        plt.show()
    
    elif testing == False:
        sts_range = abs(max(sts[:,1]) - min(sts[:,1]))
        fig, ax = plt.subplots(figsize=(4*1.2, 6*1.2), dpi=200)
        ax.plot(sts[:,0], sts[:,1], color='red', label = 'log(STS) [arb. units]')
        ax.plot(vb_line[:,0], vb_line[:,1], '--', color='black', label = 'VB Fit')
        ax.plot(gap_line[:,0], gap_line[:,1], '--', color='black', label = 'Gap Fit')
        ax.plot(cb_line[:,0], cb_line[:,1], '--', color='black', label = 'CB Fit')
        ax.set_xlabel('Energy [V]', fontdict=dict(weight='bold'), fontsize=14)
        ax.set_ylabel('log(STS) [arb. units]', fontdict=dict(weight='bold'), fontsize=14)
        ax.set_ylim([min(sts[:,1]) - 0.025 * sts_range, max(sts[:,1]) + 0.025 * sts_range])
        ax.axes.yaxis.set_ticks([])

#runs everything
#testing == True until you are happy with your bounds
#bound expects [[vb1,vb2],[gap1,gap2],[cb1,cb2]]
def run_file(address, bounds, testing):
    linear_sts = read_txt(address)
    sts = make_log(linear_sts)
    vb_fit, vb_line = fit_line(sts, bounds[0], testing)
    gap_fit, gap_line = fit_line(sts, bounds[1], testing)
    cb_fit, cb_line = fit_line(sts, bounds[2], testing)
    vb_edge, vb_edge_error = intersection(vb_fit, gap_fit)
    cb_edge, cb_edge_error = intersection(cb_fit, gap_fit)
    gap = cb_edge - vb_edge
    gap_error = np.sqrt(vb_edge_error**2 + cb_edge_error**2)
    
    vb_edge_rounded, vb_edge_error_rounded = round_to_error(vb_edge, vb_edge_error)
    cb_edge_rounded, cb_edge_error_rounded = round_to_error(cb_edge, cb_edge_error)
    gap_rounded, gap_error_rounded = round_to_error(gap, gap_error)
    
    print('{0} Properties:'.format(address))
    print('Valance Band Edge: {0} \pm {1}'.format(vb_edge_rounded, vb_edge_error_rounded))
    print('Conduction Band Edge: {0} \pm {1}'.format(cb_edge_rounded, cb_edge_error_rounded))
    print('Band Gap: {0} \pm {1}'.format(gap_rounded, gap_error_rounded))
    
    plot_fits(sts, vb_line, gap_line, cb_line, address, testing)
    
    vb_line_unlog = np.zeros_like(vb_line)
    vb_line_unlog[:,0] = vb_line[:,0]
    vb_line_unlog[:,1] = 10**vb_line[:,1]
    
    gap_line_unlog = np.zeros_like(gap_line)
    gap_line_unlog[:,0] = gap_line[:,0]
    gap_line_unlog[:,1] = 10**gap_line[:,1]
    
    cb_line_unlog = np.zeros_like(cb_line)
    cb_line_unlog[:,0] = cb_line[:,0]
    cb_line_unlog[:,1] = 10**cb_line[:,1]
    
    namebase = address[0:-4]
    np.savetxt('{0}_vb.txt'.format(namebase), vb_line_unlog)
    np.savetxt('{0}_gap.txt'.format(namebase), gap_line_unlog)
    np.savetxt('{0}_cb.txt'.format(namebase), cb_line_unlog)
    
#%%    
#Inputs Here:
#address is path to RHK-type data file for STS curver
# vb1, vb2, etc. are bound of region within which fit is preformed
#testing == True until you are happy with your bounds
#testing == False makes it pretty
    
address = '2022_07_26_11_48_allcurves_points.txt'
vb1 = -0.9
vb2 = -0.7
gap1 = -0.5
gap2 = 0.2
cb1 = 0.4
cb2 = 0.6
testing = False

#%%
#Run Everything:
bounds = [[vb1,vb2],[gap1,gap2],[cb1,cb2]]
run_file(address, bounds, testing)