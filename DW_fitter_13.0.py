"""
Domain Wall Fitter 13.0
Laurent Molino
Code last modified: 2023-03-22
Comments last modified: 2024-03-17

Used for visually fitting STM data of domain walls in P-aligned TMD homobilayer 
TMDs in out-of-plane E-feild to string-like model, as preseneted in
https://doi.org/10.1002/adma.202207816

See Work Area on line 360 for detailed use instructions
"""

#### CODE AREA, DO NOT TOUCH ####
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import cv2
import math
from scipy import optimize

#removes curves of a certain type from vs list
def remove_type(n, vs):
    vs_trimed = np.copy(vs)
    for i in range(0, len(vs[4,:])):
        if vs[i,4] == n:
            vs_trimed = np.delete(vs, i, 0)
    return vs_trimed

#returns length of line
def get_len(line_vs):
    x0, y0, x1, y1 = line_vs[0], line_vs[1], line_vs[2], line_vs[3]
    l = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    return l

#this is f from the model
def f(x, l, A, B):
    # af^3 + bf^2 + cf +d
    #[a, b, c, d]
    roots = np.roots([1, 0, -1*A, -2*B*(x - l/2)])
    is_real = np.isreal(roots)
    good_roots = []
    for i in range(0, len(roots)):
        if is_real[i] == True and abs(roots[i]) <= 1:
            good_roots.append(roots[i])
    if len(good_roots) == 1:
        return good_roots[0]
    else:
        print('ERROR: {0} GOOD ROOTS'.format(len(good_roots)))
        print(roots)
          
def int_arg(x, l, A, B):
    f_val = f(x, l, A, B)
    return f_val / np.sqrt(1 - f_val**2)

#this is F from the model
def F(x, l, A, B):
    return quad(int_arg, 0, x, args=(l,A,B))[0]

#alpha from model, acounts for iso triangles
def shift_curve(Fs, line_vs):
    x0, y0, x1, y1 = line_vs[0], line_vs[1], line_vs[2], line_vs[3]
    alpha = np.arctan2(y1 - y0, x1 - x0)
    new_Fs = np.zeros_like(Fs)
    new_Fs[:,0] = Fs[:,0] * np.cos(alpha) - Fs[:,1] * np.sin(alpha) + x0
    new_Fs[:,1] = Fs[:,1] * np.cos(alpha) + Fs[:,0] * np.sin(alpha) + y0
    return new_Fs

#generate equilateral triangle domain walls
#line vs is vertices
#A and B are model parameters
#if var_A == True, corrects for misalignment from crystal direction
def eql_DW(line_vs, A, B, theta, var_A):
    if var_A == 'all':
        theta_rad = theta * 2*np.pi/360
        line_angle = get_vs_angle(line_vs)
        grid_angles = np.ones(7) * theta_rad  +  np.arange(0,7,1) * 60 * 2*np.pi/360
        phis = np.ones(7) * line_angle - grid_angles
        phi = np.amin(abs(phis))
        A_mod = A + 3*np.cos(phi) - 3
    else:
        A_mod = A
    x0, y0, x1, y1 = line_vs[0], line_vs[1], line_vs[2], line_vs[3]
    l = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    xs = np.arange(0, l, 0.1)
    Fs = np.zeros(shape=(len(xs),2))
    Fs[:,0] = xs
    for j in range(0, len(xs)):
        Fs[j,1] = F(Fs[j,0], l, A_mod, B)*np.sign(line_vs[4]) #flips sign
    trans_Fs = shift_curve(Fs, line_vs)
    return trans_Fs

#solve model equation with bisection
def bisect(f, bounds, instep=1e-4, error_tol=1e-6):
    a, b = bounds[0] + instep, bounds[1] - instep
    assert(f(a) * f(b) <0)
    error = abs(a - b)/2.
    phi = (a + b)/2.
    count = 0
    while error > error_tol and count < 1000:
        if f(a)*f(phi) < 0:
            b = phi
        elif f(b)*f(phi)  < 0:
            a = phi
        else:
            print('ERROR')
        phi_new = (a + b)/2.
        error = abs(phi - phi_new)
        phi = phi_new
        count += 1
    return phi

#finds angle of line between two points
def get_vs_angle(line_vs):
    rise = line_vs[3] - line_vs[1]
    run  = line_vs[2] - line_vs[0]
    angle = np.arctan2(rise, run) % (2*np.pi)
    return angle

#plots DW for iso triangles, l0 is traingle base
#accounds for domain wall merging
#here a is alpha, d is delta
def iso_DW(l0_vs, l1_vs, l2_vs, A, B, scale, res, pix_cutoff, theta, var_A):
    l0, l1, l2 = get_len(l0_vs), get_len(l1_vs), get_len(l2_vs)
   
    #caculates A_mod
    if var_A == 'iso' or var_A == 'all':
        theta_rad = theta * 2*np.pi/360
        l1_angle = get_vs_angle(l1_vs)
        grid_angles = np.ones(7) * theta_rad  +  np.arange(0,7,1) * 60 * 2*np.pi/360
        phis = np.ones(7) * l1_angle - grid_angles
        phi = np.amin(abs(phis))
        A_mod = A + 3*np.cos(phi)**2 - 3
    else:
        phi = 0
        A_mod = A
   
    #calculate angles
    a = np.arccos((l1**2 + l2**2 - l0**2)/(2*l1*l2))
    a_opt = lambda a1 : ( (l1 - np.sqrt(2*l1*A_mod*np.tan(a1)/B)) / (np.cos(a1)) ) - ((l2 - np.sqrt(2*l2*A_mod*np.tan(a - a1)/B)) / (np.cos(a - a1)) )
    a1 = bisect(a_opt, [0, a])
   
    #calculate other variables
    l1_star = 2*A_mod*np.tan(a1) / B
    d1 = (l1 - np.sqrt(l1*l1_star)) * np.heaviside((l1 - l1_star),0)
    if l1 <= l1_star:
        x0 = l1/2.0
    elif l1 > l1_star:
        x0 = l1 - np.sqrt(l1*l1_star) + l1_star/(2.0)
    else:
        print('ERROR with l_star')
    L1 = d1 / np.cos(a1)
   
    #plot
    xs_ddw = np.arange(0, d1, 0.1)
    xs_crv = np.arange(d1, l1, 0.1)
    ys_ddw = np.zeros(shape=(len(xs_ddw),2))
    ys_crv = np.zeros(shape=(len(xs_crv),2))
    ys_ddw[:,0] = xs_ddw
    ys_crv[:,0] = xs_crv
    for i in range(0, len(xs_ddw)):
        ys_ddw[i,1] = np.tan(a1) * xs_ddw[i] * np.sign(l1_vs[4])
    for j in range(0, len(xs_crv)):    
            ys_crv[j,1] = (-1*B/(2.0*A_mod)) * (xs_crv[j] - l1) * (xs_crv[j] - 2*x0 + l1) * np.sign(l1_vs[4])      
    trans_ys_ddw = shift_curve(ys_ddw, l1_vs)
    trans_ys_crv = shift_curve(ys_crv, l1_vs)
   
    #calculate L_eff, point at which deviation exced resolution
    res_limit = pix_cutoff*scale/res
    ys_eff =  np.copy(ys_crv)
    ys_eff[:,1] = abs(ys_eff[:,1] - np.tan(a1) * ys_eff[:,0] * np.sign(l1_vs[4]))
    for i in range(0, len(xs_crv)):
        if ys_eff[i,1] > res_limit:
            L_eff = ys_eff[i,0]
            break
        else:
            L_eff = 0
   
    return trans_ys_ddw, trans_ys_crv, l0, L1, L_eff, A_mod, phi

#manually plots DDWs
#don't generally use this
def DDW(line_vs):
    x0, y0, x1, y1 = line_vs[0], line_vs[1], line_vs[2], line_vs[3]
    return np.array([[x0, y0],[x1, y1]])
   
#expects vs = [[x0, y0, x1, y1, sign], ...]
#main function for plotting domain wall, calls on above
#if plotting DDW manually, will report comparion of manually selected DDW to theory
#see paper for comparison details
def plot_dw(pic, vs, A, Bs, theta, var_A, suf = '', plot_pic=True, plot_points=False, plot_lines=False, plot_grid=False, report=False, scale=200, res=128, sb=True, pix_cutoff=1, plot_ddw=False):
    colors = ['violet', 'white', 'white']
   
    for B in Bs:
        fig, ax = plt.subplots()
        n_lines = len(vs[:,0])
       
        if plot_grid == True:
            ax.xaxis.set_ticks(np.arange(0, scale+1, scale/20))
            ax.yaxis.set_ticks(np.arange(0, scale+1, scale/20))
            ax.grid()
        else:
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.title.set_text('{0} \n B = {1}'.format(pic, B))
            if sb == True:
                scalebar = ScaleBar(1, "nm", fixed_value=50, location='lower right', color='white', box_alpha=0.0, label_loc='top')
                ax.add_artist(scalebar)
           
        if plot_points == True:
            n_iso = np.count_nonzero(vs[:,4] == 0)
            noDDW_vs = np.copy(vs[n_iso:,:])
            ax.plot(noDDW_vs[:,0], noDDW_vs[:,1], marker="o", markersize=10, color='white', fillstyle='none', linewidth=0)
            ax.plot(noDDW_vs[:,2], noDDW_vs[:,3], marker="o", markersize=10, color='white', fillstyle='none', linewidth=0)
           
        if plot_lines == True:
            n_iso = np.count_nonzero(vs[:,4] == 0)
            L_vs_l0 =np.zeros(shape = (2*n_iso, 4))
            A_mods = np.zeros(shape = (2*n_iso, 2))
            for i in range(0, n_lines):
                line_vs = vs[i,:]
                if abs(line_vs[4]) == 0:
                    Fs = DDW(line_vs)
                elif abs(line_vs[4]) == 1:
   
                   
                    l1_vs = vs[i,:]
                    if (i - n_iso)%2 == 0:
                        l2_vs = vs[i+1,:]
                    elif (i - n_iso)%2 == 1:
                        l2_vs = vs[i-1,:]
                    l0_index = 3*n_iso + math.floor((i - n_iso)/2.0)
                    l0_vs = vs[l0_index,:]
                   
                    Fs_ddw, Fs_curve, l0i, Li_theory, L_eff, A_mod, phi = iso_DW(l0_vs, l1_vs, l2_vs, A, B, scale, res, pix_cutoff, theta, var_A)
                    
                    if plot_ddw == True:
                        nL = int((i - n_iso)/2)
                        nl = int(i - n_iso)
                        L_vs = vs[nL,:]
                        L_exp = get_len(L_vs)
                        L_vs_l0[nl,:] = np.array([l0i, Li_theory, L_exp, L_eff])
                        A_mods[nl,0:2] = np.array([A_mod, phi*360/(2*np.pi)])
                       
                    ax.plot(Fs_curve[:,0], Fs_curve[:,1], color=colors[abs(line_vs[4])], linewidth=2)
                    ax.plot(Fs_ddw[:,0], Fs_ddw[:,1], color='red', linewidth=2)
                   
                elif abs(line_vs[4]) == 2:
                    Fs = eql_DW(line_vs, A, B, theta, var_A)
                    ax.plot(Fs[:,0], Fs[:,1], color=colors[abs(line_vs[4])], linewidth=2)
                   
                else:
                    print('ERROR: UNDEFINED DW NUMBER')
               
            if plot_ddw == True:
                print('l0, L_theory, L_exp, L_eff:')
                print(repr(L_vs_l0))
                print()
                print('line_vs, A_mod, phi:')
                print(repr(A_mods))
        if plot_pic == True:
            im = np.flip(plt.imread(pic), axis=0)
            implot = plt.imshow(im, origin='lower', extent=[0,scale,0,scale])#, extent=[500,500])
            plt.savefig('{0}_dwf{1}.png'.format(pic[0:-4], suf), dpi=600, bbox_inches='tight')
            plt.show()
           
    if report == True:
        vs_with_l = np.zeros(shape=(len(vs[:,0]), 3))
        vs_with_l[:, 0:2] = vs[:, 4:6]
        x0, y0, x1, y1 = vs[:, 0], vs[:, 1], vs[:, 2], vs[:,3]
        l = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        vs_with_l[:,2] = l
        name = '{0}_pnts_{1}.txt'.format(pic[0:-4], suf)
        np.savetxt(name, vs)

#select vertices by clicking
#https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)
        clicks.append(x)
        clicks.append(y)

def clicks_to_vs(clicks, i, n, scale, n_pixels):
    if len(clicks)%4 != 0:
        print('ERROR: UNPAIRED VERTICES')
    else:
        n_lines = int(len(clicks)/4)
        vs_points_only = np.reshape(clicks, (n_lines, 4))
        vs = np.zeros(shape=((n_lines, 6)), dtype=int)
        vs[:,0:4] = vs_points_only * scale / n_pixels #switchs from default scale to correct scale
        vs[:,1] = -1 * vs[:,1] + scale #flips y-axis
        vs[:,3] = -1 * vs[:,3] + scale
        vs[:,4] = i * np.ones(n_lines)
        if i == 1: #selects base for iso, assumes you select bases first in eql, in order
            for j in range(0, n_lines):
                vs[j,5] = n + math.floor(j/2.0)
    return(vs)

#makes sure domain wall meet by matching points within a threshold radius
def match_points(vs, thresh):
    matched_vs = np.copy(vs)
    n_lines = np.shape(vs)[0]
    for i in range(0, n_lines):
        for j in range(0, 2):
            point1 = matched_vs[i, 2*j:2*j+2]
            for k in range(i, n_lines):
                for l in range(0, 2):
                    point2 = matched_vs[k, 2*l:2*l+2]
                    if np.amax(abs(point1 - point2)) <= thresh:
                        matched_vs[k, 2*l:2*l+2] = point1
    return matched_vs

#shits points within threshold radius from new point
def shift_points(new_points, vs, thresh):
    matched_vs = np.copy(vs)
    n_lines = np.shape(vs)[0]
    for n in range(0, len(new_points)):
        new_point = new_points[n]
        for i in range(0, n_lines):
            for j in range(0, 2):
                point = matched_vs[i, 2*j:2*j+2]
                if np.amax(abs(new_point - point)) <= thresh:
                    matched_vs[i, 2*j:2*j+2] = new_point
    return matched_vs

#changes order of points in curve
def flip_curve(lines, vs, thresh):
    matched_vs = np.copy(vs)
    n_lines = np.shape(vs)[0]
    for n in range(0, len(lines)):
        point1 = lines[n,0:2]
        point2 = lines[n,2:4]
        for i in range(0, n_lines):
            point3 = matched_vs[i, 0:2]
            point4 = matched_vs[i, 2:4]
            if (np.amax(abs(point1 - point3)) <= thresh and np.amax(abs(point2 - point4)) <= thresh) or \
                (np.amax(abs(point1 - point4)) <= thresh and np.amax(abs(point2 - point4)) <= thresh):
                matched_vs[i, 4] = matched_vs[i, 4] * -1
    return matched_vs

#Changes scale on vertex network
def rescale_vs(vs, old_scale, new_scale):
    scale_factor = int(new_scale)/old_scale
    new_vs = np.zeros_like(vs)
    new_vs[:,0:4] = vs[:,0:4] * scale_factor
    new_vs[:,4:7] = vs[:,4:7]
    return new_vs

#%%
#### WORKING AREA, MODIFY HERE ####

#1: Select model parameters
#Select and A and B values
#A = 3.52 for tWS2
#Can input mulitple values of B seperated by commas, will make image for each
#B typically from 0.005 to 0.02
A = 3.52
Bs = [0.009] 

#2: Select file name
#pic is path to scan file
#suffix is appended to scan file name when saveing
pic = '632_-1.3V_8.png'
suffix = '_LM'

#3: Select scan parameters
#scale is scan size in nm
#pix_cutoff is error on domain wall position in pixels (used to calculated DDW length error)
#theta is angle of the unstrained network, used to modify A parameter
#var_A selects for what domain walls the A paramter is modified
scale = 200
pix_cutoff = 1
theta = 20
var_A = 'none'  #'iso' or 'all' or 'none'

#3: Select Location Of Vertices:
#vs_source controls how how get your verticies
#vs_source can be: 'c' to pick by clicking
#                  'f' to read from file
#                  't' to type below
vs_source ='t'

#if typing fill in here
#expects [x0, y0, x1, y1, wall_type*cuvature sign, base index (iso only)]
#wall types 0 = DDW (do not use in general), 1 = elgonagate long side, 2 = regular
#curavture sign must be 1, -1
typed_vs =np.array([[ 12, 107,  83, 152,  -2,   0],
       [ 83, 152, 150, 195,  -2,   0],
       [150, 195, 187, 130,   2,   0],
       [187, 130, 135,  95,   2,   0],
       [135,  95, 150, 195,   2,   0],
       [135,  95,  83, 152,  -2,   0],
       [ 83, 152,  74,  46,  -2,   0],
       [ 74,  46, 135,  95,  -2,   0],
       [135,  95, 174,  43,   2,   0],
       [174,  43, 117,   7,   2,   0],
       [117,   7, 135,  95,   2,   0],
       [117,   7,  74,  46,  -2,   0],
       [ 74,  46,  12, 107,  -2,   0],
       [174,  43, 187, 130,   2,   0]])


#Can adjust what kinds of DW are ploted
plot_ddw = False
plot_iso = False
plot_eql = True

#Can turn off plotting points, lines to speed up
plot_points = False 
plot_lines = True

#5: Adgust to improve fit
#If points need to be shifted, set True
#Input new values as array of [x,y], moves nearest point
new_points = np.array([[20,112],[91,160],[158,198],[195,129],[178,45],[124,9],[77,52],[138,96]])
#If cruvature is backwards enter line here
#Input lines as array of [x0,y0,x1,y1]
lines_to_flip = np.array([])

#6:Rescaling
#Set true if you did the fitting with wrong scale and want to fix it
rescale = False
old_scale = 100
new_scale = scale

#%%
#### CODE AREA, DO NOT TOUCH ####
print(Bs)

if __name__=="__main__":
    wall_types = [plot_ddw, plot_iso, plot_eql]
    wall_names = ['DDW', 'Isosceles', 'Equilateral']
   
    if vs_source == 'c':
        n_wall = 0
        for i in range(0, len(wall_types)):
            if wall_types[i] == True:
                print('Select Wall Type {0}'.format(wall_names[i]))
                clicks = []
                img = cv2.imread(pic, 1)
                n_pixels = np.shape(img)[0]        
                cv2.imshow('image', img)
                cv2.setMouseCallback('image', click_event)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                n_wall += len(clicks)/4
                vs_i = clicks_to_vs(clicks, i, n_wall, scale, n_pixels)
                if any(wall_types[0:i]) == False:
                    vs = vs_i
                else:
                    vs = np.concatenate((vs, vs_i), axis = 0)
    elif vs_source == 'f':
        vs = np.loadtxt('{0}_pnts_{1}.txt'.format(pic[0:-4], suffix), dtype=int)
    elif vs_source == 't':
        vs = typed_vs
    else:
        print('ERROR: MUST CHOOSE vs SOURCE')
       
    vs = shift_points(new_points, vs, 10)
    vs = flip_curve(lines_to_flip, vs, 5)
    vs = match_points(vs, 5)
    
    if rescale == True:
        vs = rescale_vs(vs, old_scale, new_scale)
   
    print(repr(vs))
    plot_dw(pic, vs, A, Bs, theta, var_A, plot_points=plot_points, plot_grid=False, suf=suffix, plot_lines=plot_lines, scale=scale, report=True, sb=False, pix_cutoff=pix_cutoff, plot_ddw=plot_ddw)