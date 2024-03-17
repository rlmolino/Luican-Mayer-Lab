"""
Moire Plotter 2.2
Laurent Molino
May 18th, 2021
Code last modified: 2021-05-18
Comments last modified: 2024-03-17

Models moire pattern formed between two lattices
"""
import numpy as np
import matplotlib.pyplot as plt

#twists set of lattice vectors by angle theta
def twist_lvs(lvs, theta_rad):
    t_lvs =  np.zeros(shape=(2,2))
    R = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],[np.sin(theta_rad), np.cos(theta_rad)]])
    t_lvs[0,:] = np.dot(R, lvs[0,:])
    t_lvs[1,:] = np.dot(R, lvs[1,:])
    return t_lvs

#creats 2x2 array of lattice vectors
def make_lvs(a, b, gamma_deg, theta_deg):
    gamma_rad = gamma_deg * np.pi/180
    theta_rad = theta_deg * np.pi/180
    lvs = np.zeros(shape=(2,2))
    lvs[0,0] = a 
    lvs[1,0] = b * np.cos(gamma_rad)
    lvs[1,1] = b * np.sin(gamma_rad)
    t_lvs = twist_lvs(lvs, theta_rad)
    return t_lvs

#converts point from reduced-coordinates to cartesian coordinates
def rc_to_cc(rc_point, lvs):
    cc_point = np.zeros(shape=2)
    for i in [0,1]:
        for j in [0,1]:
            cc_point[i] += rc_point[j] * lvs[j,i]
    return cc_point

#converts point from cartisian-coordinates to reduced-coordinateds
#based on general case https://en.wikipedia.org/wiki/Fractional_coordinates
#need to add dummy lattice vector in third dimension
def cc_to_rc(cc_point_2d, lvs, mod=np.inf):
    #convert everything to 3d
    cc_point_3d = np.zeros(shape=3)
    cc_point_3d[0:2] =  cc_point_2d
    lvs_3d = np.zeros(shape=[3,3])
    lvs_3d[0:2,0:2] = lvs
    lvs_3d[2,2] = 1
    #initialize local variables
    rc_point_2d = np.zeros(shape=2)
    rc_point_3d = np.zeros(shape=3)
    face_areas = np.zeros(shape=(3,3))
    #do math
    for i in range(0,3):
        face_areas[i][:] = np.cross(lvs_3d[(i+1)%3][:], lvs_3d[(i+2)%3][:])
    cell_vol = np.dot(lvs_3d[0][:], face_areas[0][:])
    for j in range(0,3):
        rc_point_3d[j] = np.dot(cc_point_3d, face_areas[j][:]) / cell_vol
    #return to 2d, apply modulo
    rc_point_2d = rc_point_3d[0:2]%mod
    return rc_point_2d

#calculates distances between point a and b, which are two element vectors
def get_dist(a, b):
    d2 = 0
    for i in range(0,2):
        d2 += (b[i] - a[i])**2
    d = np.sqrt(d2)
    return d

#get value of gaussian of given std with x=dist
def gaussian(dist, std):
    val = 1.0/(std * np.sqrt(2*np.pi)) * np.exp(-1.0/2.0 * (dist /std)**2 )
    return val

#makes a list of atoms that atom_range steps away the central unit cell
def make_atom_list(atom_range, lvs):
    atom_list = []
    for i in range(-atom_range, atom_range + 2):
        for j in range(-atom_range, atom_range + 2):
            atom = rc_to_cc([i,j], lvs)
            atom_list.append(atom)
    return atom_list
    
#normalizes 2d array, assumes data is all positive
def norm(data):
    dims = list(np.shape(data))
    norm_data = np.zeros(shape=dims)
    max_val = np.max(data)
    for i in range(0, dims[0]):
        for j in range(0, dims[1]):
            norm_data[i,j] = data[i,j] / max_val
    return norm_data

#creats a square array representing one unit cell in reduced co-ordinates
#values of 2 seem to work fine for atom_range and res_factor
#Call gaussian for atom shape by default, could use different shape
def make_uc_square(lvs, res, atom_range=2, res_factor=2, atom_func=gaussian, atom_size=0.1):
     #initialize parameters
     lvs_lengths = [get_dist([0,0],lvs[0,:]), get_dist([0,0],lvs[1,:])]
     square_size = max(lvs_lengths)*res*res_factor 
     square_size = int(square_size)
     pixel_size_rc = 1.0/square_size
     uc_square = np.zeros(shape=[square_size,square_size])
     #fill in square
     atom_list = make_atom_list(atom_range, lvs)
     for x in range(0, square_size):
         for y in range(0, square_size):
             point_val = 0
             rc_point = np.array([x*pixel_size_rc, y*pixel_size_rc])
             cc_point = rc_to_cc(rc_point, lvs)
             for atom in atom_list:
                 atom_dist = get_dist(cc_point, atom)
                 point_val += atom_func(atom_dist, atom_size)
                 uc_square[x,y] = point_val
     #normalises array
     uc_norm = norm(uc_square)
     return uc_norm

#makes lattice from lvs with one atom basis
#for each point in the lattice array, it converts the point to rc mod 1, 
#and find the appropriate value from uc_square 
#shifts lattice by shift, a vector in rc     
def make_sub_lattice(uc_square, lvs, res, size, shift):
    n_points = int(size*res)
    uc_dim = uc_square.shape[0]
    sub_lat_data = np.zeros(shape=[n_points,n_points])
    for i in range(0, n_points):
         for j in range(0, n_points):
             cc_point = [i/res - shift[0], j/res - shift[1]] 
             rc_point = cc_to_rc(cc_point, lvs, mod=0.999999999) #should be able to just be mod=1, not sure why not working
             uc_point = rc_point * uc_dim
             sub_lat_data[i,j] = uc_square[int(np.floor(uc_point[0])), int(np.floor(uc_point[1]))]
    return sub_lat_data

#plots data as a colour plot
#need each figure to have a different n-fig, otherwise issues arrise
def plot(data, size, n_fig):
    rot_data = np.rot90(np.fliplr(data)) 
    plot_extent = (0, size, 0, size)
    plt.figure(n_fig)
    plt.contourf(rot_data, cmap='magma', levels = 100, extent=plot_extent)
    plt.axis('scaled')
    cbar = plt.colorbar()
    plt.xlabel("x (nm)")
    plt.ylabel("y (nm)")
    cbar.set_label('', rotation=270)
    cbar.set_ticks([])
    plt.autoscale(enable=True)

#make a littcie with a basis
#a = lattice_vector_1 length
#b = lattice_vector_2 length    
#gamma_deg = angle between a and b in degrees
#theta_deg = twist of entire lattice in degrees
#basis = a 2D array of basis vectors
#trans = translation of entire lattice, [x,y]
#atom_sizes = a list of atoms sizes (atom gaussian std), need same number of entries as there are basis vectors
#size = size of lattice in nm
#res = number of pixels per nm
#plot_all = if true, plots UC square, sublattices, and final lattice    
def make_lattice(a, b, gamma_deg, theta_deg, basis, trans, atom_sizes, size, res, plot_all=False):
    n_points = int(size*res)
    lat_data = np.zeros(shape=[n_points,n_points])
    n_basis = basis.shape[0]
    for i in range(0, n_basis):
        lvs = make_lvs(a, b, gamma_deg, theta_deg)
        uc_square = make_uc_square(lvs, res, atom_range=2, res_factor=2, atom_func=gaussian, atom_size=atom_sizes[i])
        shift = trans
        for j in [0,1]:
            shift += basis[i,j] * lvs[j,:] 
        sub_lat = make_sub_lattice(uc_square, lvs, res, size, shift)
        lat_data += sub_lat 
        if plot_all == True:
            plot(uc_square, 1, i + 2) #trying to never have the same figure number
            plot(sub_lat, size, i + n_basis + 3)
    norm_lat_data = norm(lat_data)
    if plot_all == True:
        plot(norm_lat_data, size, 1)
    return norm_lat_data
    
#Example: Bilayer Graphene with a 15 deg twist
#Lattice 1
a = 0.246
b = 0.246
gamma_deg = 120
theta_1 = 0
trans_1 = [0,0]
basis = np.array([[0.,0.], [0.3333333333, 0.66666666667]])    
atom_sizes = [0.05, 0.05]
size = 4
res = 100
lat_1 = make_lattice(a, b, gamma_deg, theta_1, basis, trans_1, atom_sizes, size, res, plot_all=True)
#Lattice 2 (doesn't plot)
#Here we re-use some of our variables from the first lattice, you can only do this if the two lattices are the same
theta_2 = 15
trans_2 = [0,0]
lat_2 = make_lattice(a, b, gamma_deg, theta_2, basis, trans_2, atom_sizes, size, res, plot_all=False)
#Moire
moire = lat_1 + lat_2
plot(moire, size, 100)
