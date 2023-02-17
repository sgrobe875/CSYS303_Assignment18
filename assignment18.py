# Sarah Grobe
# POCS HW 18


# imports
import matplotlib.pyplot as plt
import math
import numpy as np
import random
from scipy.optimize import minimize




# Basic methodology: function which takes in n
# Use n to determine the bounds of the hexagon (n on each side)
# --> maybe find the center and work back from there?
# --> find radius and the vertices of the hexagon
# working up from the bottom, determine algorithm for number of points in each row
# assign coordinates to each node/point 
# build the adjacency list
# --> neighbor = any node within distance close to 1 of the given node (went with dist < 1.2)
# --> for now/to get the required output, assume connection b/w all neighbors




# set globals
i_0 = 1000
gamma = 1/2
GAMMA = (2*gamma) / (gamma+1)




# takes in the node of interest and the total number of nodes in the lattice and 
# returns the current calculation as given in the instructions
def sink_current(N):
    return (-1*i_0)/(N-1)


# converts coordinates from string format 'x,y' to float(x) and float(y) 
# as separate variables
def get_float_coords(coord_string):
    coord = coord_string.split(',')
    x = float(coord[0])
    y = float(coord[1])
    return x, y


# calculates Euclidean distance between two coordinates in string format
def calc_distance(coord1_string, coord2_string):
    x1, y1 = get_float_coords(coord1_string)
    x2, y2 = get_float_coords(coord2_string)
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


# takes in the edge list, returns K according to the randomly set conductances
def calc_K(k_matrix):
    k_mat_copy = k_matrix.copy()
    for i in range(len(k_matrix)):
        for j in range(len(k_matrix)):
            k_mat_copy.itemset((i,j), k_matrix[(i,j)]**gamma)
    # ks_to_gamma_power = np.matrix(map(lambda x:pow(x,gamma),k_matrix))
    sum_of_ks = np.sum(k_mat_copy)#/2
    return sum_of_ks ** (1/gamma)


# takes in edge width n, draws lattice of the corresponding size and returns the
# adjacency list of all of the nodes within the lattice
def build_lattice(n): 
    # calculate some parameters based on n
    radius = n-1
    half_radius = radius/2

    
    # assume the center is the origin (0,0)
    
    # start with finding vertices (start at lower left and go counter-clockwise)
    v1 = {'x': -1 * half_radius, 
          'y': -1 * radius}
    v2 = {'x': half_radius, 
          'y': -1 * radius}
    v3 = {'x': radius, 
          'y': 0}
    v4 = {'x': half_radius, 
          'y': radius}
    v5 = {'x': -1 * half_radius, 
          'y': radius}
    v6 = {'x': -1 * radius, 
          'y': 0}
    
    
    # use vertex coordinates to determine bounds (starting and ending points)
    start_x = v1['x']
    start_y = v1['y']
    end_x = v2['x']
    
    # build dictionary to hold non-vertex points
    # key = tuple of (x,y) coordinates of the point
    # value = adjacency list (empty set for now)
    
    points = dict()
    
    
    # starting with the bottom half of the hexagon
    x = start_x
    y = start_y
    
    
    # do the bottom half + middle row
    while y != 1:
        # start at vertex1 and add to the points dictionary
        x = start_x
        key = str(x) + ',' + str(y)
        points[key] = {'neighbors':set()}
        
        # while we're within the bounds of this row
        while x != end_x:
            # add points, moving across the row from left to right
            x += 1
            key = str(x) + ',' + str(y)
            points[key] = {'neighbors':set()}
        
        # when we've reached the end of the row, move up 1 to the next row
        y += 1
        # widen the row by 1 each time we move up
        start_x -= 0.5
        end_x += 0.5
    
    
    # repeat for the top half (same as above but with some flipped signs)
    
    # start at vertex5 (top left)
    start_x = v5['x']
    start_y = v5['y']
    end_x = v4['x']
    
    x = start_x
    y = start_y
    
    # while we're in the top half of the hexagon
    while y != 0:
        # add the first point in the row to the points dict
        x = start_x
        key = str(x) + ',' + str(y)
        points[key] = {'neighbors':set()}
        
        # move across the row within the bounds of the hexagon
        while x != end_x:
            x += 1
            key = str(x) + ',' + str(y)
            points[key] = {'neighbors':set()}
            
        # move down to the next row
        y -= 1
        # widen the row by 1
        start_x -= 0.5
        end_x += 0.5
    
    
    
    # set neighbors of each point
    for point1 in points:
        # loop through all points
        for point2 in points:
            if point1 != point2:
                # calculate the distance between the two points
                dist = calc_distance(point1, point2)
                # if the two points are close enough, they are neighbors, so add to 
                # the neighbor list for point1 and for point2
                if dist < 1.2:
                    # since neighbor list is a set, duplicates will just be skipped
                    points[point1]['neighbors'].add(point2)
                    points[point2]['neighbors'].add(point1)

                    
                   
    # build the adjacency matrix and k matrix
    
    total_points = len(points.keys())
    
    # initialize matrix of all zeros
    adj_matrix = np.zeros((total_points, total_points))
    k_matrix = np.zeros((total_points, total_points))
    
    # list of all points
    points_list = list(points.keys())
    
    # loop through all points --> point_i
    for i in range(len(points_list)):
        # get neighborhood of point i
        neighbors_i = points[points_list[i]]['neighbors']
        # loop through all points --> point_j
        for j in range(len(points_list)):
            point_j = points_list[j]
            # if point_j connected to point_i (in this case, if they're neighbors),
            # set i,j in the adjacency matrix to 1
            if point_j in neighbors_i:
                adj_matrix.itemset((i,j), 1)
                
                if k_matrix[(i,j)] == 0 and k_matrix[(j,i)] == 0:
                    k_val = random.uniform(0,1)
                    k_matrix.itemset((i,j), k_val)
                    k_matrix.itemset((j,i), k_val)

    
    
    # # finally, plot
    # fig = plt.figure()
    # ax = fig.add_subplot()
        
    # # plot lines between all neighbors
    # for point in list(points.keys()):
    #     # start by adding points of the current node
    #     point_x, point_y = get_float_coords(point)
    #     x = [point_x]
    #     y = [point_y]
        
    #     # loop through all neighbors of that node
    #     neighbors = points[point]['neighbors']
    #     for neighbor in neighbors:
    #         x_n, y_n = get_float_coords(neighbor)
            
    #         # add coordinates of the neighbor, then re-add the points of the
    #         # original node
    #         x.append(x_n)
    #         x.append(point_x)
    #         y.append(y_n)
    #         y.append(point_y)
        
    #     # plot and move on to the next point
    #     # scale point size and line width based on n; make points a different color
    #     plt.plot(x, y, c = 'black', marker=".", mfc='blue', mec='blue', 
    #              markersize=40/n, linewidth = 5/n)
        

    # # make the figure a square        
    # ax.set_aspect('equal', adjustable='box')
    
    # plt.title('Full Lattice')
    
    # plt.show()
    
    # return the list of points and their info (points), 
    # edge list & their conductance (edges), and the adjacency matrix
    return points, adj_matrix, k_matrix


# draw the lattice, get the network data, and get the adjacency matrix
network, adj_mat, k_mat = build_lattice(n = 8)

# get the value of K
K = calc_K(k_mat)

k_mat = np.divide(k_mat,K)

# list of points; starts with the source!
points = list(network.keys())
    

# build I vector
I = [i_0]
for i in range(1, len(adj_mat)):
    I.append(sink_current(len(adj_mat)))

    
    
Cs = []
for j in range(50):
        # build lambda vector
        lambda_v = []
        for i in range(len(k_mat)):
            lambda_v.append(np.sum(k_mat[i]))
        
        
        # build LAMBDA matrix
        LAMBDA_m = np.zeros((len(adj_mat), len(adj_mat)))
        for i in range(len(lambda_v)):
            LAMBDA_m.itemset((i,i), lambda_v[i])
        
        
        
        # build U vector
        step1 = np.subtract(LAMBDA_m, k_mat)
        # U = np.matrix(np.matmul(np.linalg.inv(step1), I))
        U = np.linalg.solve(step1, I)  
        
        
        # build delta_U
        delta_U = np.subtract.outer(U.T, U)
        
        
        # build I_new
        I_new = np.multiply(k_mat, delta_U)
        
        
        
        # calculate C
        C = np.sum(np.power(np.abs(I_new), GAMMA))
    
        Cs.append(C)
        
        
        # calculate k_new
        k_new = np.power(np.abs(I_new), (-1*(GAMMA - 2)))
        
        # K = calc_K(k_new)
        
        # finally, overwrite k
        k_mat = np.divide(k_new,K)
        
        # I = I_new
        
        
        
        # U = np.linalg.solve(LAMBDA_minus_k, I)         
        # delta_U = np.subtract.outer(np.transpose(U),U)    
        # I_new = k*delta_U
        # k_new = np.abs(I_new)**(-(GAMMA-2))
        # #k = k_new
        # k = k_new/K
        
result = k_mat
result = result/np.max(result)
    
# def optimization(I,k):
#     for j in range(3):
#         # K = calc_K(k)
#         lamb = np.sum(k,axis=0) # GOOD
#        # print(lamb.shape)
#         LAMBDA = np.diag(lamb) # GOOD
#         LAMBDA_minus_k = np.subtract(LAMBDA,k) 
#         U = np.linalg.solve(LAMBDA_minus_k, I)         
#         delta_U = np.subtract.outer(np.transpose(U),U)    
#         I_new = k*delta_U
#         k_new = np.abs(I_new)**(-(GAMMA-2))
#         #k = k_new
#         k = k_new/K
#     return k
        

# result = optimization(I,k_mat)


def clean_noise(x):
    x = x*10
    for i in range(len(x)):
        for j in range(len(x)):
            if x[(i,j)] < 0.2:
                x.itemset((i,j), 0)
                
    return x
    
# clean_noise_v = np.vectorize(clean_noise)

result = clean_noise(result)


def draw_weighted(point_names, network, result, gamma):
    n = len(points)
    
    # finally, plot
    fig = plt.figure()
    ax = fig.add_subplot()
    
    
    # loop through each point and plot to its neighbors according to result
    for point1 in point_names:
        neighbors = network[point1]['neighbors']
        for point2 in neighbors:
            i = point_names.index(point1)
            j = point_names.index(point2)
            x_p1, y_p1 = get_float_coords(point1)
            x_p2, y_p2 = get_float_coords(point2)
            x = [x_p1, x_p2]
            y = [y_p1, y_p2]
            plt.plot(x,y, c='black', linewidth = result[(i,j)])
            

    # make the figure a square        
    ax.set_aspect('equal', adjustable='box')
    
    plt.title('Gamma = ' + str(gamma))
    
    plt.show()
    

draw_weighted(points, network, result, gamma)

