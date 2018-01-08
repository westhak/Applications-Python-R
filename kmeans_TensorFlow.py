# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:55:55 2017

Task: Implement kmeans using tensorflow
Data: synthetic data

@author: Swetha
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Total number of points in the dataset
number_of_points = 1000

# The dimension of each point. Since we will visualize the clustered points later, we set it to 2.
dimension = 2

number_of_clusters = 4
true_centroids = np.random.rand(number_of_clusters, dimension)
true_assignments = np.random.randint(0, number_of_clusters, number_of_points)
points_values = np.array([true_centroids[i] + np.random.normal(scale=0.07, size=dimension) for i in true_assignments])

# The maximum number of iterations in the k-means algorithm
maximum_number_of_steps = 1000

points = tf.zeros([number_of_points, dimension], tf.float32)
points = tf.constant(points_values,tf.float32)
centroids = tf.zeros([number_of_clusters, dimension])
centroids = tf.Variable(tf.random_crop(points, size=[number_of_clusters,dimension], seed=12345, name="centroids"),tf.float32)

distances = tf.zeros([number_of_clusters, number_of_points],tf.float32)

# Function for calculating distance
def calculate_distance():
    expanded_vectors = tf.expand_dims(points, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)
    distances = tf.reduce_sum( tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
    return distances


# Function for assignment to clusters

assignments = tf.zeros([number_of_points])
def assignment_clusters():
    assignments = tf.argmin(calculate_distance(),axis=0)
    assignments = tf.to_int32(assignments)
    return assignments

# Update centroids
new_centroids = tf.zeros([number_of_clusters, dimension])

def update_centroids():
    partitions = tf.dynamic_partition(points, assignment_clusters(), number_of_clusters)
    new_centroids = tf.reshape(tf.concat([tf.reduce_mean(partition, 0) for partition in partitions], 0),[number_of_clusters,dimension])
    return new_centroids

old_centroids = tf.Variable(tf.zeros([number_of_clusters,dimension]), dtype=tf.float32)
assign_opx = tf.assign(old_centroids,centroids)
assign_op = tf.assign(centroids,update_centroids())

# Function for calculating distance from centroids in each iteration
def dist_centroids_iteration():
        dist = tf.reduce_sum( tf.square(tf.subtract(centroids, old_centroids)), 1)
        dist = tf.reduce_sum(dist,0)   
        return dist

    
iterations=20
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
for i in range(iterations):
    expanded_vectors = tf.expand_dims(points, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)
    distances = tf.reduce_sum( tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)

    assignments = tf.argmin(distances,axis=0)
    assignments = tf.to_int32(assignments)
    
    partitions = tf.dynamic_partition(points, assignments, number_of_clusters)
    new_centroids = tf.reshape(tf.concat([tf.reduce_mean(partition, 0) for partition in partitions], 0),[number_of_clusters,dimension])
    
    assign_opx = tf.assign(old_centroids,centroids)
    sess.run(assign_opx)
    
    assign_op = tf.assign(centroids,new_centroids)
    sess.run(assign_op)
    
    dist = tf.reduce_sum( tf.square(tf.subtract(centroids, old_centroids)), 1)
    dist = tf.reduce_sum(dist,0)
    result = dist.eval()
    
    if result==0:
        print("Centroid value converged by kmeans ",centroids.eval())
        print("True Centroid value ", true_centroids)
        break

# Run TensorFlow session

iterations=20
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
for i in range(iterations):
    sess.run(assign_opx)
    sess.run(assign_op)
    result = sess.run( dist_centroids_iteration())
    print("Difference between new and old centroid, iter",i," : ",result)    
    if result==0:
        print("Centroid value converged by kmeans ",centroids.eval())
        print("True Centroid value ", true_centroids)
        break


# Visualize the results
 plt.cla()
    plot_data = points.eval()
    clusters = assignments.eval()
    plt.scatter(plot_data[:, 0], plot_data[:, 1], c=clusters, s=100, lw=0, cmap='RdYlGn')
        
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
plot_data = points_values
clusters = true_assignments
plt.scatter(plot_data[:, 0], plot_data[:, 1], c=clusters, s=100, lw=0, cmap='RdYlGn')
plt.show()

# Visualize the results

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

for i in range(iterations):
    sess.run(assign_opx)
    sess.run(assign_op)
    result = sess.run( dist_centroids_iteration())
    if result==0:
        plot_data = points.eval()
        clusters = assignments.eval()
        #print(clusters)
        plt.scatter(plot_data[:, 0], plot_data[:, 1], c=clusters, s=100, lw=0, cmap='RdYlGn')
        plt.show()
        break


