#!/usr/bin/env python
# -*- coding: utf-8 -*- 
#
# Python KMean Clustering and Plotting.
# Code source: Ravish Gupta, Purdue University.
# License: BSD 3 clause.
#
#
#from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
#import csv
import sklearn.cluster
from sklearn.manifold import TSNE
import centroid_words as cw
import sys
import os
reload(sys)
sys.setdefaultencoding('utf-8')

######### ######### ######### ######### #########
# Numpy Array from CSV Data File
######### ######### ######### ######### #########
input_file = "both_vectors_4000.csv"
labels = np.genfromtxt(input_file, delimiter=',', usecols=0, dtype=str)
data = np.genfromtxt(input_file, delimiter=',')[:,1:-1]

# Find these words average them and create a centroid.
inputs = dict(zip(labels, data))

######### ######### ######### ######### #########
# Mean of Keywords for Centroids.
######### ######### ######### ######### #########
# Get all cluster names.
clusters = cw.keywords
cluster_count = len(clusters.keys())
initial_centroids = []
words_to_plot = []
# For each cluster name.
for cluster_name in clusters.keys():
  item_vector = []
  print cluster_name
  # For each item_label in cluster.
  for item_label in clusters[cluster_name]:
    if item_label in inputs.keys():
      item_vector.append(inputs[item_label])
      words_to_plot.append(item_label)
  # Converting to numpy array for parallel vector mean calculation.
  item_vector_array = np.array(item_vector ,np.float64)
  # calculating 128D mean for each Cluster - Centroid per Cluster.
  vector_mean = np.ndarray.mean(item_vector_array, axis=0)
  initial_centroids.append(vector_mean)
np_initial_centroids = np.array(initial_centroids,np.float64)

######### ######### ######### ######### #########
# Instantiate K-Mean Clustering.
######### ######### ######### ######### #########
km = KMeans( n_clusters=cluster_count,
             init=np_initial_centroids, #'random',
             n_init=1, #10,
             max_iter=200,
             tol=1e-04,
             random_state=1)

# Do K-Mean on data.
y_km = km.fit_predict(data)
# Convert clusters output to numpy array.
np_y_km = np.array(y_km, np.int8)
# Join Labels, Data and Clusters.
lbl_vec_cluster = np.column_stack((labels, data, np_y_km))
# Various Style Output Slicing.
# labels_data = lbl_vec_cluster[0:,:-1]
# data_cluster = lbl_vec_cluster[0:,1:]
labels_cluster = lbl_vec_cluster[0:,[0,-1]]

######### ######### ######### ######### #########
# Saving Results
######### ######### ######### ######### #########
# Used by TSNE and SaveText.
dim=3
num_points = len(labels)
# Write Results to csv
np.savetxt('08_tagger_clusters' + str(num_points) + '_'+ str(cluster_count) +'.csv', labels_cluster, delimiter=",", fmt="%s")

######### ######### ######### ######### #########
# TSNE - PCA
######### ######### ######### ######### #########
# Write 2-3D Embeddin with class.
tsne = TSNE(perplexity=30, n_components=dim, init='pca', n_iter=5000)
pca_embeddings = np.array(tsne.fit_transform(data[0:num_points+1, :]))
# PCA cluster centers
cluster_centers_val = np.array(tsne.fit_transform(np_initial_centroids[0:cluster_count+1, :]))

######### ######### ######### ######### #########
# Plotting
######### ######### ######### ######### #########
# Plotting Labels.
assert pca_embeddings.shape[0] >= len(labels), 'Labels > Embeddings'
plt.figure(figsize=(15,15))  # in inches
for i, label in enumerate(labels):
  # Only for words in keywords
  if label in words_to_plot:
    x, y = pca_embeddings[i,[0,1]]
    print x,y
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

# Plotting Scatter.
markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']
color = ['magenta', 'lightblue', 'orange', 'lightgreen', 'red', 'black', 'blue', 'yellow', 'brown', 'green', "cyan"]
for key, cluster_name in enumerate(clusters.keys()):
  plt.scatter(pca_embeddings[np_y_km==key,0], # second 0 says take only x values of row.
              pca_embeddings[np_y_km==key,1], # first 0 says select 0th column value where np_y_km val=0
              s=50,
              c=color[key],
              marker=markers[key],
              label=cluster_name)

# Plotting PCA Centroids 
plt.scatter(cluster_centers_val[:,0],
            cluster_centers_val[:,1],
            s=250,
            marker='*',
            c='red',
            label='centroids')

plt.legend()
plt.grid()
plt.show()

######### ######### ######### ######### #########
print "<<<<<<DONE>>>>>>"
exit()
