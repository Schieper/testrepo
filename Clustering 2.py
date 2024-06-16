# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:43:24 2024

@author: BerndWagner
"""

import pandas as pd 
import numpy as np
#import seaborn as sns
#from scipy import stats
import matplotlib.pyplot as plt
#from kneed import KneeLocator
#from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_score
#from sklearn.preprocessing import StandardScaler

def KMeanCluster(df):
    #instantiate the k-means class, using optimal number of clusters
    kmeans = KMeans(init="random", n_clusters=7, n_init=10, random_state=1)
    #fit k-means algorithm to data
    kmeans.fit(df)

    #view cluster assignments for each observation
    labels=kmeans.labels_

    #plt.plot(labels)

    df['cluster'] = kmeans.labels_

#cluster hdbscan
#clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
#clusterer.fit(df)
#labels2=clusterer.labels_

#df['cluster2'] = clusterer.labels_
#df_clean_correlation_matrix = corr_matrix.corr()

KMeanCluster(df)

