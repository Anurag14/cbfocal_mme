import torch
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

def do_kmeans(features,global_index,groundtruth):
    n = min(4,features.shape[0])
    kmeans = KMeans(n_clusters=n, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit_predict(features)
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_,features)
    """return kmeans.cluster_centers_, global_index[closest], [groundtruth for i in range(n)]"""
    return global_index[closest]

def cluster_the_unlabelled(features, groundtruth):
     class_wise_feature_dict={}
     class_wise_index_dict={}
     for index,label in enumerate(groundtruth):
         if label not in class_wise_feature_dict:
             class_wise_feature_dict[label]=[features[index]]
             class_wise_index_dict[label]=[index]
         else:
             class_wise_feature_dict[label].append(features[index])
             class_wise_index_dict[label].append(index)
     #centroid_features,centroid_global_indexes, centroid_labels=[],[],[]
     centroid_global_indexes=[]
     for label in class_wise_feature_dict.keys():
         indexes = do_kmeans(np.array(class_wise_feature_dict[label]),np.array(class_wise_index_dict[label]),label)
         centroid_global_indexes.extend(indexes)
         #for feature in features:
         #    centroid_features.append(feature)
         #centroid_features.extend(features)
         #centroid_labels.extend(labels)
     #print(np.array(centroid_features).shape)
     #centroid_features, centroid_labels = torch.Tensor(centroid_features).type(torch.FloatTensor),torch.Tensor(centroid_labels).type(torch.LongTensor)
     #centorid_labels = torch.zeros(len(centroid_labels), centroid_labels.max()+1).scatter_(1, centroid_labels.unsqueeze(1), 1.)
     #return centroid_features, centroid_labels, centroid_global_indexes
     return centroid_global_indexes
