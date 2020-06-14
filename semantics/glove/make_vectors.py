import numpy as np
from scipy import spatial

def find_closest_embeddings(embedding):
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))


embeddings_dict = {}
with open("glove.6B.50d.txt", 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector
"""
class_dict={}
with open("../../data/txt/multi/labeled_source_images_real.txt", 'r') as f:
     for line in f:
         class_label = line.split('/')[1]
         if class_label not in class_dict:
             class_dict[class_label]=True
for label in class_dict.keys():
    print(label, label)
"""
class_dict={}
with open("classes.txt", 'r') as f:
     for line in f:
         og_label, fk_label = line.split(' ')[0], line.split(' ')[1].split('\n')[0]
         if og_label not in class_dict:
             class_dict[og_label]=fk_label
total=[]
for og_labels in class_dict.keys():
    if og_labels in embeddings_dict:
        total.append(embeddings_dict[og_labels]/np.linalg.norm(embeddings_dict[og_labels]))
        print("computed direct label")
    else:
        mini_labels=class_dict[og_labels].split('_')
        final_embedding=np.sum([embeddings_dict[xlabel] for xlabel in mini_labels],axis=0)
        nearest_label=find_closest_embeddings(final_embedding)[0]
        total.append(embeddings_dict[nearest_label]/np.linalg.norm(embeddings_dict[nearest_label]))
        print("computed nearest neighbour")
np.save("glove_features.npy",np.array(total))        
