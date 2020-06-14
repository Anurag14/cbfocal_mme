import numpy as np
import io
from scipy import spatial

def find_closest_embeddings(embedding):
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))
    return data

embeddings_dict = load_vectors('wiki-news-300d-1M.vec')

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
        # the fake label is being used to make mini labels fk_label is class_dict[og_labels]
        mini_labels=class_dict[og_labels].split('_')
        final_embedding=np.sum([embeddings_dict[xlabel] for xlabel in mini_labels],axis=0)
        nearest_label=find_closest_embeddings(final_embedding)[0]
        total.append(embeddings_dict[nearest_label]/np.linalg.norm(embeddings_dict[nearest_label]))
        print("computed nearest neighbour")
np.save("fastext_features.npy",np.array(total).astype('float32'))
