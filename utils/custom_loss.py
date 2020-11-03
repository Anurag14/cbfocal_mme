import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def upper_triangle(matrix):
    upper = torch.triu(matrix, diagonal=1)
    return upper

def regularizer(W):
    W = F.normalize(W, dim=1)
    embedding = torch.from_numpy(np.load('semantics/glove/glove_features.npy')).cuda()
    mc = W.shape[0]
    w_expand1 = W.unsqueeze(0)
    w_expand2 = W.unsqueeze(1)
    wx = (w_expand2 - w_expand1)**2
    w_norm_mat = torch.sum((w_expand2 - w_expand1)**2, dim=-1)
    w_norm_upper = upper_triangle(w_norm_mat)
    similarity = torch.mm(embedding, torch.transpose(embedding,0,1))
    dist = upper_triangle(similarity).clamp(min=0)
    mu = 2.0 / (mc**2 - mc) * torch.sum(w_norm_upper)
    #print("w_norm_mat: \n",w_norm_mat)
    #print("w_norm_upper: \n",w_norm_upper)
    #print("dist: \n",dist)
    #print("mu: \n",mu)
    residuals = upper_triangle((w_norm_upper - mu - dist)**2)
    rw = 2.0 / (mc**2 - mc) * torch.sum(residuals)
    return rw

if __name__ == "__main__":
    torch.manual_seed(0)
    filename = '../freezed_models/%s_%s_%s.ckpt.best.pth.tar' % ("MME","real","sketch")
    main_dict = torch.load(filename)
    W=F.normalize(main_dict['F1_state_dict']['fc2.weight'],dim=1)
    print(regularizer(W))
    W=torch.from_numpy(np.load('../semantics/glove/glove_features.npy')).cuda()
    print(regularizer(W))
