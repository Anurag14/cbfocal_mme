from __future__ import print_function


import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from utils.return_dataset import return_dataset_test
from utils.ldam import cb_focal_loss

def load_weight_file(args):
    output_file="entropy_weight_files/%s_%s_%s_%s.txt" % (args.method, args.net, args.source, args.target)
    """ make a dict from paths to weights to run code faster"""
    paths_to_weights={}
    weight_file= open(output_file,"r")
    wf = weight_file.read().splitlines()
    for line in wf:
        paths_to_weights[line.split(" ")[-1]]=line.split(" ")[0]
    weight_file.close()
    return paths_to_weights
  
def eval_inference_simple(G, F1, class_list, class_num_list, args):
    output_file="entropy_weight_files/%s_%s_%s_%s.txt" % (args.method, args.net, args.source, args.target)
    target_loader_unl, class_list = return_dataset_test(args)
    im_data_t = torch.FloatTensor(1)
    gt_labels_t = torch.LongTensor(1)

    im_data_t = im_data_t.cuda()
    gt_labels_t = gt_labels_t.cuda()

    im_data_t = Variable(im_data_t)
    gt_labels_t = Variable(gt_labels_t)
    G.eval()
    F1.eval()
    size = 0
    global_paths=[]
    global_pred1=[]
    global_cosine_sim=[]
    with torch.no_grad():
        for batch_idx, data_t in enumerate(target_loader_unl):
            im_data_t.data.resize_(data_t[0].size()).copy_(data_t[0])
            gt_labels_t.data.resize_(data_t[1].size()).copy_(data_t[1])
            paths = data_t[2]
            feat = G(im_data_t)
            if "resnet" in args.net:
                output1 = F1.fc1(feat)
            output1 = F1.extrafc(output1)
            output1 = F.normalize(output1, dim=1)
            output1 = output1.mm(torch.transpose(F.normalize(F1.fc2.weight, dim=1),0,1))
            size += im_data_t.size(0)
            cosine_sim = output1.data.max(1)[0]
            pred1 = output1.data.max(1)[1]
            global_pred1.extend(pred1)
            global_cosine_sim.extend(cosine_sim)
            global_paths.extend(paths)
    class_wise_sim_path={}
    global_weights=[1 for _ in range(len(global_paths))]
    for i, pred1 in enumerate(global_pred1):
        if str(pred1.item()) not in class_wise_sim_path:
            class_wise_sim_path[str(pred1.item())]=[[global_cosine_sim[i].item()],[global_paths[i]]]
        else:
            class_wise_sim_path[str(pred1.item())][0].append(global_cosine_sim[i].item()) #append the cosine similarity
            class_wise_sim_path[str(pred1.item())][1].append(global_paths[i])                 #append the path
    for pred in class_wise_sim_path.keys():
        sorted_paths=[path for _,path in sorted(zip(class_wise_sim_path[pred][0],class_wise_sim_path[pred][1]))] # zip cosine sim and paths and sort them wrt cosine sim
        top_sorted_paths=sorted_paths[:int(0.1*len(sorted_paths))] # take top 10 percentile paths
        for top_sorted_path in top_sorted_paths:
            global_weights[global_paths.index(top_sorted_path)]=1.2

    with open(output_file, "w") as f:
        for i, path in enumerate(global_paths):
            f.write("%f %s\n" % (global_weights[i], path))
    return

def eval_inference(G, F1, class_list, class_num_list, args,  alpha=2, gamma=1):
    output_file="entropy_weight_files/%s_%s_%s_%s.txt" % (args.method, args.net, args.source, args.target)
    target_loader_unl, class_list = return_dataset_test(args)
    im_data_t = torch.FloatTensor(1)
    gt_labels_t = torch.LongTensor(1)

    im_data_t = im_data_t.cuda()
    gt_labels_t = gt_labels_t.cuda()

    im_data_t = Variable(im_data_t)
    gt_labels_t = Variable(gt_labels_t)
    G.eval()
    F1.eval()
    size = 0
    test_loss = 0
    correct = 0
    
    num_class = len(class_list)
    output_all = np.zeros((0, num_class))
    criterion = cb_focal_loss(class_num_list)
    confusion_matrix = torch.zeros(num_class, num_class) 
    
    global_paths=[]
    global_cosine_sim=[]
    
    with torch.no_grad():
        for batch_idx, data_t in enumerate(target_loader_unl):
            im_data_t.data.resize_(data_t[0].size()).copy_(data_t[0])
            gt_labels_t.data.resize_(data_t[1].size()).copy_(data_t[1])
            paths = data_t[2]
            feat = G(im_data_t)
            if "resnet" in args.net:
                output1 = F1.fc1(feat)
            output1 = F1.extrafc(output1)
            output1 = F.normalize(output1, dim=1)
            output1 = output1.mm(torch.transpose(F.normalize(F1.fc2.weight, dim=1),0,1))
            size += im_data_t.size(0)
            cosine_sim = output1.data.max(1)[0]
            global_cosine_sim.extend(cosine_sim)
            global_paths.extend(paths)
            # computing accuracy 
            pred1 = output1.data.max(1)[1]
            for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_t.data).cpu().sum()
            test_loss += criterion(output1, gt_labels_t) / len(target_loader_unl)
    print('\nTest set: Average loss: {:.4f}, '
          'Accuracy: {}/{} F1 ({:.4f}%)\n'.
          format(test_loss, correct, size,
                 100. * float(correct) / size))

    with open(output_file, "w") as f:
        for i, path in enumerate(global_paths):
            f.write("%f %s\n" % (alpha*(1-global_cosine_sim[i])**gamma, path))
    return test_loss.data, 100. * float(correct) / size
