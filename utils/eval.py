from __future__ import print_function


import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from utils.return_dataset import return_dataset_test, return_dataset_train_eval
from utils.ldam import cb_focal_loss

def set_initial_transfer_source_weights(args, initial_value=1.0):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = os.path.join(base_path,'labeled_source_images_' + args.source + '.txt')
    image_set_file_t = os.path.join(base_path,'labeled_target_images_' + args.target + '_%d.txt' % (args.num))
    image_index=[]
    with open(image_set_file_s) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]
    with open(image_set_file_t) as f:
        image_index.extend([x.split(' ')[0] for x in f.readlines()])
    weight_file={path : initial_value for path in image_index}
    return weight_file 

def eval_source(G,F1, args, running_weights, tolerance):
    target_loader = return_dataset_train_eval(args)
    im_data_s = torch.FloatTensor(1)
    gt_labels_s = torch.LongTensor(1)

    im_data_s = im_data_s.cuda()
    gt_labels_s = gt_labels_s.cuda()

    im_data_s = Variable(im_data_s)
    gt_labels_s = Variable(gt_labels_s)
    G.eval()
    F1.eval()

    global_paths=[]
    global_pred1=[]
    global_cosine_sim=[]
    with torch.no_grad():
        for batch_idx, data_s in enumerate(target_loader):
            im_data_s.data.resize_(data_s[0].size()).copy_(data_s[0])
            gt_labels_s.data.resize_(data_s[1].size()).copy_(data_s[1])
            paths = list(data_s[2])
            feat = G(im_data_s)
            if "resnet" in args.net:
                output1 = F1.fc1(feat)
            output1 = F1.extrafc(output1)
            output1 = F.normalize(output1, dim=1)
            output1 = output1.mm(torch.transpose(F.normalize(F1.fc2.weight, dim=1),0,1))
            cosine_sim = output1.data.max(1)[0]
            global_pred1.extend(gt_labels_s)
            global_cosine_sim.extend(cosine_sim)
            global_paths.extend(paths)
            
    class_wise_sim_path={}
    global_weights=running_weights
    for i, pred1 in enumerate(global_pred1):
        if str(pred1.item()) not in class_wise_sim_path:
            class_wise_sim_path[str(pred1.item())]=[[global_cosine_sim[i].item()],[global_paths[i]]]
        else:
            class_wise_sim_path[str(pred1.item())][0].append(global_cosine_sim[i].item()) #append the cosine similarity
            class_wise_sim_path[str(pred1.item())][1].append(global_paths[i])                 #append the path
    for pred in class_wise_sim_path.keys():
        sorted_paths=[path for _,path in sorted(zip(class_wise_sim_path[pred][0],class_wise_sim_path[pred][1]))] # zip cosine sim and paths and sort them wrt cosine sim
        top_sorted_paths=sorted_paths[:-3] # take closest 3
        for top_sorted_path in top_sorted_paths:
            global_weights[top_sorted_path]+=1.2/float(tolerance)

    return global_weights

def transfer_source_weights_beta(args, step=0, G=None, F1=None, weight_file=None):
    if not hasattr(transfer_source_weights, "running_weights"):
        transfer_source_weights.running_weights = set_initial_transfer_source_weights(args, initial_value=1.0)
    if step == 0:
        ## take args and load files and create the weight file
        return set_initial_transfer_source_weights(args, initial_value=1.0)
    else:
        transfer_source_weights.running_weights = eval_source(G, F1, args, weight_file, tolerance=1.0)
        if step > 3:
            print("we are in ....")
            weight_file = {key: transfer_source_weights.running_weights[key]/float(step+1) for key in transfer_source_weights.running_weights.keys()}
        return weight_file

def transfer_source_weights(args, step=0, G=None, F1=None, weight_file=None, tolerance=10, repeat_interval=5000):
    if not hasattr(transfer_source_weights, "running_weights"):
        transfer_source_weights.running_weights = set_initial_transfer_source_weights(args, initial_value=1.0/float(tolerance)) 
    if step == 0:
        ## take args and load files and create the weight file
        return set_initial_transfer_source_weights(args, initial_value=1.0)        
        
    elif repeat_interval - step%repeat_interval > tolerance:
        ## send the same weight file do nothing 
        return weight_file 
    else:
        ## keep track of average of all files and return the upadated weight file in last iteration
        if repeat_interval - step%repeat_interval == tolerance:
            transfer_source_weights.running_weights = set_initial_transfer_source_weights(args, initial_value=1.0/float(tolerance)) # flush previous running weights 
        transfer_source_weights.running_weights = eval_source(G, F1, args, transfer_source_weights.running_weights, tolerance) 
        if step%repeat_interval == repeat_interval:
           weight_file = transfer_source_weights.running_weights
        return weight_file 
  
def eval_inference(G, F1, class_list, class_num_list, args):
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
    global_pred1=[]
    global_cosine_sim=[]
    with torch.no_grad():
        for batch_idx, data_t in enumerate(target_loader_unl):
            im_data_t.data.resize_(data_t[0].size()).copy_(data_t[0])
            gt_labels_t.data.resize_(data_t[1].size()).copy_(data_t[1])
            paths = list(data_t[2])
            feat = G(im_data_t)
            if "resnet" in args.net:
                output1 = F1.fc1(feat)
            else:
                output1 = feat
            output1 = F1.extrafc(output1)
            output1 = F.normalize(output1, dim=1)
            output1 = output1.mm(torch.transpose(F.normalize(F1.fc2.weight, dim=1),0,1))
            size += im_data_t.size(0)
            cosine_sim = output1.data.max(1)[0]
            pred1 = output1.data.max(1)[1]
            global_pred1.extend(pred1)
            global_cosine_sim.extend(cosine_sim)
            global_paths.extend(paths)
            for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_t.data).cpu().sum()
            test_loss += criterion(output1, gt_labels_t) / len(target_loader_unl)
    print('\nTest set: Average loss: {:.4f}, '
          'Accuracy: {}/{} F1 ({:.4f}%)\n'.
          format(test_loss, correct, size,
                 100. * float(correct) / size))
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
    
    paths_to_weights={}
    for i, path in enumerate(global_paths):
        paths_to_weights[path]=global_weights[i]
    return test_loss.data, 100. * float(correct) / size, paths_to_weights

def eval_inference_simple(G, F1, class_list, class_num_list, args,  alpha=1, gamma=1):
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
            paths = list(data_t[2])
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
    
    paths_to_weights={}
    for i, path in enumerate(global_paths):
        paths_to_weights[path] = alpha*(1-global_cosine_sim[i])**gamma + 1.0

    return test_loss.data, 100. * float(correct) / size, paths_to_weights
