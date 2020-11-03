import os
import torch
import torch.nn as nn
import shutil


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)


def save_checkpoint(state, is_best, checkpoint='checkpoint',
                    filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))
def save_mymodel(args, state, is_best):

    filename = '%s/%s_%s_%s.ckpt.pth.tar' % (args.checkpath, args.method, args.source, args.target)
    torch.save(state, filename)
    bestfilename = filename.replace('pth.tar', 'best.pth.tar')
    if is_best:
        if os.path.exists(bestfilename):
            existing_bestfile = torch.load(bestfilename)
            if state['best_acc_test'] > existing_bestfile['best_acc_test']:
                shutil.copyfile(filename, bestfilename)
                return
        else:
            shutil.copyfile(filename, bestfilename) 
