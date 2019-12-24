from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models


# general libs
import cv2
from PIL import Image
import numpy as np
import argparse
import os
import time
#
from model import RGMP
from utils import ToCudaVariable, ToLabel, DAVIS, upsample, downsample



### set pathes
DAVIS_ROOT = '../data/davis-2017/data/DAVIS/'
palette = Image.open(DAVIS_ROOT + 'Annotations/480p/bear/00000.png').getpalette()

def get_arguments():
    parser = argparse.ArgumentParser(description="RGMP")
    parser.add_argument("-MO", action="store_true", help="Multi-object")
    return parser.parse_args()
args = get_arguments()
MO = args.MO 

if MO:
    print('Multi-object VOS on DAVIS-2017 valildation')
else:
    print('Single-object VOS on DAVIS-2016 valildation')



def Encode_MS(val_F1, val_P1, scales):
    ref = {}
    for sc in scales:
        if sc != 1.0:
            msv_F1, msv_P1 = downsample([val_F1, val_P1], sc)
            msv_F1, msv_P1 = ToCudaVariable([msv_F1, msv_P1], volatile=True)
            ref[sc] = model.module.Encoder(msv_F1, msv_P1)[0]
        else:
            msv_F1, msv_P1 = ToCudaVariable([val_F1, val_P1], volatile=True)
            ref[sc] = model.module.Encoder(msv_F1, msv_P1)[0]

    return ref

def Propagate_MS(ref, val_F2, val_P2, scales):
    h, w = val_F2.size()[2], val_F2.size()[3]
    msv_E2 = {}
    for sc in scales:
        if sc != 1.0:
            msv_F2, msv_P2 = downsample([val_F2, val_P2], sc)
            msv_F2, msv_P2 = ToCudaVariable([msv_F2, msv_P2], volatile=True)
            r5, r4, r3, r2  = model.module.Encoder(msv_F2, msv_P2)
            e2 = model.module.Decoder(r5, ref[sc], r4, r3, r2)
            msv_E2[sc] = upsample(F.softmax(e2[0], dim=1)[:,1].data.cpu(), (h,w))
        else:
            msv_F2, msv_P2 = ToCudaVariable([val_F2, val_P2], volatile=True)
            r5, r4, r3, r2  = model.module.Encoder(msv_F2, msv_P2)
            e2 = model.module.Decoder(r5, ref[sc], r4, r3, r2)
            msv_E2[sc] = F.softmax(e2[0], dim=1)[:,1].data.cpu()

    val_E2 = torch.zeros(val_P2.size())
    for sc in scales:
        val_E2 += msv_E2[sc]
    val_E2 /= len(scales)
    return val_E2


def Infer_SO(all_F, all_M, num_frames, scales=[0.5, 0.75, 1.0]):
    all_E = torch.zeros(all_M.size())
    all_E[:,:,0] = all_M[:,:,0]

    ref = Encode_MS(all_F[:,:,0], all_E[:,0,0], scales)
    for f in range(0, num_frames-1):
        all_E[:,0,f+1] = Propagate_MS(ref, all_F[:,:,f+1], all_E[:,0,f], scales)

    return all_E



def Infer_MO(all_F, all_M, num_frames, num_objects, scales=[0.5, 0.75, 1.0]):
    if num_objects == 1:
        obj_E =  Infer_SO(all_F, all_M, num_frames, scales=scales) #1,1,t,h,w
        return torch.cat([1-obj_E, obj_E], dim=1)

    _, n, t, h, w = all_M.size()
    all_E = torch.zeros((1,n+1,t,h,w))
    all_E[:,1:,0] = all_M[:,:,0]
    all_E[:,0,0] = 1-torch.sum(all_M[:,:,0], dim=1)

    ref_bg = Encode_MS(all_F[:,:,0], torch.sum(all_E[:,1:,0], dim=1), scales)
    refs = []
    for o in range(num_objects):
        refs.append(Encode_MS(all_F[:,:,0], all_E[:,o+1,0], scales))

    for f in range(0, num_frames-1):
        ### 1 - all 
        all_E[:,0,f+1] = 1-Propagate_MS(ref_bg, all_F[:,:,f+1], torch.sum(all_E[:,1:,f], dim=1), scales)
        for o in range(num_objects):
            all_E[:,o+1,f+1] = Propagate_MS(refs[o], all_F[:,:,f+1], all_E[:,o+1,f], scales)

        # Normalize by softmax
        all_E[:,:,f+1] = torch.clamp(all_E[:,:,f+1], 1e-7, 1-1e-7)
        all_E[:,:,f+1] = torch.log((all_E[:,:,f+1] /(1-all_E[:,:,f+1])))
        all_E[:,:,f+1] = F.softmax(Variable(all_E[:,:,f+1]), dim=1).data

    return all_E



if MO:
    Testset = DAVIS(DAVIS_ROOT, imset='2017/val.txt', multi_object=True)
    Testloader = data.DataLoader(Testset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
else:
    Testset = DAVIS(DAVIS_ROOT, imset='2016/val.txt')
    Testloader = data.DataLoader(Testset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

model = nn.DataParallel(RGMP())
if torch.cuda.is_available():
    model.cuda()


model.load_state_dict(torch.load('weights.pth'))

model.eval() # turn-off BN
for seq, (all_F, all_M, info) in enumerate(Testloader):
    all_F, all_M = all_F[0], all_M[0]
    seq_name = info['name'][0]
    num_frames = info['num_frames'][0]
    num_objects = info['num_objects'][0]

    tt = time.time()
    all_E = Infer_MO(all_F, all_M, num_frames, num_objects, scales=[0.5, 0.75, 1.0])
    print('{} | num_objects: {}, FPS: {}'.format(seq_name, num_objects, num_frames /(time.time()-tt)))

    # Save results for quantitative eval ######################
    if MO:
        folder = 'results/MO'
    else:
        folder = 'results/SO'    
    test_path = os.path.join(folder, seq_name)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    for f in range(num_frames):
        E = all_E[0,:,f].numpy()
        # make hard label
        E = ToLabel(E)
    
        (lh,uh), (lw,uw) = info['pad'] 
        E = E[lh[0]:-uh[0], lw[0]:-uw[0]]

        img_E = Image.fromarray(E)
        img_E.putpalette(palette)
        img_E.save(os.path.join(test_path, '{:05d}.png'.format(f)))

