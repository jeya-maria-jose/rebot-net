import argparse
import cv2
import glob
import os
import torch
import requests
import numpy as np
from os import path as osp
from models.archs import rebotnet as net
from pthflops import count_ops
from ptflops import get_model_complexity_info


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='rebot_M', help='Path to option JSON file.')
    args = parser.parse_args()


    if args.opt=="rebot_M":
        model = net(upscale=1, img_size=[2,384,384], window_size=[6,8,8], depths=[3, 3, 3, 3],
                        indep_reconsts=[9,10], embed_dims=[64, 80, 108, 116],
                        num_heads=[6,6,6,6,6,6,6, 6,6,6,6, 6,6], pa_frames=2, deformable_groups=12, mlp_dim =256,bottle_depth=4,bottle_dim=116,dropout=0.1,patch_size=1).cuda()
    elif args.opt=="rebot_S":
        model = net(upscale=1, img_size=[2,384,384], window_size=[6,8,8], depths=[4, 4, 4, 5],
                        indep_reconsts=[9,10], embed_dims=[28, 36, 48, 64],
                        num_heads=[6,6,6,6,6,6,6, 6,6,6,6, 6,6], pa_frames=2, deformable_groups=12, mlp_dim =256,bottle_depth=4,bottle_dim=64,dropout=0.1,patch_size=1).cuda()
    elif args.opt=="rebot_L":
        model = net(upscale=1, img_size=[2,384,384], window_size=[6,8,8], depths=[5, 5, 5, 5],
                        indep_reconsts=[9,10], embed_dims=[128, 224, 280, 320],
                        num_heads=[6,6,6,6,6,6,6, 6,6,6,6, 6,6], pa_frames=2, deformable_groups=12, mlp_dim =512,bottle_depth=4,bottle_dim=320,dropout=0.1,patch_size=1).cuda()
    
    elif args.opt=="rebot_XS":
        model = net(upscale=1, img_size=[2,384,384], window_size=[6,8,8], depths=[4, 3, 3, 4],
                        indep_reconsts=[9,10], embed_dims=[8, 12, 20, 32],
                        num_heads=[6,6,6,6,6,6,6, 6,6,6,6, 6,6], pa_frames=2, deformable_groups=12, mlp_dim =64,bottle_depth=4,bottle_dim=32,dropout=0.1,patch_size=1).cuda()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 1000
    timings=np.zeros((repetitions,1))
    torch.manual_seed(0)
    dummy_input = torch.randn([1,3,384,384]).cuda()

    #GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn, std_syn)

if __name__ == '__main__':
    main()