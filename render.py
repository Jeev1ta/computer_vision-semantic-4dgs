#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_semantics
import torchvision
from PIL import Image
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
import threading
import concurrent.futures

SEMANTIC_COLORS = np.array([
    [0, 0, 0],        # 0: background
    [220, 20, 60],    # 1: crimson
    [0, 191, 255],    # 2: deep sky blue
    [50, 205, 50],    # 3: lime green
    [255, 165, 0],    # 4: orange
    [148, 0, 211],    # 5: dark violet
    [255, 215, 0],    # 6: gold
    [0, 250, 154],    # 7: medium spring green
    [255, 105, 180],  # 8: hot pink
    [70, 130, 180],   # 9: steel blue
], dtype=np.uint8)
def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, cam_type, render_semantics_flag=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_list = []
    render_list = []

    do_semantics = render_semantics_flag or gaussians.has_semantics
    sem_images = []
    combined_images = []
    if do_semantics:
        sem_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_semantics")
        makedirs(sem_render_path, exist_ok=True)

    print("point nums:", gaussians._xyz.shape[0])
    time1 = time()
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, cam_type=cam_type)["render"]
        rgb_8b = to8b(rendering).transpose(1, 2, 0)
        render_images.append(rgb_8b)
        render_list.append(rendering)

        if do_semantics:
            sem_rendering = render_semantics(view, gaussians, pipeline, cam_type=cam_type)
            if sem_rendering is not None:
                pred_label = sem_rendering.argmax(dim=0).cpu().numpy()
                c_idx = np.clip(pred_label, 0, len(SEMANTIC_COLORS) - 1)
                colored_sem = SEMANTIC_COLORS[c_idx]
                sem_images.append(colored_sem)
                combined = np.concatenate([rgb_8b, colored_sem], axis=1)
                combined_images.append(combined)
                # Save both colored visualization and raw 1-channel label map for eval_semantics.py
                Image.fromarray(colored_sem).save(os.path.join(sem_render_path, f"{idx:05d}_color.png"))
                Image.fromarray(pred_label.astype(np.uint8), mode="L").save(os.path.join(sem_render_path, f"{idx:05d}.png"))

        if name in ["train", "test"]:
            if cam_type != "PanopticSports":
                gt = view.original_image[0:3, :, :]
            else:
                gt  = view['image'].cuda()
            gt_list.append(gt)

    time2 = time()
    if len(views) > 1:
        print("FPS:", (len(views)-1)/(time2-time1))

    multithread_write(gt_list, gts_path)
    multithread_write(render_list, render_path)

    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_images, fps=30)
    if do_semantics and len(sem_images) > 0:
        imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_semantics.mp4'), sem_images, fps=30)
        imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb_semantics.mp4'), combined_images, fps=30)
        print("✅ Saved video_semantics.mp4 and video_rgb_semantics.mp4")

def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool, render_semantics_flag: bool = False):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cam_type=scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, cam_type, render_semantics_flag)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, cam_type, render_semantics_flag)
        if not skip_video:
            render_set(dataset.model_path, "video", scene.loaded_iter, scene.getVideoCameras(), gaussians, pipeline, background, cam_type, render_semantics_flag)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--render_semantics", action="store_true", default=False, help="Render semantic segmentation maps and videos")
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video, getattr(args, 'render_semantics', False))