import os
import numpy as np
import argparse
from tqdm import tqdm
import imageio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_wild import read_input
from lib.utils.vismo import render_and_save

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pose3d/MB_ft_h36m_global_lite.yaml", help="Path to the config file.")
    parser.add_argument('-e', '--evaluate', default='checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-j', '--json_path', type=str, help='2D detection result json path')
    parser.add_argument('-v', '--vid_path', type=str, help='video path')
    parser.add_argument('-o', '--out_path', type=str, help='output path')
    parser.add_argument('--pixel', action='store_true', help='align with pixle coordinates')
    parser.add_argument('--focus', type=int, default=None, help='target person id')
    parser.add_argument('--clip_len', type=int, default=243, help='clip length for network input')
    parser.add_argument('--skip_trans', action='store_true', help='skip format transformation, use H36M format directly')
    parser.add_argument('--save_format', type=str, default='npy', choices=['npy', 'csv', 'json', 'npz', 'all'], help='format to save 3D pose results')
    parser.add_argument('--skip_render', action='store_true', help='skip 3D rendering and video generation, only save prediction results')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for processing multiple clips simultaneously')
    opts = parser.parse_args()
    return opts

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    
    opts = parse_args()
    args = get_config(opts.config)

    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 加载模型
    model_backbone = load_backbone(args)
    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()

    print('Loading checkpoint', opts.evaluate)
    checkpoint = torch.load(opts.evaluate, map_location=lambda storage, loc: storage)
    model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
    model_pos = model_backbone
    model_pos.eval()
    
    # 加载视频元数据
    vid = imageio.get_reader(opts.vid_path, 'ffmpeg')
    fps_in = vid.get_meta_data()['fps']
    vid_size = vid.get_meta_data()['size']
    os.makedirs(opts.out_path, exist_ok=True)

    # 预处理数据集
    print('Preparing dataset...')
    if opts.pixel:
        keypoints_data = read_input(opts.json_path, vid_size, None, opts.focus, opts.skip_trans)
    else:
        keypoints_data = read_input(opts.json_path, vid_size, [1,1], opts.focus, opts.skip_trans)

    # 处理帧
    print('Processing frames...')
    results_all = []
    clip_len = opts.clip_len
    total_frames = keypoints_data.shape[0]
    
    # 优化的批处理
    with torch.no_grad():
        for start_idx in tqdm(range(0, total_frames, clip_len)):
            # 获取当前批次数据
            end_idx = min(start_idx + clip_len, total_frames)
            current_clip = keypoints_data[start_idx:end_idx]
            
            # 转换为tensor并传输到GPU
            batch_input = torch.from_numpy(current_clip).float()
            if len(batch_input.shape) == 3:
                batch_input = batch_input.unsqueeze(0)
                
            if torch.cuda.is_available():
                batch_input = batch_input.cuda(non_blocking=True)
            
            if hasattr(args, 'no_conf') and args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            
            # 模型推理
            if hasattr(args, 'flip') and args.flip:
                batch_input_flip = flip_data(batch_input)
                predicted_3d_pos_1 = model_pos(batch_input)
                predicted_3d_pos_flip = model_pos(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
            else:
                predicted_3d_pos = model_pos(batch_input)
                
            # 后处理
            if hasattr(args, 'rootrel') and args.rootrel:
                predicted_3d_pos[:,:,0,:] = 0
            else:
                predicted_3d_pos[:,0,0,2] = 0
                
            if hasattr(args, 'gt_2d') and args.gt_2d:
                predicted_3d_pos[...,:2] = batch_input[...,:2]
            
            # 转移到CPU
            results_all.append(predicted_3d_pos.cpu().numpy())
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 整合结果
    print('Finalizing results...')
    if len(results_all) > 0:
        results_all = np.concatenate(results_all, axis=1)
        if len(results_all.shape) == 4 and results_all.shape[0] == 1:
            results_all = results_all[0]
    else:
        results_all = np.array([])

    # 保存和渲染
    if not opts.skip_render:
        render_and_save(results_all, '%s/X3D.mp4' % (opts.out_path), keep_imgs=False, fps=fps_in)
    
    if opts.pixel:
        results_all = results_all * (min(vid_size) / 2.0)
        results_all[:,:,:2] = results_all[:,:,:2] + np.array(vid_size) / 2.0
    
    # 保存结果
    if opts.save_format == 'npy' or opts.save_format == 'all':
        np.save('%s/X3D.npy' % (opts.out_path), results_all)
    if opts.save_format == 'csv' or opts.save_format == 'all':
        np.savetxt('%s/X3D.csv' % (opts.out_path), results_all.reshape(-1, 3), delimiter=',')
    if opts.save_format == 'json' or opts.save_format == 'all':
        import json
        with open('%s/X3D.json' % (opts.out_path), 'w') as f:
            json.dump(results_all.tolist(), f)
    if opts.save_format == 'npz' or opts.save_format == 'all':
        # 创建output_3D目录
        os.makedirs(os.path.join(opts.out_path, 'output_3D'), exist_ok=True)
        # 保存为npz格式，使用'reconstruction'作为键，符合train_lstm.py的要求
        np.savez(os.path.join(opts.out_path, 'output_3D', 'output_keypoints_3d.npz'), 
                 reconstruction=results_all)
        print(f"已保存NPZ文件到: {os.path.join(opts.out_path, 'output_3D', 'output_keypoints_3d.npz')}")
    
    print('Done!')