import argparse
import os
import subprocess

def downsample_scenes(master_path, resolution):

    if not os.path.isdir(os.path.join(master_path,resolution,'scenes/train')):
        os.makedirs(os.path.join(master_path,resolution,'scenes/train'))
    if not os.path.isdir(os.path.join(master_path,resolution,'scenes/val')):
        os.makedirs(os.path.join(master_path,resolution,'scenes/val'))

    if resolution == '1080p':
        res_str = '1920:1080'
    elif resolution == '720p':
        res_str = '1280:720'
    elif res_str == '540p':
        '960:540'

    for in_file in os.listdir(os.path.join(master_path,'4K/scenes/train')):
        if in_file.endswith('.mp4'):
            in_path = os.path.join(master_path,'4K/scenes/train',in_file)
            out_file = os.path.join(master_path,resolution,'scenes/train',in_file)
            cmd = ["ffmpeg", "-i", in_path, "-vf", "scale=%s" % res_str,
                   "-sws_flags", "bilinear", "-c:v", "libx264",
                   "-g", "4", "-keyint_min", "4", "-profile:v",
                   "high", "-crf", "18", out_file]
            subprocess.run(cmd)

    for in_file in os.listdir(os.path.join(master_path,'4K/scenes/val')):
        if in_file.endswith('.mp4'):
            in_path = os.path.join(master_path,'4K/scenes/val',in_file)
            out_file = os.path.join(master_path,resolution,'scenes/val',in_file)
            cmd = ["ffmpeg", "-i", in_path, "-vf", "scale=%s" % res_str,
                   "-sws_flags", "bilinear", "-c:v", "libx264",
                   "-g", "4", "-keyint_min", "4", "-profile:v",
                   "high", "-crf", "18", out_file]
            subprocess.run(cmd)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--master_data', type=str, default=None)
    parser.add_argument('--resolution', type=str, default=None)
    args = parser.parse_args()
    assert args.resolution in ['1080p', '720p', '540p'], '--resolution must be one of 1080p, 720p, 540p'
    assert args.master_data is not None, 'Provide --master_data path to root data directory containing 4K Myanmar scenes'
    downsample_scenes(args.master_data, args.resolution)
