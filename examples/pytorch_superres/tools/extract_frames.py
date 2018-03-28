import argparse
import os
import subprocess

def extract_frames(master_path, resolution):

    for in_file in os.listdir(os.path.join(master_path,resolution,'scenes/train')):
        if in_file.endswith('.mp4'):
            scene = in_file.split('_')[1].split('.')[0]
            out_path = os.path.join(master_path,resolution,'frames/train',scene)
            if not os.path.isdir(out_path):
                os.makedirs(out_path)
            in_path = os.path.join(master_path,resolution,'scenes/train',in_file)
            cmd = ["ffmpeg", "-i", in_path, os.path.join(out_path, "%05d.png")]
            subprocess.run(cmd)

    for in_file in os.listdir(os.path.join(master_path,resolution,'scenes/val')):
        if in_file.endswith('.mp4'):
            scene = in_file.split('_')[1].split('.')[0]
            out_path = os.path.join(master_path,resolution,'frames/val',scene)
            if not os.path.isdir(out_path):
                os.makedirs(out_path)
            in_path = os.path.join(master_path,resolution,'scenes/val',in_file)
            cmd = ["ffmpeg", "-i", in_path, os.path.join(out_path, "%05d.png")]
            subprocess.run(cmd)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--master_data', type=str, default=None)
    parser.add_argument('--resolution', type=str, default=None)
    args = parser.parse_args()
    assert args.resolution in ['1080p', '720p', '540p'], '--resolution must be one of 1080p, 720p, 540p'
    assert args.master_data is not None, 'Provide --master_data path to root data directory containing Myanmar scenes in desired resolution'
    extract_frames(args.master_data, args.resolution)
