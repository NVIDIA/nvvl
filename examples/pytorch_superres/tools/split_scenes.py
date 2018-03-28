import argparse
import os
import subprocess


def split_scenes(raw_data_path, out_data_path):

    if not os.path.isdir(os.path.join(raw_data_path,'4K/scenes/train')):
        os.makedirs(os.path.join(raw_data_path,'4K/scenes/train'))
    if not os.path.isdir(os.path.join(raw_data_path,'4K/scenes/val')):
        os.makedirs(os.path.join(raw_data_path,'4K/scenes/val'))

    start = "00:00:00.0"
    with open("./data/timestamps") as f:
        for i, line in enumerate(f.readlines()):
            m, s = divmod(float(line), 60)
            h, m = divmod(m, 60)
            print(h,m,s)
            end = "%02d:%02d:%02d" %(h, m, s)
            print(end)
            if i < 53:
                subset = 'train'
            else:
                subset = 'val'
            filepath = os.path.join(out_data_path, '4K/scenes', subset)
            filename = os.path.join(filepath, 'scene_' + str(i) + '.mp4') 
            cmd = ["ffmpeg", "-i", args.raw_data, "-ss", start, "-to", end, "-c:v", "libx264", "-g", "4", "-keyint_min", "4", "-profile:v", "high", "-crf", "18", "-an", filename]
            subprocess.run(cmd)
            start = end


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data', type=str, default=None)
    parser.add_argument('--out_data', type=str, default=None)
    args = parser.parse_args()
    args.raw_data is not None, 'Provide --raw_data path to Myanmar 4K mp4'
    args.out_data is not None, 'Provide --raw_data path to Myanmar 4K mp4'
    split_scenes(args.raw_data, args.out_data_path)
