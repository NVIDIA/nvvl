import argparse
import os
import subprocess

def extract_frames(master_path, resolution, codec, crf, keyint):

    desc = [resolution, 'scenes']
    desc += [codec] if codec else []
    desc += ["crf"+crf] if crf else []
    desc += ["keyint"+keyint] if keyint else []
    master_in_path = os.path.join(master_path,*desc)

    if not os.path.exists(os.path.join(master_in_path,'train')):
        raise ValueError("No training data found in "+master_in_path+".\n" +
                         "If you specified a codec, crf, or keyint in\n" +
                         "transcode_scenes.py you should do so as well here.")

    for subset in ['train', 'val']:
        for in_file in os.listdir(os.path.join(master_in_path,subset)):
            if in_file.endswith('.mp4'):
                scene = in_file.split('_')[1].split('.')[0]
                out_path = os.path.join(master_path,resolution,'frames',subset,scene)
                if not os.path.isdir(out_path):
                    os.makedirs(out_path)
                in_path = os.path.join(master_in_path,subset,in_file)
                cmd = ["ffmpeg", "-i", in_path, os.path.join(out_path, "%05d.png")]
                subprocess.run(cmd)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--master_path', type=str, default=None, required=True,
                        help="Path to root data directory")
    parser.add_argument('--resolution', type=str, default=None, required=True,
                        help="one of '4K', '1080p', '720p', or '540p'")
    parser.add_argument('--codec', type=str, default=None,
                        help="one of 'h264' or 'hevc'")
    parser.add_argument('--crf', type=str, default=None,
                        help="crf value passed to ffmpeg")
    parser.add_argument('--keyint', type=str, default=None,
                        help="keyframe interval")
    args = parser.parse_args()
    assert args.resolution in ['1080p', '720p', '540p'], '--resolution must be one of 1080p, 720p, 540p'
    assert args.master_path is not None, 'Provide --master_data path to root data directory containing Myanmar scenes in desired resolution'
    extract_frames(**vars(args))
