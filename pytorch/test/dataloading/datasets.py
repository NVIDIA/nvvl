import copy
import sys, time, argparse, os, subprocess, shutil
import math, numbers, random, bisect
import subprocess

from random import Random

from skimage import io, transform
from os import listdir
from os.path import join
from glob import glob

import numpy as np

import torch
import torch.utils.data as data

import nvvl
import lintel


class dataset(object):
    def __init__(self, width, height, frames):
        self.width = width
        self.height = height
        self.num_frames = frames

class lintelDataset():
    def __init__(self, frames, is_cropped, crop_size,
                 root, batch_size, frame_size = [-1, -1]):
        self.root = root
        self.frames = frames
        self.is_cropped = is_cropped
        self.crop_size = crop_size

        self.files = glob(os.path.join(self.root, '*.mp4'))

        assert len(self.files) > 1, "[Error] No video files in %s" % self.root

        image_shape = nvvl.video_size_from_file(self.files[0])
        self.image_shape = [image_shape.height, image_shape.width]
        self.frame_size = frame_size
        print("Video size: ", self.image_shape[0], "x", self.image_shape[1], "\n")
        if self.is_cropped:
            self.frame_size = self.crop_size
        else:
            self.frame_size = self.image_shape

        self.dataset = dataset(width=self.image_shape[1],
                               height=self.image_shape[0],
                               frames=self.frames)

        self.gt_step = (self.frames - 1) * 2 + 1

        self.total_frames = 0
        self.frame_counts = []
        self.start_index = []
        self.videos = []
        for i, filename in enumerate(self.files):

            with open(filename, 'rb') as f:
                video = f.read()
                self.videos.append(video)

            cmd = ["ffprobe", "-v", "error", "-count_frames", "-select_streams",
                    "v:0", "-show_entries", "stream=nb_frames", "-of",
                    "default=nokey=1:noprint_wrappers=1", filename]
            count = int(subprocess.check_output(cmd))
            count -= self.gt_step
            if count < 0:
                print("[Warning] Video does not have enough frames\n\t%s" % f)
                count = 0
            self.total_frames += count
            self.frame_counts.append(count)
            self.start_index.append(self.total_frames)

        assert self.total_frames >= 1, "[Error] Not enough frames at \n\t%s" % self.root

        self.frame_buffer = np.zeros((3, self.frames,
                                      self.frame_size[0], self.frame_size[1]),
                                      dtype = np.float32)

    def __len__(self):
        return self.total_frames

    def __getitem__(self, index):
        index = index % self.total_frames
        # we want bisect_rigtht here so that the first frame in a file gets the
        # file, not the previous file
        file_index = bisect.bisect_right(self.start_index, index)
        frame = index - self.start_index[file_index - 1] if file_index > 0 else index
        filename = self.files[file_index]

        video = self.videos[file_index]
        frames, seek_distance = lintel.loadvid(
            video,
            should_random_seek=True,
            width=self.dataset.width,
            height=self.dataset.height,
            num_frames=self.dataset.num_frames,
            fps_cap=60)
        frames = np.frombuffer(frames, dtype=np.uint8)
        frames = np.reshape(
            frames, newshape=(self.dataset.num_frames, self.dataset.height,
                              self.dataset.width, 3))

        for i in range(self.frames):
            #TODO(jbarker): Tidy this up and remove redundant computation
            if i == 0 and self.is_cropped:
                crop_x = random.randint(0, self.image_shape[1] - self.frame_size[1])
                crop_y = random.randint(0, self.image_shape[0] - self.frame_size[0])
            elif self.is_cropped == False:
                crop_x = math.floor((self.image_shape[1] - self.frame_size[1]) / 2)
                crop_y = math.floor((self.image_shape[0] - self.frame_size[0]) / 2)
                self.crop_size = self.frame_size

            image = frames[i, crop_y:crop_y + self.crop_size[0],
                            crop_x:crop_x + self.crop_size[1],
                            :]

            self.frame_buffer[:, i, :, :] = np.rollaxis(image, 2, 0)

        return torch.from_numpy(self.frame_buffer)


class imageDataset():
    def __init__(self, frames, is_cropped, crop_size,
                 root, batch_size):
        self.root = root
        self.frames = frames
        self.is_cropped = is_cropped
        self.crop_size = crop_size

        self.files = glob(os.path.join(self.root, '*/*.png'))
        if len(self.files) < 1:
            self.files = glob(os.path.join(self.root, '*/*.jpg'))

        if len(self.files) < 1:
            print(("[Error] No image files in %s" % (self.root)))
            raise LookupError

        self.files = sorted(self.files)

        self.total_frames = 0
        # Find start_indices for different folders
        self.start_index = [0]
        prev_folder = self.files[0].split('/')[-2]
        for (i, f) in enumerate(self.files):
            folder = f.split('/')[-2]
            if i > 0 and folder != prev_folder:
                self.start_index.append(i)
                prev_folder = folder
                self.total_frames -= (self.frames + 1)
            else:
                self.total_frames += 1
        self.total_frames -= (self.frames + 1)
        self.start_index.append(i)

        self.image_shape = list(io.imread(self.files[0]).shape[:2])
        print("Image size: ", self.image_shape[0], "x", self.image_shape[1], "\n")
        if self.is_cropped:
            self.image_shape = self.crop_size

        self.frame_size = self.image_shape

        self.frame_buffer = np.zeros((3, self.frames,
                                      self.frame_size[0], self.frame_size[1]),
                                      dtype = np.float32)

    def __len__(self):
        return self.total_frames

    def __getitem__(self, index):

        index = index % self.total_frames
        # we want bisect_right here so that the first frame in a file gets the
        # file, not the previous file
        next_file_index = bisect.bisect_right(self.start_index, index)
        if self.start_index[next_file_index] < index + self.frames:
            index = self.start_index[next_file_index] - self.frames - 1

        for (i, file_idx) in enumerate(range(index, index + self.frames)):

            image = io.imread(self.files[file_idx])

            #TODO(jbarker): Tidy this up and remove redundant computation
            if i == 0 and self.is_cropped:
                crop_x = random.randint(0, self.image_shape[1] - self.frame_size[1])
                crop_y = random.randint(0, self.image_shape[0] - self.frame_size[0])
            elif self.is_cropped == False:
                crop_x = math.floor((self.image_shape[1] - self.frame_size[1]) / 2)
                crop_y = math.floor((self.image_shape[0] - self.frame_size[0]) / 2)
                self.crop_size = self.frame_size

            image = image[crop_y:crop_y + self.crop_size[0],
                          crop_x:crop_x + self.crop_size[1],
                          :]

            self.frame_buffer[:, i, :, :] = np.rollaxis(image, 2, 0)

        return torch.from_numpy(self.frame_buffer)
