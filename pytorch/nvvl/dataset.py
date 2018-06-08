import bisect
import cffi
import collections
import random
import sys
import torch
import torch.utils.data

from . import lib

class ProcessDesc(object):
    """Describes processing done on a decoded frame.

    Parameters
    ----------
    type : string, optional
        Type of the output, can be one of "float", "half", or "byte"
        (Default: "float")

    width, height : int, optional
        width and height to crop frame to, set to 0 for full frame
        size (Default: 0)

    scale_width, scale_height : int, optional
        width and height to scale image to before cropping, set to 0
        for no scaling (Default: 0)

    normalized : bool, optional
        Normalize all values to [0, 1] instead of [0, 255] (Default: False)

    random_crop : bool, optional
        If True, the origin of the crop is randomly choosen. If False,
        the crop originates at (0, 0).  (Default: False)

    random_flip : bool, optional
        If True, flip the image horizontally before cropping. (Default: False)

    color_space : enum, optional
        Color space to return images in, one of "RGB" or "YCbCr". (Default: RGB)

    index_map : list of ints, optional
        Map from indices into the decoded sequence to indices in this Layer.

        None indicates a 1-to-1 mapping of the frames from sequence to
        layer.

        For examples, To reverse the frames of a 5 frame sequence, set
        index_map to [4, 3, 2, 1, 0].

        An index of -1 indicates that the decoded frame should not
        be used in this layer. For example, to extract just the
        middle frame from a 5 frame sequence, set index_map to
        [-1, -1, 0, -1, -1].

        The returned tensors will be sized to fit the maximum index in
        this array (if it is provided, they will fit the full sequence
        if it is None).

        (Default: None)

    dimension_order : string, optional
        Order of dimensions in the returned tensors. Must contain
        exactly one of 'f', 'c', 'h', and 'w'. 'f' for frames in the
        sequence, 'c' for channel, 'h' for height, 'w' for width, and
        'h'. (Default: "fchw")

    """


    def __init__(self, type="float",
                 width=0, height=0, scale_width=0, scale_height=0,
                 normalized=False, random_crop=False, random_flip=True,
                 color_space="RGB", index_map=None, dimension_order="fchw"):
        self.ffi = lib._ffi
        self._desc = self.ffi.new("struct NVVL_LayerDesc*")

        self.width = width
        self.height = height
        self.scale_width = scale_width
        self.scale_height = scale_height
        self.normalized = normalized
        self.random_crop = random_crop
        self.random_flip = random_flip

        if index_map:
            self.index_map = self.ffi.new("int[]", index_map)
            self.count = max(index_map) + 1
            self.index_map_length = len(index_map)
        else:
            self.index_map = self.ffi.NULL
            self.count = 0
            self.index_map_length = 0

        if color_space.lower() == "rgb":
            self.color_space = lib.ColorSpace_RGB
            self.channels = 3
        elif color_space.lower() == "ycbcr":
            self.color_space = lib.ColorSpace_YCbCr
            self.channels = 3
        else:
            raise ValueError("Unknown color space")

        if type == "float":
            self.tensor_type = torch.cuda.FloatTensor
        elif type == "half":
            self.tensor_type = torch.cuda.HalfTensor
        elif type == "byte":
            self.tensor_type = torch.cuda.ByteTensor
        else:
            raise ValueError("Unknown type")

        self.dimension_order = dimension_order

    def _get_dim(self, dim):
        if dim == 'c':
            return self.channels
        elif dim == 'f':
            return self.count
        elif dim == 'h':
            return self.height
        elif dim == 'w':
            return self.width
        raise ValueError("Invalid dimension")

    def get_dims(self):
        dims = []
        for d in self.dimension_order:
            dims.append(self._get_dim(d))
        return dims

    def __getattribute__(self, name):
        try:
            d = super().__getattribute__("_desc")
            return d.__getattribute__(name)
        except AttributeError:
            return super().__getattribute__(name)
        raise AttributeError()

    def __setattr__(self, name, value):
        try:
            self._desc.__setattr__(name, value)
        except:
            super().__setattr__(name, value)

    def desc(self):
        return self._desc

log_levels = {
    "debug" : lib.LogLevel_Debug,
    "info"  : lib.LogLevel_Info,
    "warn"  : lib.LogLevel_Warn,
    "error" : lib.LogLevel_Error,
    "none"  : lib.LogLevel_None
    }

class VideoDataset(torch.utils.data.Dataset):
    """VideoDataset

    Parameters
    ----------
    filenames : collection of strings
        list of video files to draw from

    sequence_length : int
        how many frames are in each sample

    device_id : int, optional
        GPU device to use (Default: 0)

    processing : dict {string -> ProcessDesc}, optional
        Describes processing to be done on the sequence to generate
        each data item. If None, each frame in the sequence will be
        returned as is. (Default: None)

    log_level : string, optional
        One of "debug", "info", "warn", "error", or "none".
        (Default: "warn")
    """
    def __init__(self, filenames, sequence_length, device_id=0,
                 processing = None, log_level = "warn"):
        self.ffi = lib._ffi
        self.filenames = filenames
        self.sequence_length = sequence_length
        self.device_id = device_id

        self.processing = processing
        if self.processing is None:
            self.processing = {"default" : ProcessDesc()}

        try:
            log_level = log_levels[log_level]
        except KeyError:
            print("Invalid log level", log_level, "using warn.", file=sys.stderr)
            log_level = lib.LogLevel_Warn

        if not filenames:
            raise ValueError("Empty filenames list given to VideoDataset")

        if sequence_length < 1:
            raise ValueError("Sequence length must be at least 1")

        self.loader = lib.nvvl_create_video_loader_with_log(self.device_id, log_level)

        self.total_frames = 0
        self.frame_counts = []
        self.start_index = []
        for f in filenames:
            count = lib.nvvl_frame_count(self.loader, str.encode(f));
            if count < self.sequence_length:
                print("NVVL WARNING: Ignoring", f, "because it only has", count,
                      "frames and the sequence length is", self.sequence_length)
                continue
            count = count - self.sequence_length + 1
            self.frame_counts.append(count)
            self.total_frames += count
            self.start_index.append(self.total_frames) # purposefully off by one for bisect to work

        size = lib.nvvl_video_size(self.loader)
        self.width = size.width
        self.height = size.height

        for name, desc in self.processing.items():
            if desc.width == 0:
                desc.width = self.width

            if desc.height == 0:
                desc.height = self.height

            if desc.count == 0:
                desc.count = self.sequence_length

        self.samples_left = 0

        self.seq_queue = collections.deque()

        self.get_count = 0
        self.get_count_warning_threshold = 1000
        self.disable_get_warning = False

    def get_stats(self):
        return lib.nvvl_get_stats(self.loader)

    def reset_stats(self):
        return lib.nvvl_reset_stats(self.loader)

    def set_log_level(self, level):
        """Sets the log level from now forward

        Parameters
        ----------
        level : string
            The log level, one of "debug", "info", "warn", "error", or "none"
        """
        lib.nvvl_set_log_level(self.loader, log_levels[level])

    def _read_sample(self, index):
        # we want bisect_right here so the first frame in a file gets the file, not the previous file
        file_index = bisect.bisect_right(self.start_index, index)
        frame = index - self.start_index[file_index - 1] if file_index > 0 else index

        lib.nvvl_read_sequence(self.loader,
                               str.encode(self.filenames[file_index]),
                               frame, self.sequence_length)
        self.samples_left += 1

    def _get_layer_desc(self, desc):
        d = desc.desc()

        if (desc.random_crop and (self.width > desc.width)):
            d.crop_x = random.randint(0, self.width - desc.width)
        else:
            d.crop_x = 0

        if (desc.random_crop and (self.height > desc.height)):
            d.crop_y = random.randint(0, self.height - desc.height)
        else:
            d.crop_y = 0

        if (desc.random_flip):
            d.horiz_flip = random.random() < 0.5
        else:
            d.horiz_flip = False

        return d

    def _start_receive(self, tensor_map, index=0):
        seq = lib.nvvl_create_sequence_device(self.sequence_length, self.device_id)

        for name, desc in self.processing.items():
            tensor = tensor_map[name]
            layer = self.ffi.new("struct NVVL_PicLayer*")
            if desc.tensor_type == torch.cuda.FloatTensor:
                layer.type = lib.PDT_FLOAT
            elif desc.tensor_type == torch.cuda.HalfTensor:
                layer.type = lib.PDT_HALF
            elif desc.tensor_type == torch.cuda.ByteTensor:
                layer.type = lib.PDT_BYTE

            strides = tensor[index].stride()
            try:
                desc.stride.x = strides[desc.dimension_order.index('w')]
                desc.stride.y = strides[desc.dimension_order.index('h')]
                desc.stride.n = strides[desc.dimension_order.index('f')]
                desc.stride.c = strides[desc.dimension_order.index('c')]
            except ValueError:
                raise ValueError("Invalid dimension order")
            layer.desc = self._get_layer_desc(desc)[0]
            if desc.index_map_length > 0:
                layer.index_map = desc.index_map
                layer.index_map_length = desc.index_map_length
            else:
                layer.index_map = self.ffi.NULL
            layer.data = self.ffi.cast("void*", tensor[index].data_ptr())
            lib.nvvl_set_layer(seq, layer, str.encode(name))

        self.seq_queue.append(seq)
        lib.nvvl_receive_frames(self.loader, seq)
        return seq

    def _finish_receive(self, synchronous=False):
        if not self.seq_queue:
            raise RuntimeError("Unmatched receive")

        if self.samples_left <= 0:
            raise RuntimeError("No more samples left in decoder pipeline")

        seq = self.seq_queue.popleft()

        if synchronous:
            lib.nvvl_sequence_wait(seq)
        else:
            lib.nvvl_sequence_stream_wait_th(seq)
        lib.nvvl_free_sequence(seq)
        self.samples_left -= 1

    def _create_tensor_map(self, batch_size=1):
        tensor_map = {}
        with torch.cuda.device(self.device_id):
            for name, desc in self.processing.items():
                tensor_map[name] = desc.tensor_type(batch_size, *desc.get_dims())
        return tensor_map

    def __getitem__(self, index):
        if (self.samples_left >  0):
            raise RuntimeError("Can't synchronously get an item when asyncronous frames are pending")

        self.get_count += 1
        if (self.get_count > self.get_count_warning_threshold
            and not self.disable_get_warning):
            print("WARNING: Frequent use of VideoDataset's synchronous get operation\n"
                  "detected. This operation is slow and should only be used for \n"
                  "debugging and other limited cases. To turn this warning off, set\n"
                  "the disable_get_warning attribute of the VideoDataset to True.\n")
            self.disable_get_warning = True

        self._read_sample(index)
        tensor_map = self._create_tensor_map()
        seq = self._start_receive(tensor_map)
        self._finish_receive(True)

        if len(tensor_map) == 1 and "default" in tensor_map:
            return tensor_map["default"][0].cpu()
        return {name: tensor[0].cpu() for name, tensor in tensor_map.items()}

    def __len__(self):
        return self.total_frames
