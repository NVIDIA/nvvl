import collections
import torch

from .dataset import VideoDataset


class VideoLoader(object):
    """Loads batches of sequences of frames from a video file. Meant to be
    nearly a drop-in replacement for a torch.util.data.DataLoader.

    Parameters
    ----------

    dataset : VideoDataset
        dataset from which to load the frames, must be a
        nvvl.VideoDataset.

    batch_size : int, optional
        how many samples (i.e. sequences) per batch to load (Default: 1)

    shuffle : bool, optional
        shuffle the order of samples (Default: False)

    distributed : bool, optional
        use a distributed sampler, requires shuffle (Default: False)

    sampler : torch.utils.data.Sampler, optional
        defines the strategy to draw samples from the
        dataset. Mutually exclusive with shuffle and distributed.

    batch_sampler : torch.utils.data.Sampler, optional
        like sampler, but returns a batch of indices at a
        time. Mutually exclusive with batch_size, shuffle,
        distributed, sampler, and drop_last.

    drop_last : bool, optional
        drop the last incomplete batch. It is currently not
        implemented to have this set to False. (Default: True)

    buffer_length : int, optional
        number of batches to preload (Default: 3)

    """
    def __init__(self, dataset, batch_size=1, shuffle=False,
                 distributed=False, sampler=None,
                 batch_sampler=None, drop_last=True, buffer_length=3):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        if batch_sampler is not None:
            if (batch_size > 1 or shuffle or distributed
                or sampler is not None or drop_last):
                raise ValueError('batch_sampler is mutually exclusive with '
                                 'batch_size, shuffle, distributed, sampler, '
                                 'and drop_last')

        if sampler is not None:
            if shuffle or distributed:
                raise ValueError("sampler is mutually exclusive with shuffle and distributed")

        if batch_sampler is None:
            if sampler is None:
                if distributed:
                    if not shuffle:
                        raise ValueError("pytorch distributed is always shuffled")
                    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                elif shuffle:
                    sampler = torch.utils.data.sampler.RandomSampler(dataset)
                else:
                    sampler = torch.utils.data.sampler.SequentialSampler(dataset)
            batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler

        self.tensor_queue = collections.deque()
        self.batch_size_queue = collections.deque()
        self.buffer_length = buffer_length

    def _receive_batch(self):
        batch_size = self.batch_size_queue.popleft()
        t = self.dataset._create_tensor_map(batch_size)
        labels = []
        for i in range(batch_size):
            _, label = self.dataset._start_receive(t, i)
            labels.append(label)

        self.tensor_queue.append((batch_size, t, labels))

    def get_stats(self):
        return self.dataset.get_stats()

    def reset_stats(self):
        return self.dataset.reset_stats()

    def __len__(self):
        return len(self.batch_sampler)

    def __next__(self):
        if not self.tensor_queue:
            assert self.dataset.samples_left == 0, "Tensor queue is empty but there are samples left in the VideoDataset"
            raise StopIteration

        # first fire off a receive to keep the pipe filled
        if self.batch_size_queue:
            self._receive_batch()

        batch_size, t, labels = self.tensor_queue.popleft()
        for i in range(batch_size):
            self.dataset._finish_receive()

        if any(label is not None for label in labels):
            t["labels"] = labels

        return t

    def __iter__(self):
        if self.dataset.samples_left != 0:
            raise RuntimeError("Need to exhaust iterator before creating a new one")

        for b in iter(self.batch_sampler):
            for i in b:
                self.dataset._read_sample(i)
            self.batch_size_queue.append(len(b))

        for i in range(self.buffer_length):
            self._receive_batch()
        return self
