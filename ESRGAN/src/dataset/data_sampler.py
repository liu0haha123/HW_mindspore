import math
import mindspore
import mindspore.dataset as ds

class EnlargedSampler(ds.Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    Modified from torch.utils.data.distributed.DistributedSampler
    Support enlarging the dataset for iteration-based training, for saving
    time when restart the dataloader after each epoch
    """

    def __init__(self, dataset, num_replicas, rank, ratio=1):
        super(EnlargedSampler, self).__init__()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = math.ceil(
            len(self.dataset) * ratio / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.randperm = mindspore.ops.Randperm(max_length=self.total_size,pad=-1)
    def __iter__(self):
        # deterministically shuffle based on epoch
        indices = self.randperm(mindspore.Tensor( [20], dtype=mindspore.int32)).tolist()
        dataset_size = len(self.dataset)
        indices = [v % dataset_size for v in indices]

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
