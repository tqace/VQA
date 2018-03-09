import torch.utils.data as data

class DatasetFactory(data.Dataset):
  def __init__(self, *datasets):
    super(DatasetFactory, self).__init__()
    for dataset in datasets:
      assert isinstance(dataset, data.Dataset)
    self.datasets = datasets

  def __len__(self):
    length = 0
    for ds in self.datasets:
      length += len(ds)
    return length

  def __getitem__(self, index):
    base_index = 0
    for ds in self.datasets:
      if index < (base_index + len(ds)):
        return ds[index - base_index]
      else:
        base_index += len(ds)
    raise ValueError
