import os
import sys
sys.path.append('..')
import numpy as np
import torch
import torch.utils.data as data
from .factory import DatasetFactory
from lib import utils
from mpi4py import MPI
import h5py
from . import vgenome

class FeaturesDataset(data.Dataset):
  def __init__(self, data_split, opt, img_ids=None):
    super(FeaturesDataset, self).__init__()
    if data_split == 'testdev2015':
      data_split = 'test2015'
    self.data_split = data_split
    if data_split not in ['train2014', 'val2014', 'test2015', 'vgenome']:
      raise NotImplementedError
    self.opt = opt
    self.__processed()
    self.hdf5_file = h5py.File(self.path_h5, 'r', driver='mpio', comm=MPI.COMM_WORLD)
    self.dataset_features = self.hdf5_file[self.opt['coco']['feature']['type']]
    self.index_to_name, self.name_to_index = self._load_dicts()
    if self.opt['coco']['feature']['preload']:
      dataset_features = {}
      print('[FeaturesDataset : %s] Loading features into memory ...'%(data_split))
      for i, idx in enumerate(self.name_to_index.values()):
        utils.xprocess(i+1, len(self.index_to_name), end_log="Done!")
        dataset_features[idx] = self.dataset_features[idx]
      self.dataset_features = dataset_features
 
  def __processed(self):
    self.dir_processed = os.path.join(self.opt['dirs']['resource'], 'h5features_resnet152')
    if not os.path.exists(self.dir_processed):
      os.makedirs(self.dir_processed)
    self.path_h5 = os.path.join(self.dir_processed, '%sset.h5'%self.data_split)
    self.path_fnames = os.path.join(self.dir_processed, '%sset_fnames.txt'%self.data_split)
    self.feature_dir = os.path.join(self.opt['dirs']['coco_feature'], self.data_split)
    if False in [os.path.exists(f) for f in [self.path_h5, self.path_fnames]]:
      os.system('rm -fv %s'%(self.path_fnames))
      os.system('rm -fv %s'%(self.path_h5))
      assert os.path.isdir(self.feature_dir)
      feature_file_names = sorted(os.listdir(self.feature_dir))
      image_names, image_ids = [], []
      for ffn in feature_file_names:
        img_name = ffn[:-4]
        img_id = int(img_name.split('.')[0].split('_')[-1])
        if self.data_split == 'vgenome':
          img_id = vgenome.vg_iid_2_vqa_iid(img_id, self.opt) 
        image_names.append(img_name)
        image_ids.append(img_id)
      shape = [len(image_names)] + list(self.load_feature_file(os.path.join(self.feature_dir, image_names[0] + '.npz')).shape)
      hdf5_file = h5py.File(self.path_h5, 'w')
      hdf5_dataset = hdf5_file.create_dataset('%s'%self.opt['coco']['feature']['type'], shape, dtype='f')
      print('Building hdf5 for %s'%self.data_split)
      for i, img_name in enumerate(image_names):
        utils.xprocess(i+1, len(image_names), end_log='Done!')
        hdf5_dataset[i] = self.load_feature_file(os.path.join(self.feature_dir, img_name + '.npz'))
      with open(self.path_fnames, 'w') as f:
        for img_name in image_names:
          f.write(img_name + '\n')

  def _load_dicts(self):
    with open(self.path_fnames, 'r') as f:
      self.index_to_name = f.readlines()
    self.index_to_name = [name[:-1] for name in self.index_to_name]
    self.name_to_index = {name:index for index,name in enumerate(self.index_to_name)}
    return self.index_to_name, self.name_to_index

  def load_feature_file(self, fpath):
    if self.opt['coco']['feature']['type'] == 'mcb':
      return np.load(fpath)['x']
    else:
      raise NotImplementedError

  def __getitem__(self, index):
    item = {}
    item['name'] = self.index_to_name[index]
    item['visual'] = self.get_features(index)
    return item

  def get_features(self, index):
    if self.dataset_features is None:
      return torch.Tensor(self.load_feature_file(os.path.join(self.feature_dir, self.index_to_name[index] + '.npz')))
    else:
      return torch.Tensor(self.dataset_features[index])

  def get_by_name(self, image_name):
      index = self.name_to_index[image_name]
      return self[index]

  def __len__(self):
      return len(self.name_to_index)

class FeaturesDatasetFactory(DatasetFactory):
  def __init__(self, *datasets):
    super(FeaturesDatasetFactory, self).__init__(*datasets)
    for ds in self.datasets:
      assert isinstance(ds, FeaturesDataset)
    self.name_to_dataset_idx = {}
    for i, ds in enumerate(self.datasets):
      for image_name in ds.name_to_index:
        self.name_to_dataset_idx[image_name] = i

  def get_by_name(self, image_name):
    return self.datasets[self.name_to_dataset_idx[image_name]].get_by_name(image_name)

def factory(data_splits, opt, img_ids={}):
  data_splits = data_splits.split('+')
  datasets = [FeaturesDataset(data_split, opt, img_ids.get(data_split, None)) for data_split in data_splits]
  return FeaturesDatasetFactory(*datasets)
