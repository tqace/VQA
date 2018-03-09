import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

def get_criterion(opt):
  criterion_type = opt['optim']['criterion']['type']
  normalize = opt['optim']['criterion']['normalize']
  print('Using criterion: %s; normalize: %s'%(criterion_type, normalize))
  if criterion_type == 'BCELoss':
    if normalize == 'batch_size':
      weight = torch.from_numpy(np.ones([opt['vqa']['nans']], dtype=np.float32)/opt['optim']['batch_size'])
      criterion = nn.BCELoss(weight=weight, size_average=False)
    elif normalize == 'element':
      criterion = nn.BCELoss(size_average=True)
    elif normalize == 'none':
      criterion = nn.BCELoss(size_average=False)
    else:
      raise NotImplementedError
  else:
    raise NotImplementedError
  return criterion.cuda()

def get_criterion_AC(opt):
  criterion_type = opt['optim']['criterion']['type']
  normalize = opt['optim']['criterion']['normalize']
  print('Using criterion: %s; normalize: %s'%(criterion_type, normalize))
  if criterion_type == 'BCELoss':
    if normalize == 'batch_size':
      weight = torch.from_numpy(np.ones([opt['AC']['topK']], dtype=np.float32)/opt['optim']['batch_size'])
      criterion = nn.BCELoss(weight=weight, size_average=False)
    elif normalize == 'element':
      criterion = nn.BCELoss(size_average=True)
    elif normalize == 'none':
      criterion = nn.BCELoss(size_average=False)
    else:
      raise NotImplementedError
  else:
    raise NotImplementedError
  return criterion.cuda()

