import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def get_optimizer(model, opt):
  optimizer_type = opt['optim']['optimizer']
  base_lr = opt['optim']['lr']
  params_dict = dict(model.named_parameters())
  params = []
  for k, v in params_dict.items():
    if k in opt['optim']['specific_lr']:
      lr = opt['optim']['specific_lr'][k]*base_lr
      params += [{'params':[v], 'lr':lr}]
    else:
      lr = base_lr
      params += [{'params':[v], 'lr':lr}]
    print('Set learning rate for [%s] : %f' % (k, lr))
  print('Using optimizer: %s'% optimizer_type)
  if optimizer_type == 'Adadelta':
    optimizer = optim.Adadelta(params,
        lr=base_lr,
        rho=0.95,
        eps=1e-6,
        weight_decay=0.0)
  elif optimizer_type == 'Adam':
    optimizer = optim.Adam(params, lr=base_lr)
  elif optimizer_type == 'SGD':
    optimizer = optim.SGD(params, lr=base_lr, weight_decay=0.0005, momentum=0.95) 
  else:
    raise NotImplementedError
  return optimizer
