#!/usr/bin/env python
import torch
import sys
sys.path.append('../')
sys.path.append('../vqa')
import json
from lib import utils
from lib import options
from models.teney_gru import TeneyNet_GRU
from models.teney_gru_att2 import TeneyNet_GRU_ATT2
from models.teney_lstm import TeneyNet_LSTM
from models.teney_lstm_att2 import TeneyNet_LSTM_ATT2
from optim.engine import test
from datasets import vqa as vqadataset
import pickle
import os

def ensemble(paths,file_name,model):
  assert len(paths) > 0
  ckpt_save = torch.load(paths[0])
  print ('ensembling...')
  if len(paths)>1:
    for i in range(1,len(paths)):
      ckpt = torch.load(paths[i])
      for param in ckpt_save['model'].keys():
        ckpt_save['model'][param] += ckpt['model'][param]
    for param in ckpt_save['model'].keys():
      ckpt_save['model'][param] /= len(paths)
  #torch.save(ckpt_save,file_name)
  print ('ensemble done')

  opt = options.get_optiotns()
  testset = vqadataset.factory(mode='test', data_split=opt['coco']['test_split'], opt=opt)
  testloader = testset.data_loader(batch_size=opt['optim']['batch_size'],
      num_workers=opt['optim']['workers'],
      shuffle=False)
  print ('tring to make the test result...')
  print('Model type: %s'%mode)
  if mode == 'teney_gru':
    model = TeneyNet_GRU(15219 , 3000, opt)
  elif mode == 'teney_gru_att2':
    model = TeneyNet_GRU_ATT2(15219 , 3000, opt)
  elif mode == 'teney_lstm':
    model = TeneyNet_LSTM(15219 , 3000, opt)
  elif mode == 'teney_lstm_att2':
    model = TeneyNet_LSTM_ATT2(15219 , 3000, opt)
  print ('loading model from ckpt...')
  model.cuda()
  model.load_state_dict(ckpt_save['model'])
  torch.backends.cudnn.benchmark = True
  print ('testing...')
  results = test(testloader,model,opt)
  with open(file_name,'w') as f:
    json.dump(results['testdev2015'], f, indent=2)
    


if __name__ == '__main__':
  for folder in ['gru_att2','lstm','lstm_att2']:
    mode = 'teney_'+folder
    for sub in ['/1/lr7e-4','/2/lr7e-4']:
      ckpt_path = os.path.join('/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/data/result/vqa1/checkpoint_nowe_novg/',folder+sub)
      test_path = os.path.join('/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/data/result/vqa1/test_nowe_novg/',folder+sub)
      for file_name in os.listdir(ckpt_path):
        if 'epoch4' in file_name or 'epoch1' in file_name:
          ensemble([os.path.join(ckpt_path,file_name)],os.path.join(test_path,file_name.split('.')[0]+'.json'),mode)




