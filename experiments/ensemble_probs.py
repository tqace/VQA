#!/usr/bin/env python
import torch
import sys
sys.path.append('../')
sys.path.append('../vqa')
import json
from lib import utils
from lib import options
from models.teney_lstm import TeneyNet_LSTM
from models.teney_gru import TeneyNet_GRU
from models.teney_gru_att2 import TeneyNet_GRU_ATT2
from models.teney_lstm_att2 import TeneyNet_LSTM_ATT2
from optim.engine import test
from datasets import vqa as vqadataset
import pickle
from torch.autograd import Variable
import pdb
import numpy as np

def ensemble(path,mod,file_name):
  opt = options.get_optiotns()
  testset = vqadataset.factory(mode='test', data_split=opt['coco']['test_split'], opt=opt)
  loader = testset.data_loader(batch_size=opt['optim']['batch_size'],
      num_workers=opt['optim']['workers'],
      shuffle=False)
  print (len(loader))

  results_testdev2015 = []
  results_test2015 = []
  save_prob = opt['test']['save_prob']
  if save_prob:
    probs_test2015 = []
    probs_testdev2015 = []
    qid_test2015 = []
    qid_testdev2015 = []
  models = []
  for i in range(len(path)):
    ckpt = torch.load(path[i])
    if mod[i] == 'teney_lstm':
      model = TeneyNet_LSTM(18936 , 3000, opt)
    elif mod[i] == 'teney_gru':
      model = TeneyNet_GRU(18936 , 3000, opt)
    elif mod[i] == 'teney_gru_att2':
      model = TeneyNet_GRU_ATT2(18936 , 3000, opt)
    elif mod[i] == 'teney_lstm_att2':
      model = TeneyNet_LSTM_ATT2(18936 , 3000, opt)
    model.cuda()
    model.load_state_dict(ckpt['model'])
    torch.backends.cudnn.benchmark = True
    models.append(model)
  for indx, sample in enumerate(loader):
    for i in range(len(path)):
      model = models[i]
      model.eval()
      utils.xprocess(indx+1, len(loader))
      batch_size = sample['visual'].size(0)
      input_visual   = Variable(sample['visual'].cuda(), volatile=True)
      input_question = Variable(sample['question'].cuda(), volatile=True)
      input_data_split = sample['data_split']
      output = model(input_visual, input_question)
      if i==0:
        probs = output.data.cpu().numpy()
      else:
        probs += output.data.cpu().numpy()
    print ('probs calculated')
    probs /= len(path)
    pred = np.argsort(-probs)[:,0]
    for j in range(batch_size):
      pred_ans = {'question_id': sample['question_id'][j],
          'answer': loader.dataset.aid_to_ans[pred[j]]}
      if input_data_split[j] == 'test2015':
        results_test2015.append(pred_ans)
        if save_prob:
          probs_test2015.append(probs[j, ...].copy())
          qid_test2015.append(sample['question_id'][j])
      elif input_data_split[j] == 'testdev2015':
        results_testdev2015.append(pred_ans)
        if save_prob:
          probs_testdev2015.append(probs[j, ...].copy())
          qid_testdev2015.append(sample['question_id'][j])
      else:
        raise ValueError
  results = {}
  results['ans_to_aid'] = loader.dataset.ans_to_aid
  results['aid_to_ans'] = loader.dataset.aid_to_ans
  results['test2015'] = results_test2015
  results['testdev2015'] = results_testdev2015
  if save_prob:
    if input_data_split[j] == 'test2015':
      results['probs_test2015'] = np.array(probs_test2015)
      np.savez(file_name+'_test.npz',qid = qid_test2015,probs = results['probs_test2015'])
    elif input_data_split[j] == 'testdev2015':
      results['probs_testdev2015'] = np.array(probs_testdev2015)
      np.savez(file_name+'_testdev.npz',qid = qid_testdev2015,probs = results['probs_testdev2015'])
  with open(file_name+'.json','w') as f:
    json.dump(results['testdev2015'], f, indent=2)
    


if __name__ == '__main__':
    path = []
    mod = ['teney_gru']+['teney_lstm']
    '''
    path.append('/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/data/result/checkpoint/baseline/ensemble_lstm_att2/1.pth')
    path.append('/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/data/result/checkpoint/baseline/ensemble_lstm_att2/2.pth')
    path.append('/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/data/result/checkpoint/baseline/ensemble_lstm_att2/3.pth')
    path.append('/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/data/result/checkpoint/baseline/ensemble_lstm_att2/4.pth')
    path.append('/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/data/result/checkpoint/baseline/ensemble_lstm_att2/5.pth')
    path.append('/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/data/result/checkpoint/baseline/ensemble_gru_att2/1.pth')
    path.append('/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/data/result/checkpoint/baseline/ensemble_gru_att2/2.pth')
    path.append('/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/data/result/checkpoint/baseline/ensemble_gru_att2/3.pth')
    path.append('/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/data/result/checkpoint/baseline/ensemble_gru_att2/4.pth')
    path.append('/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/data/result/checkpoint/baseline/ensemble_gru_att2/5.pth')
    path.append('/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/data/result/checkpoint/baseline/ensemble_lstm/67.01.pth')
    path.append('/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/data/result/checkpoint/baseline/ensemble_lstm/66.88.pth')
    path.append('/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/data/result/checkpoint/baseline/ensemble_lstm/66.71.pth')
    path.append('/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/data/result/checkpoint/baseline/ensemble_lstm/66.66.pth')
    path.append('/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/data/result/checkpoint/baseline/ensemble_lstm/66.61.pth')
    path.append('/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/data/result/checkpoint/baseline/ensemble_gru/67.03.pth')
    path.append('/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/data/result/checkpoint/baseline/ensemble_gru/66.94.pth')
    path.append('/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/data/result/checkpoint/baseline/ensemble_gru/66.9.pth')
    path.append('/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/data/result/checkpoint/baseline/ensemble_gru/66.8.pth')
    path.append('/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/data/result/checkpoint/baseline/ensemble_gru/66.7.pth')
    '''
    #path.append('/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/data/result/vqa1/checkpoint_novg/gru/lr1e-4/train+val_lr1e-4_epoch1.pth')
    #path.append('/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/data/result/vqa1/checkpoint_novg/gru_att2/lr1e-4/train+val_lr1e-4_epoch1.pth')
    #path.append('/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/data/result/vqa1/checkpoint_novg/lstm/lr1e-4/train+val_lr1e-4_epoch1.pth')
    #path.append('/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/data/result/vqa1/checkpoint_novg/lstm_att2/lr1e-4/train+val_lr1e-4_epoch1.pth')
    path.append('/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/data/result/vqa1/checkpoint/gru/vg_removed/train+val_lr1e-4_epoch1.pth')
    path.append('/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/data/result/vqa1/checkpoint/lstm/vg_removed/train+val_lr1e-4_epoch1.pth')
    #path.append('/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/data/result/vqa1/checkpoint_nowe_novg/lstm_att2/2/lr1e-4/train+val_lr1e-4_epoch1.pth')
    file_name = '/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/experiments/result/test/vqa_OpenEnded_mscoco_test-dev2015_ensemble2_results'

    ensemble(path,mod,file_name)
