import sys
sys.path.append('./vqa')
from vqa.lib import options
from vqa.datasets import features
import json
from vqa.datasets import vqa as vqadataset
from vqa.models import teney_lstm
from vqa.models import teney_gru
from vqa.models import teney_gru_att2
from vqa.models import teney_lstm_att2
from vqa.lib import utils
from vqa import optim
import torch
from vqa.optim import engine
import os

def trainval(opt):
  # Build result dirs
  utils.prepare_environment(opt)
  
  # Build Dataloader
  print('Building trainloader ...')
  trainset = vqadataset.factory(mode='train', data_split=opt['coco']['train_split'], opt=opt)
  trainloader = trainset.data_loader(batch_size=opt['optim']['batch_size'],
      num_workers = opt['optim']['workers'],
      shuffle=True)
  if opt['optim']['validation']['interval'] > 0:
    print('Building valdataloader ...')
    valset = vqadataset.factory(mode='val', data_split=opt['coco']['val_split'], opt=opt)
    valloader = valset.data_loader(batch_size=opt['optim']['batch_size'],
        num_workers=opt['optim']['workers'],
        shuffle=False)

  # Building model
  if opt['model']['type'] == 'teney':
    print('Model type: %s'%opt['model']['type'])
    model = teney.TeneyNet(len(trainloader.dataset.wid_to_word), len(trainloader.dataset.aid_to_ans), opt)
  else:
    raise NotImplementedError
  if opt['optim']['resume'] == 'none':
    print('Building model from scratch ...')
    # init lookup table
    if not opt['model']['init_with_glove'] == 'none':
      print('Init some params with GolVe: %s ...'%opt['model']['init_with_glove'])
      glove_matrix = utils.get_glove_matrix_for_vocab(trainloader.dataset.word_to_wid, opt, start_index=1)
      model.lookup_table.weight.data[...] = torch.from_numpy(glove_matrix)
      glove_matrix = utils.get_glove_matrix_for_vocab(trainloader.dataset.ans_to_aid, opt, start_index=0)
      model.iq_branch1_linear.weight.data[...] = torch.from_numpy(glove_matrix)
  else:
    print('Resume from checkpoint %s'%opt['optim']['resume'])
    ckpt = torch.load(opt['optim']['resume'])
    if ckpt['device'] == 'gpu':
      model.cuda()
    model.load_state_dict(ckpt['model'])
  model.cuda()
  if opt['optim']['use_cudnn_benchmark']:
    torch.backends.cudnn.benchmark = True
  
  # Building Logger
  print('Building logger ...')
  logger = utils.Logger(opt) 

  # optimizer
  print('Building optimizer')
  optimizer = optim.get_optimizer(model, opt)
  criterion = optim.get_criterion(opt)

  # main loop
  for epoch in range(opt['optim']['max_epoch']):
    engine.train(trainloader, model, criterion, optimizer, logger.train_logger, epoch, opt) 

    validation_interval = opt['optim']['validation']['interval']
    if validation_interval > 0 and epoch % validation_interval == validation_interval - 1:
      engine.val(valloader, model, criterion, optimizer, logger.val_logger, epoch, opt)
  
    snapshot_interval = opt['snapshot']['interval']
    if snapshot_interval > 0 and epoch % snapshot_interval == snapshot_interval - 1:
      utils.snapshot(model, epoch + 1, opt)
  
def traintest(opt):
  # Build result dirs
  utils.prepare_environment(opt)
  
  # Build Dataloader
  print('Building trainloader ...')
  trainset = vqadataset.factory(mode='train', data_split=opt['coco']['train_split'], opt=opt)
  trainloader = trainset.data_loader(batch_size=opt['optim']['batch_size'],
      num_workers = opt['optim']['workers'],
      shuffle=True)
  print('Building testdataloader ...')
  testset = vqadataset.factory(mode='test', data_split=opt['coco']['test_split'], opt=opt)
  testloader = testset.data_loader(batch_size=opt['optim']['batch_size'],
      num_workers=opt['optim']['workers'],
      shuffle=False)

  # Building model
  len_nVoc=len(trainloader.dataset.wid_to_word)
  if opt['vgenome']['contrib_to_vocab'] and 'vgenome' not in opt['coco']['train_split']:
    len_nVoc+=3717
  print('Model type: %s'%opt['model']['type'])
  if opt['model']['type'] == 'teney_gru':
    model = teney_gru.TeneyNet_GRU(len_nVoc, len(trainloader.dataset.aid_to_ans), opt)
  elif opt['model']['type'] == 'teney_lstm':
    model = teney_lstm.TeneyNet_LSTM(len_nVoc, len(trainloader.dataset.aid_to_ans), opt)
  elif opt['model']['type'] == 'teney_gru_att2':
    model = teney_gru_att2.TeneyNet_GRU_ATT2(len_nVoc, len(trainloader.dataset.aid_to_ans), opt)
  elif opt['model']['type'] == 'teney_lstm_att2':
    model = teney_lstm_att2.TeneyNet_LSTM_ATT2(len_nVoc, len(trainloader.dataset.aid_to_ans), opt)
  else:
    raise NotImplementedError
  if opt['optim']['resume'] == 'none':
    print('Building model from scratch ...')
    # init lookup table
    if not opt['model']['init_with_glove'] == 'none':
      print('Init some params with GolVe: %s ...'%opt['model']['init_with_glove'])
      glove_matrix = utils.get_glove_matrix_for_vocab(trainloader.dataset.word_to_wid, opt, start_index=1)
      model.lookup_table.weight.data[...] = torch.from_numpy(glove_matrix)
      glove_matrix = utils.get_glove_matrix_for_vocab(trainloader.dataset.ans_to_aid, opt, start_index=0)
      model.iq_branch1_linear.weight.data[...] = torch.from_numpy(glove_matrix)
  else:
    print('Resume from checkpoint %s'%opt['optim']['resume'])
    ckpt = torch.load(opt['optim']['resume'])
    if ckpt['device'] == 'gpu':
      model.cuda()
    model.load_state_dict(ckpt['model'])
  model.cuda()
  if opt['optim']['use_cudnn_benchmark']:
    torch.backends.cudnn.benchmark = True
  
  # Building Logger
  print('Building logger ...')
  logger = utils.Logger(opt) 

  # optimizer
  print('Building optimizer')
  optimizer = optim.get_optimizer(model, opt)
  criterion = optim.get_criterion(opt)

  # main loop
  for epoch in range(opt['optim']['max_epoch']):
    engine.train(trainloader, model, criterion, optimizer, logger.train_logger, epoch, opt) 

    snapshot_interval = opt['snapshot']['interval']
    if snapshot_interval > 0 and epoch % snapshot_interval == snapshot_interval - 1:
      utils.snapshot(model, epoch + 1, opt)
    
    if (epoch + 1) in utils.test_epoches(opt):
      print('Testing for %s'%(opt['coco']['test_split']))
      results = engine.test(testloader, model, opt)
      utils.save_test_results(results, epoch, opt)

def test(opt):
  raise NotImplementedError

if __name__ == '__main__':
  opt = options.get_optiotns()
  mode = opt['mode']['mode']
  if mode == 'trainval':
    trainval(opt)
  elif mode == 'traintest':
    traintest(opt)
  elif mode == 'test':
    test(opt)
  else:
    raise NotImplementedError
