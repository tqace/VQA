import torch
from torch import nn
import spacy
import numpy as np
import os, sys
import logging
import pickle
import json
def xprocess(cur_step, total_steps, end_with_clean=True, end_log=None):
  if cur_step <0 or cur_step > total_steps:
    return
  cols = os.get_terminal_size().columns
  processed_char = '>'
  unprocessed_char = ' '
  output = '['
  rate = cur_step / float(total_steps)
  xrate = '%.2f'%(rate * 100) + '%'
  nXChar = cols - len(xrate) - 2
  nProChar = int(nXChar * rate + 0.5)
  nUnproChar = nXChar - nProChar
  output += processed_char * nProChar
  output += unprocessed_char * nUnproChar
  output += ']' 
  output += xrate
  if cur_step == total_steps:
    if end_with_clean:
      output = ' ' * len(output) + '\r'
    else:
      output += '\n'
    if end_log is not None:
      print(end_log)
  else:
    output += '\r'
  sys.stdout.write(output)
  sys.stdout.flush()

def clip_grad(parameters, max_grad):
  parameters = list(filter(lambda p: p.grad is not None, parameters))
  max_grad = float(max_grad)
  for param in parameters:
    param.grad.data.clamp_(-max_grad, max_grad)

def select_hiddens_with_index(in_hiddens, idx):
  '''
  select hidden from rnn hiddens
  in_hiddens: (N, T, hidden_size), the firsr output of a rnn.
  idx: (N), selected time-step index.
  return:
  ret_hiddens: (N, hidden_size)
  '''
  idx = list(idx.data)
  if not len(idx) == in_hiddens.size(0):
    raise Exception('lenght of idx must be equal to in_hiddens.size(0)')
  selected_hiddens = []
  for n, i in enumerate(idx):
    selected_hiddens.append(in_hiddens[n, i, :].unsqueeze(0))
  return torch.cat(selected_hiddens, 0)

class GlovecStf(object):
  def __init__(self, vectors_file):
    with open(vectors_file, 'r') as f:
      vectors = {}
      for line in f:
        vals = line.rstrip().split(' ')
        vectors[vals[0]] = [float(x) for x in vals[1:]]
    vocab_size = len(vectors)
    words = sorted(list(vectors.keys()))
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}
    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
      if word == '<unk>':
        continue
      W[vocab[word], :] = v
    self.W = W
    self.vector_dim = vector_dim
    self.vocab = vocab
    self.ivocab = ivocab

  def __call__(self, words):
    wvec = np.zeros([self.vector_dim])
    words = words.split(' ')
    n_valid_words = 0
    for word in words:
      if word in self.vocab:
        n_valid_words += 1
        wvec += self.W[self.vocab[word], :]
    if not n_valid_words == 0:
      wvec /= n_valid_words
    return wvec

def get_glove_matrix_for_vocab(vocab_dict, opt, start_index=0):
  n_vocab = len(vocab_dict)
  if opt['model']['init_with_glove'] == 'google':
    nlp = spacy.load('en')
    res = np.zeros([n_vocab + start_index, nlp.vocab.vectors_length])
    for k, v in vocab_dict.items():
      res[v, :] = nlp(u'%s'%k).vector
  elif opt['model']['init_with_glove'] == 'stanford':
    nlp = GlovecStf(os.path.join(opt['dirs']['resource'], 'glove_stanford.txt'))
    res = np.zeros([n_vocab + start_index, nlp.vector_dim])
    for k, v in vocab_dict.items():
      res[v, :] = nlp('%s'%k)
  else:
    raise NotImplementedError
  return res

def prepare_environment(opt):
  # result dirs
  for k, v in opt['dirs']['result'].items():
    if not os.path.exists(v):
      os.makedirs(v)

class Logger(object):
  def __init__(self, opt, **kargs):
    self.opt = opt
    self.g_formatter = logging.Formatter('[%(asctime)s] %(message)s')
    self.g_seperator = '============='
    self.g_level = logging.INFO
    self.train_logger_name = 'train.log'
    self.val_logger_name = 'val.log'
    self.test_logger_name = 'test.log'
    self.train_logger = self.__init_logger('train', self.train_logger_name)
    self.val_logger = self.__init_logger('val', self.val_logger_name)
    self.test_logger = self.__init_logger('test', self.test_logger_name)
  
  def __init_logger(self, mode, logger_name):
    logger = logging.getLogger(mode)
    logger.setLevel(self.g_level)
    fpath = os.path.join(self.opt['dirs']['result']['log_dir'], logger_name)
    fh = logging.FileHandler(fpath)
    fh.setFormatter(self.g_formatter)
    logger.addHandler(fh)
    logger.info(self.g_seperator)
    return logger

def snapshot(model, epoch, opt, **kargs):
  opt_snapshot = opt['snapshot']
  prefix = opt_snapshot['prefix']
  format_epoch = epoch
  if prefix == '':
    prefix = os.path.join(opt['dirs']['result']['checkpoint'], 'epoch%s.pth')
  else:
    prefix = os.path.join(opt['dirs']['result']['checkpoint'], prefix + '_epoch%s.pth')
  fpath = prefix % format_epoch
  state_dict = {}
  if opt_snapshot['device'] == True:
    if opt_snapshot['force_to_cpu'] == True:
      state_dict['device'] = 'cpu'
    else:
      state_dict['device'] = 'gpu'
  if opt_snapshot['model'] == True:
    state_dict_model = model.state_dict()
    if opt_snapshot['force_to_cpu'] == True:
      for k, v in state_dict_model.items():
        state_dict_model[k] = v.cpu()
    state_dict['model'] = state_dict_model
  if opt_snapshot['optimizer'] == True:
    raise NotImplementedError
  torch.save(state_dict, fpath)

def parse_data_split(data_split, opt):
  data_splits = data_split.split('+')
  vqa_split = None
  vg_split = None
  for ds in data_splits:
    if ds in ['train2014', 'val2014', 'test2015']:
      if vqa_split is None:
        vqa_split = ds
      else:
        vqa_split += ('+' + ds)
    elif ds in ['vgenome']:
      vg_split = ds
    else:
      raise ValueError
  return vqa_split, vg_split

def test_epoches(opt):
  epoch_type = opt['test']['models']['type']
  if epoch_type == 'model_path':
    raise Exception('Cannot judge this type')
  if epoch_type == 'epoch':
    test_epoches = opt['test']['models']['epoch']
  elif epoch_type == 'interval':
    interval = opt['test']['models']['interval']
    test_epoches = range(1, opt['optim']['max_epoch'] + 1, interval)
  elif epoch_type == 'range':
    start, end, interval = opt['test']['models']['range']
    test_epoches = range(start, end, interval)
  else:
    raise NotImplementedError
  return list(test_epoches)

def save_test_results(results, epoch, opt):
  results_test2015_path = 'Result_Train_%s_Test_%s_Epoch_%d.json'%(opt['coco']['train_split'], 'test2015', epoch+1)
  results_test2015_path = os.path.join(opt['dirs']['result']['test'], results_test2015_path)
  results_testdev2015_path = 'Result_Train_%s_Test_%s_Epoch_%d.json'%(opt['coco']['train_split'], 'testdev2015', epoch+1)
  results_testdev2015_path = os.path.join(opt['dirs']['result']['test'], results_testdev2015_path)
  if len(results['test2015']) > 0:
    with open(results_test2015_path, 'w') as f:
      json.dump(results['test2015'], f, indent=2)
  if len(results['testdev2015']) > 0:
    with open(results_testdev2015_path, 'w') as f:
      json.dump(results['testdev2015'], f, indent=2)
  save_prob = opt['test']['save_prob']
  if save_prob:
    results_path = 'Result_Train_%s_Test_%s_Epoch_%d.pickle'%(opt['coco']['train_split'], opt['coco']['test_split'], epoch+1)
    results_path = os.path.join(opt['dirs']['result']['test'], results_path)
    with open(results_path, 'wb') as f:
      pickle.dump(results, f)

def wait_for_quit():
  print('Type "quit" to exit the process...')
  while True:
    c = input()
    if c == 'quit':
      return
    else:
      print('Unknown commoand: %s'%(c))
      print('Type "quit" to exit the process...')

if __name__ == '__main__':
  nlp = GlovecStf('/mnt/mfs3/yuyin/train/zhouwang/dataset/GloVe/glove.6B.300d.txt')
  print(nlp.vector_dim)
  print(nlp('I love dog'))
  print(nlp.W.shape)

if __name__ == '#__main__':
  D = torch.autograd.Variable(torch.LongTensor([
    [1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 0]
    ])).cuda()
  H = torch.autograd.Variable(torch.FloatTensor(4, 6, 3)).cuda()
  idx = D.sum(1)
  out = select_hiddens_with_index(H, idx-1)
  print(out)
