import os
import pickle
import torch
import torch.utils.data as data
import copy
import numpy as np
import copy
import json
import sys
sys.path.append('..')
from lib import utils
from . import nlp
from . import features
from . import vgenome

class AbstractVQADataset(data.Dataset):
  def __init__(self, mode, data_split, opt, dataset_img=None):
    super(AbstractVQADataset, self).__init__()
    self.mode = mode
    if self.mode not in ['train', 'val', 'test']:
      raise ValueError
    self.data_split = data_split
    if self.mode in ['train', 'val']:
      assert 'test' not in self.data_split

    #if self.mode == 'train':
    #  assert self.data_split == opt['coco']['train_split']
    #if self.mode == 'val':
    #  assert self.data_split == opt['coco']['val_split']
    #if self.mode == 'test':
    #  assert self.data_split == opt['coco']['test_split']

    self.opt = copy.deepcopy(opt)
    self.dataset_img = dataset_img

    self.dir_interim = os.path.join(self.opt['dirs']['intermedia'], 'interim')
    self._interim()

    self.dir_processed = os.path.join(self.opt['dirs']['intermedia'], 'processed')
    self._processed()

    with open(self.path_wid_to_word, 'rb') as handle:
      self.wid_to_word = pickle.load(handle)

    with open(self.path_word_to_wid, 'rb') as handle:
      self.word_to_wid = pickle.load(handle)

    with open(self.path_aid_to_ans, 'rb') as handle:
      self.aid_to_ans = pickle.load(handle)

    with open(self.path_ans_to_aid, 'rb') as handle:
      self.ans_to_aid = pickle.load(handle)

    with open(self.path_dataset, 'rb') as handle:
      self.raw_dataset = pickle.load(handle)
      self.dataset = self.raw_dataset

    self.AC_mode = False
   
  def _raw(self):
    raise NotImplementedError

  def _interim(self):
    raise NotImplementedError

  def _processed(self):
    raise NotImplementedError

  def __getitem__(self, index):
    raise NotImplementedError

  def open_AC_mode(self):
    raise NotImplementedError


class AbstractVQA(AbstractVQADataset):
  def __init__(self, mode, data_split, opt, dataset_img=None):
    super(AbstractVQA, self).__init__(mode, data_split, opt, dataset_img)
    if not self.mode == 'train':
      if self.opt['vqa']['target_answer'] == 'random_sample':
        self.opt['vqa']['target_answer'] = 'most_common'
  
  def __getitem__(self, index):
    item = {}

    # TODO: better handle cascade of dict items
    item_vqa = self.dataset[index]

    # Process Visual (image or features)
    if self.dataset_img is not None:
      item_img = self.dataset_img.get_by_name(item_vqa['image_name'])
      item['visual'] = item_img['visual']
    
    # Process Question (word token)
    item['question_id'] = item_vqa['question_id']
    item['question'] = torch.LongTensor(item_vqa['question_wids'])
    if self.AC_mode:
      item['answer_candidates'] = torch.LongTensor(item_vqa['answer_candidates_aids'])
    
    if self.mode == 'train':
    ## Process Answer if exists
      if self.opt['vqa']['target_answer'] == 'most_common':
        if self.AC_mode:
          ans_idx = None
          for i, ans in enumerate(item_vqa['answers']):
            if ans in item_vqa['answer_candidates']:
              ans_idx = i
          if ans_idx is None:
            raise Exception('This should not happen')
          item['answer'] = ans_idx
        else:
          item['answer'] = item_vqa['answer_aid']
      elif self.opt['vqa']['target_answer'] == 'random_sample':
        if self.AC_mode:
          ans_to_cnt = {ans:item_vqa['answers_count'][i] for i, ans in enumerate(item_vqa['answers'])}
          proba = []
          for ans in item_vqa['answer_candidates']:
            if ans in ans_to_cnt:
              proba.append(ans_to_cnt[ans])
            else:
              proba.append(0)
          proba = np.array(proba, dtype=np.float32)
          if np.sum(proba) == 0:
            raise Exception('This should not happen!')
          proba = proba / np.sum(proba)
          item['answer'] = int(np.random.choice(range(len(proba)), p=proba))
        else:
          proba = item_vqa['answers_count']
          if np.sum(proba) == 0:
            raise Exception('This should not happen!')
          proba = proba / np.sum(proba)
          item['answer'] = int(np.random.choice(item_vqa['answers_aid'], p=proba))
      elif self.opt['vqa']['target_answer'] == 'occurance_prob':
        if self.AC_mode:
          ans_to_cnt = {ans:item_vqa['answers_count'][i] for i, ans in enumerate(item_vqa['answers'])}
          proba = []
          for ans in item_vqa['answer_candidates']:
            if ans in ans_to_cnt:
              proba.append(ans_to_cnt[ans])
            else:
              proba.append(0)
          proba = np.array(proba, dtype=np.float32)
          if np.sum(proba) == 0:
            raise Exception('This should not happen!')
          proba = proba / np.sum(proba)
          item['answer'] = torch.from_numpy(proba)
        else:
          occurance_prob = torch.zeros(self.opt['vqa']['nans'])
          for i, aid in enumerate(item_vqa['answers_aid']):
            occurance_prob[aid] = item_vqa['answers_count'][i]
          if torch.sum(occurance_prob) == 0:
            raise Exception('This should not happen!')
          item['answer'] = occurance_prob / torch.sum(occurance_prob)
      else:
        raise NotImplementedError
    elif self.mode == 'val':
      if self.opt['vqa']['target_answer'] == 'most_common':
        if self.AC_mode:
          ans_idx = 0
          for i, ans in enumerate(item_vqa['answers']):
            if ans in item_vqa['answer_candidates']:
              ans_idx = i
          item['answer'] = ans_idx
        else:
          item['answer'] = item_vqa['answer_aid']
      elif self.opt['vqa']['target_answer'] == 'occurance_prob':
        if self.AC_mode:
          ans_to_cnt = {ans:item_vqa['answers_count'][i] for i, ans in enumerate(item_vqa['answers'])}
          proba = []
          for ans in item_vqa['answer_candidates']:
            if ans in ans_to_cnt:
              proba.append(ans_to_cnt[ans])
            else:
              proba.append(0)
          proba = np.array(proba, dtype=np.float32)
          if np.sum(proba) == 0:
            pass
          else:
            proba = proba / np.sum(proba)
          item['answer'] = torch.from_numpy(proba)
        else:
          occurance_prob = torch.zeros(self.opt['vqa']['nans'])
          for i, aid in enumerate(item_vqa['answers_aid']):
            occurance_prob[aid] = item_vqa['answers_count'][i]
          if torch.sum(occurance_prob) == 0:
            item['answer'] = occurance_prob
          else:
            item['answer'] = occurance_prob / torch.sum(occurance_prob)
    elif self.mode == 'test':
      item['data_split'] = item_vqa['data_split'] 
    else:
      raise ValueError
    return item

  def __len__(self):
    return len(self.dataset)

  def num_classes(self):
    return len(self.aid_to_ans)

  def vocab_words(self):
    return list(self.wid_to_word.values())

  def vocab_answers(self):
    return self.aid_to_ans

  def data_loader(self, batch_size=10, num_workers=4, shuffle=False):
    return data.DataLoader(self,
        batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True)

  def open_AC_mode(self):
    '''
    opening answer candidates mode
    '''
    # load answer candidates and update raw_dataset
    mode = self.mode
    data_split = self.opt['coco']['%s_split'%mode]
    fpath = os.path.join(self.opt['dirs']['intermedia'], 'acs', '%s_%s_answer_candidates_top10.json'%(mode, data_split))
    assert os.path.exists(fpath)
    with open(fpath) as f:
      acs = json.load(f)
    qid_to_acs = {a['question_id']:a for a in acs}
    for i, ex in enumerate(self.dataset):
      utils.xprocess(i+1, len(self.dataset))
      qid = ex['question_id']
      t_acs = qid_to_acs[qid]['answer_candidates']
      t_acs = [a['answer'] for a in t_acs]
      t_acs_aids = [self.ans_to_aid[ans] for ans in t_acs]
      ex['answer_candidates'] = t_acs
      ex['answer_candidates_aids'] = t_acs_aids
    # filter out samples whose answer candidates don't cover correct answer
    if mode == 'train': # Only filtering in train mode
      t_dataset = []
      for ex in self.dataset:
        if len(set(ex['answer_candidates']) & set(ex['answers'])) > 0:
          t_dataset.append(ex)
      if len(t_dataset) < len(self.dataset):
        print('Waring: Reduce the dataset from %d to %d'%(len(self.dataset), len(t_dataset)))
      self.dataset = t_dataset
    # set flag
    self.AC_mode = True
    print('Switch to Answer Candidate Mode.')

class VQA(AbstractVQA):
  def __init__(self, mode, data_split, opt, dataset_img=None):
    super(VQA, self).__init__(mode, data_split, opt, dataset_img)

  def _interim(self):
    if not os.path.exists(self.dir_interim):
      os.makedirs(self.dir_interim)
    self.path_qa = os.path.join(self.dir_interim, '%s_question_annotation.json'%(self.data_split))
    if not os.path.exists(self.path_qa):
      dataset = nlp.load_data(self.data_split, self.opt)
      with open(self.path_qa, 'w') as fout:
        json.dump(dataset, fout, indent=2)
  
  def _processed(self):
    if not os.path.exists(self.dir_processed):
      os.makedirs(self.dir_processed)
    self.path_wid_to_word = os.path.join(self.dir_processed, '%sset_wid_to_word.pickle'%(self.opt['coco']['train_split']))
    self.path_word_to_wid = os.path.join(self.dir_processed, '%sset_word_to_wid.pickle'%(self.opt['coco']['train_split']))
    self.path_aid_to_ans  = os.path.join(self.dir_processed,  '%sset_aid_to_ans.pickle'%(self.opt['coco']['train_split']))
    self.path_ans_to_aid  = os.path.join(self.dir_processed,  '%sset_ans_to_aid.pickle'%(self.opt['coco']['train_split']))
    if self.mode == 'train':
      self.path_dataset = os.path.join(self.dir_processed, 'trainmode_%sset.pickle'%(self.opt['coco']['train_split']))
    else:
      self.path_dataset = os.path.join(self.dir_processed, '%smode_%sset.pickle'%(self.mode, self.data_split))

    # select correct vocabe
    if not self.mode == 'train':
      if  self.opt['vgenome']['contrib_to_vocab']:
        print('Using extern vocabulary.')
        self.path_wid_to_word = self.path_wid_to_word.split('.pickle')[0] + '_ext.pickle'
        self.path_word_to_wid = self.path_word_to_wid.split('.pickle')[0] + '_ext.pickle'
        self.path_aid_to_ans  = self.path_aid_to_ans.split('.pickle')[0] + '_ext.pickle'
        self.path_ans_to_aid  = self.path_ans_to_aid.split('.pickle')[0] + '_ext.pickle'

    if False in [os.path.exists(f) for f in [self.path_wid_to_word, self.path_word_to_wid, self.path_aid_to_ans, self.path_ans_to_aid]]:
      if not self.mode == 'train':
        raise Exception('You must run Dataset for train data_split first!')
      else: # train mode
        examples = json.load(open(self.path_qa, 'r'))
        ans_to_aid, aid_to_ans = nlp.make_answer_vocab(examples, self.opt)
        examples = nlp.remove_examples(examples, ans_to_aid, self.opt)
        examples = nlp.tokenlize_questions(examples)
        examples, top_words = nlp.remove_long_tail_words(examples, self.opt['vqa']['minwcount'])
        wid_to_word = {i+1:w for i,w in enumerate(top_words)}
        word_to_wid = {w:i+1 for i,w in enumerate(top_words)}
        examples = nlp.encode_questions(examples, word_to_wid, self.opt['vqa']['maxlength'])
        examples = nlp.encode_answers(examples, ans_to_aid)
        examples = nlp.encode_answers_occurence(examples, ans_to_aid)
        with open(self.path_wid_to_word, 'wb') as fout:
          pickle.dump(wid_to_word, fout)
        with open(self.path_word_to_wid, 'wb') as fout:
          pickle.dump(word_to_wid, fout)
        with open(self.path_ans_to_aid, 'wb') as fout:
          pickle.dump(ans_to_aid, fout)
        with open(self.path_aid_to_ans, 'wb') as fout:
          pickle.dump(aid_to_ans, fout)
        with open(self.path_dataset, 'wb') as fout:
          pickle.dump(examples, fout)
    else:
      if not os.path.exists(self.path_dataset):
        if self.mode == 'train':
          raise Exception('Broken intermedia data! Please clear your intermedia dir and run again.')
        with open(self.path_word_to_wid, 'rb') as fin:
          word_to_wid = pickle.load(fin)
        with open(self.path_wid_to_word, 'rb') as fin:
          wid_to_word = pickle.load(fin)
        with open(self.path_ans_to_aid, 'rb') as fin:
          ans_to_aid = pickle.load(fin)
        with open(self.path_aid_to_ans, 'rb') as fin:
          aid_to_ans = pickle.load(fin)
        examples = json.load(open(self.path_qa, 'r'))
        examples = nlp.tokenlize_questions(examples)
        examples = nlp.remove_long_tail_test(examples, word_to_wid)
        examples = nlp.encode_questions(examples, word_to_wid, self.opt['vqa']['maxlength'])
        if self.mode == 'val':
          examples = nlp.encode_answers(examples, ans_to_aid)
          examples = nlp.encode_answers_occurence(examples, ans_to_aid)
        with open(self.path_dataset, 'wb') as fout:
          pickle.dump(examples, fout)

class VisualGenomeVQA(AbstractVQA):
  def __init__(self, mode, data_split, opt, dataset_vqa, dataset_img=None):
    assert 'vgenome' == data_split
    self.data_split = dataset_vqa.data_split + '+' + data_split
    self.dataset_vqa = dataset_vqa
    assert mode == dataset_vqa.mode
    if not mode == 'train':
      raise NotImplementedError
    super(VisualGenomeVQA, self).__init__(mode, self.data_split, opt, dataset_img)
   
  def _interim(self):
    fpath = os.path.join(self.opt['dirs']['vgenome'], 'question_answers.json')
    if not os.path.exists(self.dir_interim):
      os.makedirs(self.dir_interim)
    self.path_qa = os.path.join(self.dir_interim, 'vgenome_question_annotation.json')
    if not os.path.exists(self.path_qa):
      questions_annotations = json.load(open(fpath, 'r'))
      dataset = vgenome.interim(questions_annotations, self.opt)
      json.dump(dataset, open(self.path_qa, 'w'), indent=2)
  
  def _processed(self):
    if not os.path.exists(self.dir_processed):
      os.makedirs(self.dir_processed)
    need_update_vocab = False
    if not self.opt['vgenome']['contrib_to_vocab']:
      self.path_wid_to_word = self.dataset_vqa.path_wid_to_word 
      self.path_word_to_wid = self.dataset_vqa.path_word_to_wid
      self.path_aid_to_ans  = self.dataset_vqa.path_aid_to_ans
      self.path_ans_to_aid  = self.dataset_vqa.path_ans_to_aid
      self.path_dataset = self.dataset_vqa.path_dataset
      if False in [os.path.exists(f) for f in [self.path_wid_to_word, self.path_word_to_wid, self.path_aid_to_ans, self.path_ans_to_aid, self.path_dataset]]:
        raise Exception('VisualGenome dataset need vqa vocabulary.')
    else:
      print('Using extern vocabulary.')
      self.path_wid_to_word = self.dataset_vqa.path_wid_to_word.split('.pickle')[0] + '_ext.pickle'
      self.path_word_to_wid = self.dataset_vqa.path_word_to_wid.split('.pickle')[0] + '_ext.pickle'
      self.path_aid_to_ans  = self.dataset_vqa.path_aid_to_ans.split('.pickle')[0] + '_ext.pickle'
      self.path_ans_to_aid  = self.dataset_vqa.path_ans_to_aid.split('.pickle')[0] + '_ext.pickle'
      self.path_dataset  = self.dataset_vqa.path_dataset.split('.pickle')[0] + '_ext.pickle'
      if False in [os.path.exists(f) for f in [self.path_wid_to_word, self.path_word_to_wid, self.path_aid_to_ans, self.path_ans_to_aid]]:
        need_update_vocab = True
    if (not os.path.exists(self.path_dataset)) or need_update_vocab:
      wid_to_word = self.dataset_vqa.wid_to_word
      word_to_wid = self.dataset_vqa.word_to_wid
      aid_to_ans = self.dataset_vqa.aid_to_ans
      ans_to_aid = self.dataset_vqa.ans_to_aid
      examples = json.load(open(self.path_qa, 'r'))
      examples = nlp.remove_examples(examples, ans_to_aid, self.opt)
      examples = nlp.tokenlize_questions(examples)
      if need_update_vocab:
        raw_vocab_length = len(wid_to_word)
        word_to_wid_vg, wid_to_word_vg = nlp.make_question_vocab(examples, self.opt)
        wid = max(wid_to_word.keys()) + 1
        for word_vg in word_to_wid_vg:
          if not word_vg in word_to_wid:
            word_to_wid[word_vg] = wid
            wid_to_word[wid] = word_vg
            wid += 1
        assert len(wid_to_word) == len(word_to_wid)
        print('Vocabulary size extended from %d to %d'%(raw_vocab_length, len(wid_to_word)))
        with open(self.path_wid_to_word, 'wb') as fout:
          pickle.dump(wid_to_word, fout)
        with open(self.path_word_to_wid, 'wb') as fout:
          pickle.dump(word_to_wid, fout)
        with open(self.path_ans_to_aid, 'wb') as fout:
          pickle.dump(ans_to_aid, fout)
        with open(self.path_aid_to_ans, 'wb') as fout:
          pickle.dump(aid_to_ans, fout)
      # remove long tail words
      examples = nlp.remove_long_tail_test(examples, word_to_wid)
      examples = nlp.encode_questions(examples, word_to_wid, self.opt['vqa']['maxlength'])
      examples = nlp.encode_answers(examples, ans_to_aid)
      examples = nlp.encode_answers_occurence(examples, ans_to_aid)
      examples = self.dataset_vqa.dataset + examples
      with open(self.path_dataset, 'wb') as fout:
        pickle.dump(examples, fout)

def factory(mode, data_split, opt):
  image_dataset = features.factory(data_split, opt)
  if mode == 'train':
    vqa_split, vg_split = utils.parse_data_split(data_split, opt)
  else:
    vqa_split, vg_split = data_split, None
  if vqa_split is not None:
    dataset_vqa = VQA(mode, vqa_split, opt, image_dataset)
  else:
    dataset_vqa = None

  if vg_split is None:
    assert dataset_vqa is not None
    return dataset_vqa
  else:
    assert dataset_vqa is not None
    dataset_vg = VisualGenomeVQA(mode, vg_split, opt, dataset_vqa, image_dataset)
    return dataset_vg
