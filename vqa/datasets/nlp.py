import sys
sys.path.append('..')
import numpy as np
import re, json, random, time, os
from lib import utils
from collections import Counter

def load_vqa1_json(data_split, opt):
    """
    Parses the question and answer json files for the given data split. 
    Returns the question dictionary and the answer dictionary.
    """
    assert data_split in ['train2014', 'val2014', 'test2015', 'testdev2015']
    file_tag = data_split
    if file_tag == 'testdev2015':
      file_tag = 'test-dev2015'

    coco_split = data_split
    if coco_split in ['test-dev2015', 'test2015', 'testdev2015']:
      coco_split = 'test2015'

    ques_file = 'OpenEnded_mscoco_%s_questions.json'%(file_tag)
    if data_split in ['train2014', 'val2014']:
      ans_file = 'mscoco_%s_annotations.json'%(file_tag)

    dataset = {}
    with open(os.path.join(opt['dirs']['vqa1'], ques_file), 'r') as f:
      qdata = json.load(f)['questions']
      print('Loading %s'%ques_file)
      for i, q in enumerate(qdata):
        utils.xprocess(i+1, len(qdata))
        dataset[q['question_id']] = \
            {
                'question': q['question'], 
                'image_id': q['image_id'], 
                'question_id': q['question_id'],
                'data_split': data_split,
                'image_name' : 'COCO_%s_%012d.jpg'%(coco_split, q['image_id'])}
    if data_split in ['train2014', 'val2014']:
      with open(os.path.join(opt['dirs']['vqa1'], ans_file), 'r') as f:
        print('Loading %s'%ans_file)
        adata = json.load(f)['annotations']
        for i, a in enumerate(adata):
          utils.xprocess(i+1, len(adata))
          dataset[a['question_id']]['answer'] = a['multiple_choice_answer']
          answers = [x['answer'] for x in a['answers']]
          dataset[a['question_id']]['answers_occurence'] = Counter(answers).most_common()
    dataset = sorted(dataset.values(), key=lambda x:x['question_id'])
    return dataset

def load_vqa2_json(data_split, opt):
    """
    Parses the question and answer json files for the given data split. 
    Returns the question dictionary and the answer dictionary.
    """
    assert data_split in ['train2014', 'val2014', 'test2015', 'testdev2015']
    file_tag = data_split
    if file_tag == 'testdev2015':
      file_tag = 'test-dev2015'

    coco_split = data_split
    if coco_split in ['test-dev2015', 'test2015', 'testdev2015']:
      coco_split = 'test2015'

    ques_file = 'v2_OpenEnded_mscoco_%s_questions.json'%(file_tag)
    if data_split in ['train2014', 'val2014']:
      ans_file = 'v2_mscoco_%s_annotations.json'%(file_tag)

    dataset = {}
    with open(os.path.join(opt['dirs']['vqa2'], ques_file), 'r') as f:
      qdata = json.load(f)['questions']
      print('Loading %s'%ques_file)
      for i, q in enumerate(qdata):
        utils.xprocess(i+1, len(qdata))
        dataset[q['question_id']] = \
            {
                'question': q['question'], 
                'image_id': q['image_id'], 
                'question_id': q['question_id'],
                'data_split': data_split,
                'image_name' : 'COCO_%s_%012d.jpg'%(coco_split, q['image_id'])}
    if data_split in ['train2014', 'val2014']:
      with open(os.path.join(opt['dirs']['vqa2'], ans_file), 'r') as f:
        print('Loading %s'%ans_file)
        adata = json.load(f)['annotations']
        for i, a in enumerate(adata):
          utils.xprocess(i+1, len(adata))
          dataset[a['question_id']]['answer'] = a['multiple_choice_answer']
          answers = [x['answer'] for x in a['answers']]
          dataset[a['question_id']]['answers_occurence'] = Counter(answers).most_common()
    dataset = sorted(dataset.values(), key=lambda x:x['question_id'])
    return dataset

def load_data(data_splits, opt):
  data_splits = data_splits.split('+')
  dataset = []
  for ds in data_splits:
    if opt['vqa']['type'] == 'vqa1':
      dataset += load_vqa1_json(ds, opt)
    elif opt['vqa']['type'] == 'vqa2':
      dataset += load_vqa2_json(ds, opt)
    else:
      raise ValueError
  dataset = sorted(dataset, key=lambda x: x['question_id'])
  return dataset

def seq_to_list(s):
  t_str = s.lower()
  for i in [r'\?',r'\!',r'\'',r'\"',r'\$',r'\:',r'\@',r'\(',r'\)',r'\,',r'\.',r'\;']:
    t_str = re.sub( i, '', t_str)
  for i in [r'\-',r'\/']:
    t_str = re.sub( i, ' ', t_str)
  q_list = re.sub(r'\?','',t_str.lower()).split(' ')
  q_list = filter(lambda x: len(x) > 0, q_list)
  return list(q_list)

def make_answer_vocab(dataset, opt):
  """
  Returns a dictionary that maps words to indices.
  """
  vocab_size = opt['vqa']['nans']
  nadict = {'':100000}
  # vid = 1
  for ex in dataset:
    answer_obj = ex['answers_occurence']
    for ans in answer_obj:
      # create dict
      nadict[''] += ans[1]  # make sure '' is the most frequent ans
      if ans[0] in nadict:
        nadict[ans[0]] += ans[1]
      else:
        nadict[ans[0]] = ans[1]
  nalist = []
  for k,v in sorted(nadict.items(), key=lambda x:(-x[1])):
    nalist.append((k,v))
  ans_to_index = {}
  index_to_ans = {}
  for i, w in enumerate(nalist[:vocab_size]):
    ans_to_index[w[0]] = i
    index_to_ans[i] = w[0]
  
  return ans_to_index, index_to_ans

def make_question_vocab(dataset, opt):
  """
  Returns a dictionary that maps words to indices.
  """
  nvdict = {'UNK':10000}
  for ex in dataset:
    # sequence to list
    q_str = ex['question']
    q_list = seq_to_list(q_str)
    # create dict
    for w in q_list:
      nvdict['UNK'] += 1 # make sure 'UNK' is the most frequent word
      if w not in nvdict:
        nvdict[w] = 1
      else:
        nvdict[w] += 1
  nvlist = []
  for k, v in sorted(nvdict.items(), key=lambda x:(-x[1])):
    nvlist.append((k, v))
  word_to_index = {w[0]:i+1 for i, w in enumerate(nvlist)}
  index_to_word = {i+1:w[0] for i, w in enumerate(nvlist)}
  return word_to_index, index_to_word

def remove_examples(examples, ans_to_inx, opt):
  new_examples = []
  ans_voc = set(ans_to_inx.keys())
  for ex in examples:
    ans = set([a[0] for a in ex['answers_occurence']])
    if len(ans & ans_voc) == 0:
      continue
    new_examples.append(ex)
  print('Number of examples reduced from %d to %d '%(len(examples), len(new_examples)))
  return new_examples

def tokenlize_questions(examples):
  print('Example of generated tokens after preprocessing some questions:')
  for i, ex in enumerate(examples):
    s = ex['question']
    ex['question_words'] = seq_to_list(s)
    if i < 10:
      print(ex['question_words'])
    else:
      utils.xprocess(i+1, len(examples))
  return examples

def remove_long_tail_words(examples, minwcount=0):
  counts = {}
  for ex in examples:
    for w in ex['question_words']:
      counts[w] = counts.get(w, 0) + 1
  cw = sorted([(count,w) for w, count in counts.items()], reverse=True)
  print('Top words and their counts:')
  print('\n'.join(map(str,cw[:20])))
  total_words = sum(counts.values())
  print('Total words:', total_words)
  bad_words = [w for w,n in counts.items() if n <= minwcount]
  vocab     = [w for w,n in counts.items() if n > minwcount]
  bad_count = sum(counts[w] for w in bad_words)
  print('Number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
  print('Number of words in vocab would be %d' % (len(vocab), ))
  print('Number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))
  print('Insert the special UNK token')
  vocab.append('UNK')
  for ex in examples:
    words = ex['question_words']
    question = [w if counts.get(w,0) > minwcount else 'UNK' for w in words]
    ex['question_words_UNK'] = question
  return examples, vocab

def remove_long_tail_test(examples, word_to_wid):
  for ex in examples:
    ex['question_words_UNK'] = [w if w in word_to_wid else 'UNK' for w in ex['question_words']]
  return examples

def encode_questions(examples, word_to_wid, maxlength):
  for i, ex in enumerate(examples):
    ex['question_length'] = min(maxlength, len(ex['question_words_UNK']))
    ex['question_wids'] = [0]*maxlength
    for k, w in enumerate(ex['question_words_UNK']):
      if k < maxlength:
        ex['question_wids'][k] = word_to_wid[w]
  return examples

def encode_answers(examples, ans_to_aid):
  print('Warning: aid of answer not in vocab is 0')
  for i, ex in enumerate(examples):
    ex['answer_aid'] = ans_to_aid.get(ex['answer'], 0)
  return examples

def encode_answers_occurence(examples, ans_to_aid):
  for i, ex in enumerate(examples):
    answers = []
    answers_aid = []
    answers_count = []
    for ans in ex['answers_occurence']:
       aid = ans_to_aid.get(ans[0], -1) # -1 means answer not in vocab
       if aid != -1:
         answers.append(ans[0])
         answers_aid.append(aid) 
         answers_count.append(ans[1])
    ex['answers']       = answers
    ex['answers_aid']   = answers_aid
    ex['answers_count'] = answers_count
  return examples

