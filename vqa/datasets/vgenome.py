import json, os, sys
sys.path.append('..')
from collections import Counter
import re
from lib import utils

def vg_iid_2_vqa_iid(iid, opt):
  return iid + opt['vgenome']['base_image_id']

def vqa_iid_2_vg_iid(iid, opt):
  if iid < opt['vgenome']['base_image_id']:
    raise ValueError
  return iid - opt['vgenome']['base_image_id']

def vg_qid_2_vqa_qid(qid, opt):
  return qid + opt['vgenome']['base_question_id']

def vqa_qid_2_vg_qid(qid, opt):
  if qid < opt['vgenome']['base_question_id']:
    raise ValueError
  return qid - opt['vgenome']['base_question_id']

def interim(questions_annotations, opt):
  data = []
  print('Process raw vgenome data...')
  for i in range(len(questions_annotations)):
    utils.xprocess(i+1, len(questions_annotations))
    qa_img = questions_annotations[i]
    qa_img_id = qa_img['id']
    for j in range(len(qa_img['qas'])):
      qa = qa_img['qas'][j]
      row = {}
      row['question_id'] = vg_qid_2_vqa_qid(qa['qa_id'], opt)
      row['image_id'] = vg_iid_2_vqa_iid(qa_img_id, opt)
      row['data_split'] = 'vgenome'
      row['image_name'] = 'visual_genome_' + str(qa_img_id).zfill(12) + '.jpg'
      row['question'] = qa['question']
      answer = qa['answer'].lower()
      for i in [r'\?',r'\!',r'\'',r'\"',r'\$',r'\:',r'\@',r'\(',r'\)',r'\,',r'\.',r'\;']:
        answer = re.sub( i, '', answer)
      for i in [r'\-',r'\/']:
        answer = re.sub( i, ' ', answer)
      row['answer'] = answer
      answers = [answer]
      row['answers_occurence'] = Counter(answers).most_common()
      data.append(row)
  return data

