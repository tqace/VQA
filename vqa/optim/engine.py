import sys
sys.path.append('..')
import torch
from torch.autograd import Variable
import numpy as np
import  os
import json
from lib import utils
import ipdb
from datasets import vqa as vqadataset

def train(loader, model, criterion, optimizer, logger, epoch, opt):
  # set model to train mode
  model.train()
  display_interval = opt['optim']['display_interval']
  losses = np.zeros([display_interval]) 
  
  sub_epoch = [int(len(loader)/5),int(2*len(loader)/5),int(3*len(loader)/5),int(4*len(loader)/5)]
  for i, sample in enumerate(loader):  
      batch_size = sample['visual'].size(0)
      input_visual   = Variable(sample['visual'].cuda())
      input_question = Variable(sample['question'].cuda())
      target_answer  = Variable(sample['answer'].cuda())

      output = model(input_visual, input_question)
      loss = criterion(output, target_answer)

      # compute gradient and do SGD step
      optimizer.zero_grad()
      loss.backward()
      
      grad_clip_type = opt['optim']['grad_clip_type']
      if grad_clip_type == 'norm':
        torch.nn.utils.clip_grad_norm(model.parameters(), opt['optim']['grad_clip_value'])
      elif grad_clip_type == 'none':
        pass
      else:
        raise NotImplementedError

      optimizer.step()

      losses[i % display_interval] = loss.cpu().data[0]
      if i % display_interval == display_interval - 1:
        print('[Epoch/Iter %d/%d] Train loss:%f'%(epoch+1, i++1, losses.mean()))
        logger.info('%d/%d %f'%(epoch, i+1, losses.mean()))
      # snapshot  every 1/5 len of one epoch
      if opt['snapshot']['epoch1/5'] is True:
        if i in sub_epoch:
          epoch_part = str(sub_epoch.index(i)+1)+'_in_5_of_epoch'+str(epoch+1)
          utils.snapshot(model,epoch_part,opt)
      
      #snapshot every 200 iter
      if opt['snapshot']['iter200'] is True:
        if i%200 == 0 and i != 0:
          epoch_part = str(i)+'_in_5020_of_Epoch'+str(epoch+1)
          utils.snapshot(model,epoch_part,opt) 

def val(loader, model, criterion, optimizer, logger, epoch, opt):
  # set model to train mode
  model.eval()
  losses = [] 
  results = []
  for i, sample in enumerate(loader):
    utils.xprocess(i+1, len(loader))
    batch_size = sample['visual'].size(0)
    input_visual   = Variable(sample['visual'].cuda(), volatile=True)
    input_question = Variable(sample['question'].cuda(), volatile=True)
    target_answer  = Variable(sample['answer'].cuda(), volatile=True)

    output = model(input_visual, input_question)
    loss = criterion(output, target_answer)
    losses.append(loss.cpu().data[0])

    _, pred = output.data.cpu().max(1)
    pred.squeeze_()
    for j in range(batch_size):
      results.append({'question_id': sample['question_id'][j],
        'answer': loader.dataset.aid_to_ans[pred[j]]})
  fpath = os.path.join(opt['dirs']['result']['debug'], 'val_result_tmp.json')
  with open(fpath, 'w') as f:
    json.dump(results, f, indent=2)

  vqa_tools_dir = os.path.join(opt['dirs']['vqa_tool'], 'PythonHelperTools')
  vqa_eval_tools_dir = os.path.join(opt['dirs']['vqa_tool'], 'PythonEvaluationTools')
  if not vqa_tools_dir in sys.path:
    sys.path.append(vqa_tools_dir)
  if not vqa_eval_tools_dir in sys.path:
    sys.path.append(vqa_eval_tools_dir)
  from vqaEvaluation.vqaEval import VQAEval
  from vqaTools.vqa import VQA
 
  if opt['vqa']['type'] == 'vqa1':
    annFile = os.path.join(opt['dirs']['vqa1'], 'mscoco_val2014_annotations.json')
    quesFile = os.path.join(opt['dirs']['vqa1'], 'MultipleChoice_mscoco_val2014_questions.json')
  elif opt['vqa']['type'] == 'vqa2':
    annFile = os.path.join(opt['dirs']['vqa2'], 'v2_mscoco_val2014_annotations.json')
    quesFile = os.path.join(opt['dirs']['vqa2'], 'v2_OpenEnded_mscoco_val2014_questions.json')
  else:
    raise ValueError
  vqa = VQA(annFile, quesFile)
  vqaRes = vqa.loadRes(fpath, quesFile)
  vqaEval = VQAEval(vqa, vqaRes, n=2)
  vqaEval.evaluate()
  acc_overall = vqaEval.accuracy['overall']
  acc_perQuestionType = vqaEval.accuracy['perQuestionType']
  acc_perAnswerType = vqaEval.accuracy['perAnswerType']
  
  print('[Epoch %d] Test loss: %f, Test acc: %f'%(epoch+1, np.array(losses).mean(), acc_overall))
  logger.info('%d %f %f %s'%(epoch+1, np.array(losses).mean(), acc_overall, json.dumps(acc_perAnswerType)))

def test(loader, model, opt):
  # set model to train mode
  model.eval()
  results_testdev2015 = []
  results_test2015 = []
  save_prob = opt['test']['save_prob']
  if save_prob:
    probs_test2015 = []
    probs_testdev2015 = []
  for i, sample in enumerate(loader):
    utils.xprocess(i+1, len(loader))
    batch_size = sample['visual'].size(0)
    input_visual   = Variable(sample['visual'].cuda(), volatile=True)
    input_question = Variable(sample['question'].cuda(), volatile=True)
    input_data_split = sample['data_split']

    output = model(input_visual, input_question)

    probs = output.data.cpu().numpy()

    _, pred = output.data.cpu().max(1)
    pred.squeeze_()
    for j in range(batch_size):
      pred_ans = {'question_id': sample['question_id'][j],
        'answer': loader.dataset.aid_to_ans[pred[j]]}
      if input_data_split[j] == 'test2015':
        results_test2015.append(pred_ans)
        if save_prob:
          probs_test2015.append(probs[j, ...].copy())
      elif input_data_split[j] == 'testdev2015':
        results_testdev2015.append(pred_ans)
        if save_prob:
          probs_test2015.append(probs[j, ...].copy())
      else:
        raise ValueError
  results = {}
  results['ans_to_aid'] = loader.dataset.ans_to_aid
  results['aid_to_ans'] = loader.dataset.aid_to_ans
  results['test2015'] = results_test2015
  results['testdev2015'] = results_testdev2015
  if save_prob:
    results['probs_test2015'] = np.array(probs_test2015)
    results['probs_testdev2015'] = np.array(probs_testdev2015)
  return results

def test_arb(loader, model, opt):
  # set model to train mode
  model.eval()
  results_answer = []
  save_prob = opt['test']['save_prob']
  if save_prob:
    probs_answer = []
  for i, sample in enumerate(loader):
    utils.xprocess(i+1, len(loader))
    batch_size = sample['visual'].size(0)
    input_visual   = Variable(sample['visual'].cuda(), volatile=True)
    input_question = Variable(sample['question'].cuda(), volatile=True)

    output = model(input_visual, input_question)

    probs = output.data.cpu().numpy()

    _, pred = output.data.cpu().max(1)
    pred.squeeze_()
    for j in range(batch_size):
      pred_ans = {'question_id': sample['question_id'][j],
        'answer': loader.dataset.aid_to_ans[pred[j]]}
      results_answer.append(pred_ans)
      if save_prob:
        probs_answer.append(probs[j, ...].copy())
  results = {}
  results['ans_to_aid'] = loader.dataset.ans_to_aid
  results['aid_to_ans'] = loader.dataset.aid_to_ans
  results['test'] = results_answer
  if save_prob:
    results['probs'] = np.array(probs_answer)
  return results

def train_AC(loader, model, criterion, optimizer, logger, epoch, opt):
  # set model to train mode
  model.train()
  display_interval = opt['optim']['display_interval']
  losses = np.zeros([display_interval]) 
  for i, sample in enumerate(loader):
    batch_size = sample['visual'].size(0)
    input_visual   = Variable(sample['visual'].cuda())
    input_question = Variable(sample['question'].cuda())
    input_acs = Variable(sample['answer_candidates'].cuda())
    target_answer  = Variable(sample['answer'].cuda())

    output = model(input_visual, input_question, input_acs)
    loss = criterion(output, target_answer)

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
      
    grad_clip_type = opt['optim']['grad_clip_type']
    if grad_clip_type == 'norm':
      torch.nn.utils.clip_grad_norm(model.parameters(), opt['optim']['grad_clip_value'])
    elif grad_clip_type == 'none':
      pass
    else:
      raise NotImplementedError

    optimizer.step()

    losses[i % display_interval] = loss.cpu().data[0]
    if i % display_interval == display_interval - 1:
      print('[Epoch/Iter %d/%d] Train loss:%f'%(epoch, i++1, losses.mean()))
      logger.info('%d/%d %f'%(epoch, i+1, losses.mean()))

def val_AC(loader, model, criterion, optimizer, logger, epoch, opt):
  # set model to train mode
  model.eval()
  losses = [] 
  results = []
  for i, sample in enumerate(loader):
    utils.xprocess(i+1, len(loader))
    batch_size = sample['visual'].size(0)
    input_visual   = Variable(sample['visual'].cuda(), volatile=True)
    input_question = Variable(sample['question'].cuda(), volatile=True)
    input_acs = Variable(sample['answer_candidates'].cuda(), volatile=True)
    target_answer  = Variable(sample['answer'].cuda(), volatile=True)

    output = model(input_visual, input_question, input_acs)
    loss = criterion(output, target_answer)
    losses.append(loss.cpu().data[0])

    _, pred = output.data.cpu().max(1)
    pred.squeeze_()
    for j in range(batch_size):
      aid = input_acs.data[j, pred[j]]
      ans = loader.dataset.aid_to_ans[aid]
      results.append({'question_id': sample['question_id'][j],
        'answer': ans})
  fpath = os.path.join(opt['dirs']['result']['debug'], 'val_result_tmp.json')
  with open(fpath, 'w') as f:
    json.dump(results, f, indent=2)

  vqa_tools_dir = os.path.join(opt['dirs']['vqa_tool'], 'PythonHelperTools')
  vqa_eval_tools_dir = os.path.join(opt['dirs']['vqa_tool'], 'PythonEvaluationTools')
  if not vqa_tools_dir in sys.path:
    sys.path.append(vqa_tools_dir)
  if not vqa_eval_tools_dir in sys.path:
    sys.path.append(vqa_eval_tools_dir)
  from vqaEvaluation.vqaEval import VQAEval
  from vqaTools.vqa import VQA
 
  if opt['vqa']['type'] == 'vqa1':
    annFile = os.path.join(opt['dirs']['vqa1'], 'mscoco_val2014_annotations.json')
    quesFile = os.path.join(opt['dirs']['vqa1'], 'MultipleChoice_mscoco_val2014_questions.json')
  elif opt['vqa']['type'] == 'vqa2':
    annFile = os.path.join(opt['dirs']['vqa2'], 'v2_mscoco_val2014_annotations.json')
    quesFile = os.path.join(opt['dirs']['vqa2'], 'v2_OpenEnded_mscoco_val2014_questions.json')
  else:
    raise ValueError
  vqa = VQA(annFile, quesFile)
  vqaRes = vqa.loadRes(fpath, quesFile)
  vqaEval = VQAEval(vqa, vqaRes, n=2)
  vqaEval.evaluate()
  acc_overall = vqaEval.accuracy['overall']
  acc_perQuestionType = vqaEval.accuracy['perQuestionType']
  acc_perAnswerType = vqaEval.accuracy['perAnswerType']
  
  print('[Epoch %d] Test loss: %f, Test acc: %f'%(epoch+1, np.array(losses).mean(), acc_overall))
  logger.info('%d %f %f %s'%(epoch+1, np.array(losses).mean(), acc_overall, json.dumps(acc_perAnswerType)))

def test_AC(loader, model, opt):
  # set model to train mode
  model.eval()
  results_testdev2015 = []
  results_test2015 = []
  save_prob = opt['test']['save_prob']
  if save_prob:
    probs_test2015 = []
    probs_testdev2015 = []
  for i, sample in enumerate(loader):
    utils.xprocess(i+1, len(loader))
    batch_size = sample['visual'].size(0)
    input_visual   = Variable(sample['visual'].cuda(), volatile=True)
    input_question = Variable(sample['question'].cuda(), volatile=True)
    input_acs = Variable(sample['answer_candidates'].cuda(), volatile=True)
    input_data_split = sample['data_split']

    output = model(input_visual, input_question, input_acs)

    probs = output.data.cpu().numpy()

    _, pred = output.data.cpu().max(1)
    pred.squeeze_()
    for j in range(batch_size):
      aid = input_acs.data[j, pred[j]]
      ans = loader.dataset.aid_to_ans[aid]
      pred_ans = {'question_id': sample['question_id'][j],
        'answer': ans}
      if input_data_split[j] == 'test2015':
        results_test2015.append(pred_ans)
        if save_prob:
          probs_test2015.append(probs[j, ...].copy())
      elif input_data_split[j] == 'testdev2015':
        results_testdev2015.append(pred_ans)
        if save_prob:
          probs_test2015.append(probs[j, ...].copy())
      else:
        raise ValueError
  results = {}
  results['ans_to_aid'] = loader.dataset.ans_to_aid
  results['aid_to_ans'] = loader.dataset.aid_to_ans
  results['test2015'] = results_test2015
  results['testdev2015'] = results_testdev2015
  if save_prob:
    results['probs_test2015'] = np.array(probs_test2015)
    results['probs_testdev2015'] = np.array(probs_testdev2015)
  return results
