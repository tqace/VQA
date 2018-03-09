from vqa.lib import options
from vqa.datasets import features
import json
from vqa.datasets import vqa as vqadataset
import numpy as np
from vqa.lib import utils

opt = options.get_optiotns()
print('Building train')
trainset = vqadataset.factory('train', opt['coco']['train_split'], opt)
trainloader = trainset.data_loader(batch_size=opt['optim']['batch_size'],
    num_workers = opt['optim']['workers'],
    shuffle=True)
print(len(trainloader))
trainloader.dataset.open_AC_mode()
print(len(trainloader))
for i, sample in enumerate(trainloader):
  print(i)
valset = vqadataset.factory('val', opt['coco']['val_split'], opt)
valloader = valset.data_loader(batch_size=opt['optim']['batch_size'],
    num_workers = opt['optim']['workers'],
    shuffle=True)
print(len(valloader))
valloader.dataset.open_AC_mode()
print(len(valloader))
for i, sample in enumerate(valloader):
  print(i)
testset = vqadataset.factory('test', opt['coco']['test_split'], opt)
testloader = testset.data_loader(batch_size=opt['optim']['batch_size'],
    num_workers = opt['optim']['workers'],
    shuffle=True)
print(len(testloader))
testloader.dataset.open_AC_mode()
print(len(testloader))
for i, sample in enumerate(testloader):
  print(i)
#print(len(trainset))
#valset = vqadataset.factory('val', opt['coco']['val_split'], opt)
#print(len(valset))
#print(train_features[100])
#print(len(train_features))
#trainset = vqadataset.VQA2(mode='train', data_split=opt['coco']['train_split'], opt=opt, dataset_img=train_features)
#for i in range(len(trainset)):
#  print(i)
#  sample = trainset[i]
#  img_name = sample['vqa']['image_name']
#  visual = sample['visual'].numpy().astype(np.float32)
#  feature = train_features.get_by_name(img_name)['visual'].numpy().astype(np.float32)
#  raw_feature = np.load('/data/zwfang/VQA/QANet/answer_subset/QANet_adam_v39/data/resource/features/train2014/' + img_name + '.npz')['x']
#  d1 = (visual - raw_feature).sum()
#  d2 = (feature - raw_feature).sum()
#  if d1 > 0.01 or d2 > 0.01:
#    print(sample['vqa'])
#    print(visual.mean(), feature.mean(), raw_feature.mean())
#    input()
#trainloader = trainset.data_loader(batch_size=opt['optim']['batch_size'],
#    num_workers = opt['optim']['workers'],
#    shuffle=True)
#for i, sample in enumerate(trainloader):
#  print(i)
#  batch_size = sample['visual'].size(0)
#  visual = sample['visual']
#  qid = sample['question_id']
#  que = sample['question']
#  ans = sample['answer']
#  img_names = sample['image_name']
#  for bs in range(batch_size):
#    feature = visual[bs, ...].numpy()
#    img_name = img_names[bs]
#    raw_feature = np.load('/data/zwfang/VQA/QANet/answer_subset/QANet_adam_v39/data/resource/features/train2014/' + img_name + '.npz')['x']
#    if (feature - raw_feature).sum() > 0.001:
#      print(img_name, qid[bs])
#      print(raw_feature.sum(), feature.sum())
#      print(feature.mean(), raw_feature.mean(), (feature - raw_feature).mean())
#      input()
#
#val_features = features.factory(opt['coco']['val_split'], opt)
#valset = vqadataset.VQA2(mode='val', data_split=opt['coco']['val_split'], opt=opt, dataset_img=val_features)
#print(valset[100])
#
#test_features = features.factory(opt['coco']['test_split'], opt)
#testset = vqadataset.VQA2(mode='test', data_split=opt['coco']['test_split'], opt=opt, dataset_img=test_features)
#print(testset[100])
utils.wait_for_quit()
