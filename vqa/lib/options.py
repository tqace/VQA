import argparse
import yaml
import json

def update_values(dict_from, dict_to):
  for key, value in dict_from.items():
    if isinstance(value, dict):
      update_values(dict_from[key], dict_to[key])
    elif value is not None:
      dict_to[key] = dict_from[key] 
  return dict_to

def merge_dict(a, b):
  if isinstance(a, dict) and isinstance(b, dict):
    d = dict(a)
    d.update({k: merge_dict(a.get(k, None), b[k]) for k in b})
  if isinstance(a, list) and isinstance(b, list):
    return b
    #return [merge_dict(x, y) for x, y in itertools.zip_longest(a, b)]
  return a if b is None else b

def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')  

def get_optiotns():
  parser = argparse.ArgumentParser(description="Option for VQA Model, QANet adam.")
  parser.add_argument('--name', type=str, 
      help='name to specify this process in runtime')
  parser.add_argument('--path_opt', default='/mnt/mfs3/yuyin/train/fujun/workspace/storage/VQA_1/options/default.yaml', type=str,
      help='path to a yaml options file')
  args = parser.parse_args()
  opt = {}
  with open(args.path_opt, 'r') as fin:
    print('Loading options from %s'%args.path_opt)
    opt_yaml = yaml.load(fin)
  opt = update_values(opt, opt_yaml)
  return opt 

