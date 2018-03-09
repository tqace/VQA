''' Build the network for vqa
'''
import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib import utils

class GatedTanh(nn.Module):
  def __init__(self, in_features, out_features):
    super(GatedTanh, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.linear1 = nn.Linear(in_features, out_features)
    self.linear2 = nn.Linear(in_features, out_features)
  
  def forward(self, x):
    out = F.tanh(self.linear1(x))
    gate = F.sigmoid(self.linear2(x))
    out = out * gate
    return out

class GatedTanhConv(nn.Module):
  def __init__(self, in_features, out_features):
    super(GatedTanhConv, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.conv1 = nn.Conv2d(in_features, out_features, kernel_size=1)
    self.conv2 = nn.Conv2d(in_features, out_features, kernel_size=1)

  def forward(self, x):
    out = F.tanh(self.conv1(x))
    gate = F.sigmoid(self.conv2(x))
    out = out * gate
    return out
  
class LSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
    super(LSTM, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dropout = dropout
    self.rnn = nn.LSTM(
        input_size = input_size,
        hidden_size = hidden_size,
        num_layers = num_layers,
        dropout = dropout,
        batch_first = True # input and output are in format(batch_size, seq_len, input_size)
        )
    self.reset_parameters()
  
  def reset_parameters(self):
    pass

  def forward(self, input, mask):
    output, hn = self.rnn(input)
    lens = mask.sum(1)
    output_last = utils.select_hiddens_with_index(output, lens-1)
    return output_last, output, hn

class TeneyNet_LSTM_ATT2(nn.Module):
  def __init__(self, nVoc, nAns, opt):
    super(TeneyNet_LSTM_ATT2, self).__init__()
    self.nVoc = nVoc
    self.nAns = nAns
    self.opt  = opt
    self.embed_size = 300
    self.rnn_hidden_size = 300
    self.lookup_table = nn.Embedding(nVoc + 1, self.embed_size)
    self.rnn = LSTM(input_size=self.embed_size, 
        hidden_size=self.rnn_hidden_size,
        num_layers=1,
        dropout=0)
    self.que_gtanh = GatedTanh(self.rnn_hidden_size, 512)
    self.att_gtanh = GatedTanh(self.rnn_hidden_size+2048, 512)
    self.att_linear = nn.Linear(512, 2)
    self.img_gtanh = GatedTanh(2048, 512)
    self.img_gtanh_2 = GatedTanh(4096, 512)
    self.iq_branch1_gtanh = GatedTanh(512, 300)
    self.iq_branch1_linear = nn.Linear(300, nAns)
    self.iq_branch2_gtanh = GatedTanh(512, 2048)
    self.iq_branch2_linear = nn.Linear(2048, nAns)

  def forward(self, in_imgs, in_ques):
    '''
    in_ques: Variable of LongTensor, (batch_size, num_of_max_words)
    in_mask: Variable of LongTensor, (batch_size, num_of_max_words)
    in_imgs: Variable of FloatTensor, (batch_size, channels, height, width)
    '''
    # Text
    q_feature = self.lookup_table(in_ques) # (N, T, 300)
    q_feature, _, _ = self.rnn(q_feature, in_ques>0) # (N, 300)
    q_feature = q_feature.unsqueeze(1)

    # Image
    batch_size, channels, height, width = in_imgs.size()
    num_img_vectors = height * width
    in_imgs = in_imgs.view(batch_size, channels, num_img_vectors).transpose(2, 1).contiguous()
    i_feature = in_imgs / torch.sqrt((in_imgs * in_imgs).sum(2).unsqueeze(2))

    # attention
    q_feature_att = q_feature.repeat(1, num_img_vectors, 1).view(-1, self.rnn_hidden_size)
    i_feature_att = i_feature.view(-1, channels)
    iq_feature_att = torch.cat([q_feature_att, i_feature_att], 1)
    attention = self.att_gtanh(iq_feature_att)
    attention = self.att_linear(attention)
    attention = attention.view(batch_size, num_img_vectors,2)
    attention = F.softmax(attention, dim=1) #(batch_size, num_img_vectors,attention_dim)
    attention_0_feature = (i_feature * attention[:,:,0].unsqueeze(2)).sum(1)
    attention_1_feature = (i_feature * attention[:,:,1].unsqueeze(2)).sum(1)
    # joint embedding
    q_feature = self.que_gtanh(q_feature.squeeze())
    attention_feature = torch.cat([attention_0_feature,attention_1_feature],1)
    attention_feature = self.img_gtanh_2(attention_feature)
    joint_feature = attention_feature * q_feature.squeeze()
    # answer prediction
    pred_branch1 = self.iq_branch1_linear(self.iq_branch1_gtanh(joint_feature))
    pred_branch2 = self.iq_branch2_linear(self.iq_branch2_gtanh(joint_feature))
    prediction = F.softmax(pred_branch1 + pred_branch2, dim=1)
    return prediction

class TeneyNetAC(nn.Module):
  def __init__(self, nVoc, nAns, opt):
    super(TeneyNetAC, self).__init__()
    self.nVoc = nVoc
    self.nAns = nAns
    self.opt  = opt
    self.embed_size = 300
    self.rnn_hidden_size = 300
    self.lookup_table = nn.Embedding(nVoc + 1, self.embed_size)
    self.rnn = LSTM(input_size=self.embed_size, 
        hidden_size=self.rnn_hidden_size,
        num_layers=1,
        dropout=0)
    # answer to vector
    self.embed_ans = nn.Embedding(self.nAns, self.embed_size)
    self.embed_bias_ans = nn.Embedding(self.nAns, 1)
    self.embed_ans2 = nn.Embedding(self.nAns, 2048)
    self.embed_bias_ans2 = nn.Embedding(self.nAns, 1)
    # attention
    self.att_gtanh = GatedTanh(self.rnn_hidden_size+2048, 512)
    self.att_linear = nn.Linear(512, 1)
    # joint fusion 
    self.que_gtanh = GatedTanh(self.rnn_hidden_size, 512)
    self.img_gtanh = GatedTanh(2048, 512)
    # classification
    self.iq_branch1_gtanh = GatedTanh(512, 300)
    self.iq_branch2_gtanh = GatedTanh(512, 2048)

  def forward(self, in_imgs, in_ques, in_acs):
    '''
    in_ques: Variable of LongTensor, (batch_size, num_of_max_words)
    in_mask: Variable of LongTensor, (batch_size, num_of_max_words)
    in_imgs: Variable of FloatTensor, (batch_size, channels, height, width)
    '''
    # Text
    q_feature = self.lookup_table(in_ques) # (N, T, 300)
    q_feature, _, _ = self.rnn(q_feature, in_ques>0) # (N, 300)
    q_feature = q_feature.unsqueeze(1)
    # Image
    batch_size, channels, height, width = in_imgs.size()
    num_img_vectors = height * width
    in_imgs = in_imgs.view(batch_size, channels, num_img_vectors).transpose(2, 1).contiguous()
    i_feature = in_imgs / torch.sqrt((in_imgs * in_imgs).sum(2).unsqueeze(2))
    # Answer Candidates
    num_acs_vectors = in_acs.size(1)
    ac_feature = self.embed_ans(in_acs)
    ac_feature_bias = self.embed_bias_ans(in_acs)
    ac_feature2 = self.embed_ans2(in_acs)
    ac_feature_bias2 = self.embed_bias_ans2(in_acs)
    # attention
    q_feature_att = q_feature.repeat(1, num_img_vectors, 1).view(-1, self.rnn_hidden_size)
    i_feature_att = i_feature.view(-1, channels)
    iq_feature_att = torch.cat([q_feature_att, i_feature_att], 1)
    attention = self.att_gtanh(iq_feature_att)
    attention = self.att_linear(attention)
    attention = attention.view(batch_size, num_img_vectors)
    attention = F.softmax(attention, dim=1) #(batch_size, num_img_vectors)
    attention_feature = (i_feature * attention.unsqueeze(2)).sum(1)
    # joint embedding
    q_feature = self.que_gtanh(q_feature.squeeze())
    attention_feature = self.img_gtanh(attention_feature)
    joint_feature = attention_feature * q_feature.squeeze()
    # answer prediction
    prediction1 = self.iq_branch1_gtanh(joint_feature)
    prediction1 = prediction1.view(prediction1.size(0), prediction1.size(1), 1)
    prediction1 = torch.bmm(ac_feature, prediction1) + ac_feature_bias
    prediction2 = self.iq_branch2_gtanh(joint_feature)
    prediction2 = prediction2.view(prediction2.size(0), prediction2.size(1), 1)
    prediction2 = torch.bmm(ac_feature2, prediction2) + ac_feature_bias2
    prediction = (prediction1 + prediction2).view(batch_size, num_acs_vectors)
    prediction = F.softmax(prediction, dim=1)
    return prediction

  def resume_from_checkpoint(self, path_checkpoint=None):
    if path_checkpoint is None:
      path_checkpoint = self.opt['optim']['resume']
    params_dict = dict(self.named_parameters())
    ckpt = torch.load(path_checkpoint)['model']
    for k, v in params_dict.items():
      if k in ckpt:
        v.data[...] = ckpt[k].cpu()
    params_dict['embed_ans.weight'].data[...] = ckpt['iq_branch1_linear.weight']
    params_dict['embed_bias_ans.weight'].data[...] = ckpt['iq_branch1_linear.bias']
    params_dict['embed_ans2.weight'].data[...] = ckpt['iq_branch2_linear.weight']
    params_dict['embed_bias_ans2.weight'].data[...] = ckpt['iq_branch2_linear.bias']
