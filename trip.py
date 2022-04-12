import torch
import numpy as np
import pandas as pd
import datetime as dt
import random
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt

import torch.nn.functional as F
import os
import glob
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence,pad_packed_sequence
import torch.nn.init as weight_init
import scipy

device='cuda' if torch.cuda.is_available() else 'cpu'

class Encoder(nn.Module):
  def __init__(self, input_dim,z_dim,a_dim=None,n_layers=12,penul=2,hidden_dim=512):
    """
    input_dim: dimension of the input
    z_dim: hidden dimension
    a_dim: action dimension IF we need to consider the action conditioned embeddings
    penul: If we can using a 1d z_dim can use a lowerD penul layer for visualization

    projects into the Z_dim sphere
    """
    super(Encoder,self).__init__()
    
    self.input=nn.Linear(input_dim,hidden_dim)
    self.hiddens=nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.ELU()) for _ in range(n_layers)])
    self.a=a_dim
    self.z_dim=z_dim

    if a_dim:
      self.a=nn.Linear(a_dim,hidden_dim)
      self.a_h=self.Linear(a_dim+hidden_dim,hidden_dim)

    #Penultimate layer
    if penul:
        self.pen_layer=nn.Linear(hidden_dim,penul)
    # self.pen_bn=nn.BatchNorm1d(penul)
    self.output=nn.Linear(hidden_dim,z_dim)
    self.init_weights()

  def  init_weights(self):
      for w in self.parameters(): # initialize the gate weights with orthogonal
          if w.dim()>1:
                weight_init.orthogonal_(w)

  def forward(self,obs,a=None,penul=False):
    out=F.elu(self.input(obs)) #B*H
    for layer in self.hiddens:
      out=layer(out)

    if self.a:
      a_=F.elu(self.a(a))  #B*H
      out=F.elu(self.a_h(torch.cat([out,a_],dim=-1).to(device))) #B*H
    
    # out=F.elu(self.pen_layer(out)) #B*P
    out=self.output(out) #B*Z
    
    # out=torch.tanh(out)/self.z_dim #Now norm will always be <=1

class trip_dataset(Dataset):
  def __init__(self,df_,k=41,t=12,cosine_em=False):

    """
    Class for the Dataset of generating triplets and additional info

    df_: pd.DataFrame instance
    t: final t points of a patient will be taken for positive/negative
    k: the Dimensions of the state: Should be the first 41 columns
    
    """
    self.df=df_
    self.k=k
    ## Take unique patients as the dataset length
    """ 
    Need to make sure death and relsease indicators are in the dataframe
    """
    assert 'Death' in df_.columns
    assert 'Release' in df_.columns

    self.pats=df_.Pat.unique()
    self.pat_list=list(df_.Pat.unique())
    self.dead_pats=list(df_[df_.Death==1].Pat)
    self.surv_pats=list(df_[df_.Release==1].Pat)
    self.t=t
    
    #If we will be using a cosine embedding as a loss (then need +1/-1 as y)
    self.cosine_em=cosine_em
   
  def __len__(self):
    return self.pats.shape[0]

  def __getitem__(self,idx):
    #get a patient indexed by the index
    pat=self.pats[idx]
    pat_df=self.df[self.df.Pat==pat]

    
    is_dead=pat_df.Death.iloc[-1]==1

    #if the patient is dead let's keep a track of their worst organ system
    if is_dead:
      worst_organ=pat_df.worst_organ.iloc[-1]
      
  

    #Generate an anchor this will always be a terminal state
    anchor=torch.Tensor(pat_df.iloc[-1,:self.k].values).to(device)  #T*O
  
    """
    Now let's get the positive and negative states
    """   
    # generate a dead_patient
    pat1=np.random.choice(list(set(self.dead_pats)-set([idx])))

    if is_dead:
      pat1=np.random.choice(self.df[(self.df.Pat!=pat)&(self.df.worst_organ!=worst_organ)].Pat)
    
    
    pat1_df=self.df[self.df.Pat==pat1]
    idx1=np.random.choice(np.arange(min(self.t,pat1_df.shape[0])))
    dead_state=torch.Tensor(pat1_df.iloc[-idx1,:self.k].values).to(device)

    #generate a survivor
    pat2=np.random.choice(list(set(self.surv_pats)-set([idx])))
    pat2_df=self.df[self.df.Pat==pat2]
    idx2=np.random.choice(np.arange(min(self.t,pat2_df.shape[0])))
    surv_state=torch.Tensor(pat2_df.iloc[-idx2,:self.k].values).to(device)
    
    if is_dead:
       
       positive=dead_state
       
       if self.cosine_em:
         y=-1 if y==0 else 1
       else: 
         y=(pat1_df.worst_organ.iloc[-idx1]==worst_organ).astype('int')

       negative=surv_state
       return anchor.to(device),int(is_dead),positive.to(device),negative.to(device),y
    #In this case let's send y=1 and mask this out?
     
    positive=surv_state
    negative=dead_state
    y=1
    return anchor.to(device),int(is_dead),positive.to(device),negative.to(device),y    


def get_trip_loss(model,anchor,is_dead,positive,negative,y,triplet_loss,beta,lam1,lam2,lam3,lam4,alpha=3,cos_em_loss=False):
    """
    beta : float \in (0,1)--> balances triplet loss and absorption loss
    lam: float : Uses the penalize norms of postive/negative if they are non survivors
    lam1 : float : penalizes if the norm is greater than 1
    lam2: <=1 : In absorption loss for survivors is lam2*absorption loss for non survivors
    alpha: exp(-alpha*x) is used along with lam for positive/negative non-survivors
    triplet_loss: triplet loss instance
    r: The ratio to penalize the non terminal survivors norm^2
    """
    ### get lower dimensional representations
    anchor_out=model(anchor)
    pos_out=model(positive)
    neg_out=model(negative)
    
    ##These are norm^2 
    anchor_norm=F.mse_loss(anchor_out,torch.zeros_like(anchor_out).to(device),reduction='none').sum(dim=-1)  #B*Z-->B #B
    pos_norm=F.mse_loss(pos_out,torch.zeros_like(pos_out).to(device),reduction='none').sum(dim=-1)  #B*Z-->B #B
    neg_norm=F.mse_loss(neg_out,torch.zeros_like(neg_out).to(device),reduction='none').sum(dim=-1)    #B*Z-->B #B
    
    """
    Compute two losses for the anchor's norm will mask out at the end depending on survival
    """
    #Find the norm^2 and ||norm^2-1||^2
    loss1=F.mse_loss(F.mse_loss(anchor_out,torch.zeros_like(anchor_out).to(device),reduction='none'
    ).sum(dim=-1),torch.ones(anchor_out.shape[0]).to(device),reduction='none') #B
    # loss2=F.mse_loss(F.mse_loss(anchor_out,torch.zeros_like(anchor_out).to(device),reduction='none').sum(dim=-1),torch.ones(anchor_out.shape[0]).to(device)*0.25,reduction='none') #B
    loss2=F.mse_loss(anchor_out,torch.zeros_like(anchor_out).to(device),reduction='none').sum(dim=-1)  #B*Z-->B
    
    loss=loss1*(is_dead).to(device)+loss2*(1-is_dead).to(device)*lam1
    

    """
    If anchor was dead make sure the positive elemnt have norm >>0 and 
    similarly if anchor is not dead we need the negative to have norm>>0
    """
    # w_decay1=lam*((is_dead)*(1/(pos_norm+1e-1))+(1-is_dead)*1/(neg_norm+1e-1))
    
    int_loss1=lam2*((pos_norm>1.0).float().to(device)*(pos_norm)+(neg_norm>1.0).float().to(device)*(neg_norm))
    
    int_loss2=lam3*((is_dead).to(device)*torch.exp(-alpha*pos_norm).to(device)+(1-is_dead).to(device)*torch.exp(-alpha*neg_norm).to(device))

    #Now also let's penalize the norm of survivors but not as much
   
    int_loss2+=lam4*((is_dead).to(device)*neg_norm+(1-is_dead).to(device)*pos_norm)
    
    #To make sure the projected point is in the unit sphere
    
    int_loss=int_loss1+int_loss2
   
    ### Find Contrastive loss

    if cos_em_loss:
        cont_loss=(1-is_dead).to(device)*triplet_loss(anchor_out.detach(), pos_out, neg_out)+(is_dead).to(device)*cos_em_loss(anchor_out,
    pos_out,y.to(device))
    else:
        #Inner Product
        cos=(anchor_out*pos_out).sum(dim=-1)
        cont_loss=(1-is_dead).to(device)*triplet_loss(anchor_out.detach(), 
        pos_out, neg_out)+(is_dead).to(device)*1*(cos(anchor_out,pos_out))*(1-y).to(device)
    
    # trip_loss=triplet_loss(anchor_out.detach(), pos_out, neg_out)*(pos_hzd-neg_hzd)
    # return (beta*loss+(1-beta)*trip_loss).mean()
    return (beta*loss+(1-beta)*cont_loss+int_loss).mean()

def train_epoch(model,loader,optimizer,beta,triplet_loss,cos_em_loss=None,i=0,lam=0.0,lam1=0.0,lam2=0.0,alpha=3,loss_func=get_trip_loss):
  model.train()
  total_loss=0
  for batch,(anchor,is_dead,positive,negative,y) in enumerate(loader):
    optimizer.zero_grad()
    # loss=get_trip_loss2(model,anchor,is_dead,positive,negative,y,beta,lam,lam1,lam2,alpha)
    # loss=get_trip_loss_old(model,anchor,is_dead,positive,negative,y,beta,lam,lam1,lam2,alpha)

    loss=loss_func(model,anchor,is_dead,positive,negative,y,beta,lam,lam1,lam2,alpha,triplet_loss,cos_em_loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
    optimizer.step()
    total_loss+=loss.item()
    
    
    if batch%30==0:
        print('Epoch, Batch Number : ',i,', ',batch, 'Batch_loss :',loss.item(),' Total Loss :', total_loss/(batch+1))

def validate(model,loader,optimizer,beta,triplet_loss,cos_em_loss=None,i=0,lam=0.0,lam1=0.0,lam2=0.0,alpha=3,loss_func=get_trip_loss):
  model.eval()
  total_loss=0
  for batch,(anchor,is_dead,positive,negative,y) in enumerate(loader):
    with torch.no_grad():
      loss=loss_func(model,anchor,is_dead,positive,negative,y,beta,lam,lam1,lam2,alpha,triplet_loss,cos_em_loss)
      total_loss+=loss.item()

  print('Validating Total Loss :', total_loss/(batch+1))
  return total_loss/(batch+1)
