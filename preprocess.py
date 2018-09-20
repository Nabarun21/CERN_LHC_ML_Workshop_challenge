
# coding: utf-8

# # LHC Machine Learing workshop challenge
# https://gitlab.cern.ch/IML-WG/IML_challenge_2018/wikis/home
# 
# Task: Regress the soft-drop mass of jets with high transverse momentum

# In[1]:


#get the data
import numpy as np


my_array=np.load('../data/qcd.npy',encoding='bytes')

my_rec_array=my_array.view(np.recarray)



num_jets=my_rec_array.shape[0]
fields=my_array.dtype.names
for field in fields:
    print(field)
    x=getattr(my_rec_array,field)
    x=np.reshape(x,[1,num_jets])
    try:
        my_train_val_array=np.concatenate((my_train_val_array,x))
    except:
        print(field)
        my_train_val_array=x
        


# In[61]:


my_train_val_array=my_train_val_array.T



from sklearn.model_selection import train_test_split

train_array, val_array = train_test_split(my_train_val_array, test_size=0.30,random_state=8521)



reco_features_train=train_array[:,0:10] #recofeatures+target
#features1[:3]
const_features_train=train_array[:,9:] #constituents
reco_features_val=val_array[:,0:10] #recofeatures+target
#features1[:3]
const_features_val=val_array[:,9:] #constituents


# In[ ]:



def prep_constituents(event):
    n_const=int(event[0])
    if n_const<=130:
        event[1]=event[1]/float(100.)
        event[-1]=event[-1]/float(100.)
        event[-2]=event[-2]/float(100.)
        a=np.zeros([8,130])-5959
        s=event[1:]
        t=np.stack(s)
        t=t[:,np.argsort(t[0,:],kind='mergesort')[::-1]]
        a[:,:n_const]=t
        a=np.reshape(a,[1,8,130])
    else:
        event[1]=event[1]/float(100.)
        event[-1]=event[-1]/float(100.)
        event[-2]=event[-2]/float(100.)
        s=event[1:]
        t=np.stack(s)
        t=t[:,np.argsort(t[0,:],kind='mergesort')[::-1]]
        a=t[:,:130]
        a=np.reshape(a,[1,8,130])
       # out_array=np.append(out_array,a,axis=0)
    return a


# In[102]:


import time
out_array=[]
a=time.time()
for event in const_features_train:
    prepped_event=prep_constituents(event)
    #out_array=np.append(out_array,prepped_event,axis=0)
    out_array.append(prepped_event)
#out_array=np.delete(out_array,0,0)
#print(out_array[0:2])
out_array=np.array(out_array)
print(out_array.shape)
b=time.time()
print(b-a)
np.savez('train_consts_reg',out_array)

out_array2=[]
a=time.time()
for event in const_features_val:
    prepped_event=prep_constituents(event)
    #out_array=np.append(out_array,prepped_event,axis=0)
    out_array2.append(prepped_event)
#out_array=np.delete(out_array,0,0)
#print(out_array[0:2])
out_array2=np.array(out_array2)
print(out_array2.shape)
b=time.time()
print(b-a)
np.savez('val_consts_reg',out_array2)






