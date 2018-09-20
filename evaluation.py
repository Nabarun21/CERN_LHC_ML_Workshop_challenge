
# coding: utf-8

# In[2]:
import helpers


from keras import regularizers,optimizers
from keras.layers import Reshape,Flatten,Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,ELU,LeakyReLU,Dropout,BatchNormalization,concatenate
from keras.models import Model,load_model
from keras import backend as K
from keras import callbacks

helpers.mask_busy_gpus()
# In[4]:


#get the data
import numpy as np

my_array=np.load('../data/qcd.npy',encoding='bytes')

my_rec_array=my_array.view(np.recarray)

#get the data
import numpy as np

my_array_test=np.load('../data/qcd_test.npy',encoding='bytes')

my_rec_array_test=my_array_test.view(np.recarray)


# In[68]:


fields=my_array.dtype.names
num_jets=my_rec_array.shape[0]
for field in fields:
    print(field)
    x=getattr(my_rec_array,field)
    x=np.reshape(x,[1,num_jets])
    try:
        my_train_val_array=np.concatenate((my_train_val_array,x))
    except:
        print(field)
        my_train_val_array=x
my_train_val_array=my_train_val_array.T

from sklearn.model_selection import train_test_split

train_array, val_array = train_test_split(my_train_val_array, test_size=0.30,random_state=8521)
train_array.shape,val_array.shape

fields=my_array_test.dtype.names
num_jets=my_rec_array_test.shape[0]
for field in fields:
    print(field)
    x=getattr(my_rec_array_test,field)
    x=np.reshape(x,[1,num_jets])
    try:
        my_test_val_array=np.concatenate((my_test_val_array,x))
    except:
        print(field)
        my_test_val_array=x

test_array=my_test_val_array.T



reco_features_train=train_array[:,0:10] #recofeatures+target
#features1[:3]
#const_features_train=train_array[:,9:] #constituents
reco_features_val=val_array[:,0:10] #recofeatures+target
reco_features_test=test_array[:,0:9] #recofeatures+target
#features1[:3]
#const_features_val=val_array[:,9:]


# In[6]:



train_consts=np.load('train_consts_reg.npz')['arr_0']
train_consts.shape
val_consts=np.load('val_consts_reg.npz')['arr_0']
val_consts.shape
test_consts=np.load('test_consts_reg.npz')['arr_0']
val_consts.shape

#!!!!!!change it



# In[14]:

encoder=load_model('my_encoder_997105.122481_189.h5')
train_features=encoder.predict(train_consts)
val_features=encoder.predict(val_consts)
test_features=encoder.predict(test_consts)
train_features.shape


# In[70]:


train_target=reco_features_train[:,0]
val_target=reco_features_val[:,0]
reco_features_train=reco_features_train[:,1:]
reco_features_test=reco_features_test[:,:]
reco_features_val=reco_features_val[:,1:]#!!!!!!change it
reco_features_train.shape


# In[71]:


train_final=np.hstack((reco_features_train,train_features))
val_final=np.hstack((reco_features_val,val_features))
test_final=np.hstack((reco_features_test,test_features))
train_final.shape


# In[114]:


trained_model=load_model('my_regressor_7544.79952436_9.h5')
trained_model.summary()

# In[115]:


train_pred=trained_model.predict(train_final)

train_pred=np.reshape(train_pred,(train_pred.shape[0]))

test_pred=trained_model.predict(test_final)

test_pred=np.reshape(test_pred,(test_pred.shape[0]))

val_pred=trained_model.predict(val_final)

val_pred=np.reshape(val_pred,(val_pred.shape[0]))







def evaluate_loss(predictions, truth):  
    truth=truth+1e-15
    predictions=predictions+1e-15
    ratio = predictions/truth
    a = np.percentile(ratio, 84, interpolation='nearest')  
    b = np.percentile(ratio, 16, interpolation='nearest')  
    c = np.percentile(ratio, 50, interpolation='nearest')  
    loss = (a-b)/(2.*c)  
    return loss
from sklearn.metrics import mean_squared_error


# In[113]:


print('train metric: ',evaluate_loss(train_pred,train_target))
print('val metric: ',evaluate_loss(val_pred,val_target))

val_reco_sd=reco_features_val[:,7]
val_reco=reco_features_val[:,3]
print('val metric recosd: ',evaluate_loss(val_reco_sd,val_target))
print('val metric: reco',evaluate_loss(val_reco,val_target))




# In[97]:


print('train loss: ',mean_squared_error(train_pred,train_target))
print('val loss: ',mean_squared_error(val_pred,val_target))

print(test_pred)
np.save('regression_target_test',test_pred)
