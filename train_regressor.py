
# coding: utf-8

# # LHC Machine Learing workshop challenge
# https://gitlab.cern.ch/IML-WG/IML_challenge_2018/wikis/home
# 
# Task: Regress the soft-drop mass of jets with high transverse momentum

# In[1]:


#get the data
import numpy as np
import helpers
helpers.mask_busy_gpus()

my_array=np.load('../data/qcd.npy',encoding='bytes')

my_rec_array=my_array.view(np.recarray)


# In[3]:


fields=my_array.dtype.names


# In[5]:


#del my_train_val_array
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
        


# In[6]:


my_train_val_array=my_train_val_array.T
print(my_train_val_array.shape)


# In[7]:


from sklearn.model_selection import train_test_split

train_array, val_array = train_test_split(my_train_val_array, test_size=0.30,random_state=8521)
print(train_array.shape,val_array.shape)


# In[8]:


reco_features_train=train_array[:,0:10] #recofeatures+target
#features1[:3]
const_features_train=train_array[:,9:] #constituents
reco_features_val=val_array[:,0:10] #recofeatures+target
#features1[:3]
const_features_val=val_array[:,9:] #constituents


# In[72]:


from keras import regularizers,optimizers
from keras.layers import Reshape,Flatten,Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,ELU,LeakyReLU,Dropout,BatchNormalization,concatenate
from keras.models import Model,load_model
from keras import backend as K
from keras import callbacks


# In[49]:



train_consts=np.load('train_consts_reg.npz')['arr_0']
train_consts.shape


# In[51]:


val_consts=np.load('val_consts_reg.npz')['arr_0']
val_consts.shape


# In[62]:


encoder=load_model('my_encoder_997105.122481_189.h5')
train_features=encoder.predict(train_consts)
val_features=encoder.predict(val_consts)


# In[67]:


print(train_features.shape)
print(val_features.shape)


# In[69]:


train_target=reco_features_train[:,0]
val_target=reco_features_val[:,0]
#train_target=np.reshape(train_target,[tr])
print(train_target.shape)

reco_features_train=reco_features_train[:,1:]
reco_features_val=reco_features_val[:,1:]


# In[60]:


print(train_features.shape,train_target.shape,reco_features_train.shape)
print(val_features.shape,val_target.shape,reco_features_val.shape)


# In[77]:


train_final=np.hstack((reco_features_train,train_features))
val_final=np.hstack((reco_features_val,val_features))
#train_final_trial=np.reshape(train_final_trial,[train_final_trial.shape[0],1,train_final_trial.shape[1]])
print(train_final.shape)
print(val_final.shape)


# In[79]:

#regressor model
input_vec=Input(shape=(train_final.shape[1],))

x=BatchNormalization(axis=1)(input_vec)

x=Dense(500,name='densea',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.02))(x)
x=BatchNormalization(axis=1)(x)
x=LeakyReLU(alpha=0.3)(x)
x=Dropout(0.3)(x)


x=Dense(300,name='denseb',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.01))(x)
x=BatchNormalization(axis=1)(x)
x=LeakyReLU(alpha=0.3)(x)
x=Dropout(0.3)(x)

x=Dense(200,name='densec',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.01))(x)
x=BatchNormalization(axis=1)(x)
x=LeakyReLU(alpha=0.3)(x)
x=Dropout(0.2)(x)

x=Dense(150,name='densed',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.01))(x)
x=Dropout(0.2)(x)
x=LeakyReLU(alpha=0.3)(x)
x=BatchNormalization(axis=1)(x)

x=Dense(100,name='densee',kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.01))(x)
x=BatchNormalization(axis=1)(x)
x=LeakyReLU(alpha=0.3)(x)
x=Dropout(0.1)(x)

y_pred=Dense(1,name='final',activation='relu')(x)

regressor=Model(input_vec,y_pred)
regressor.compile(optimizer='adadelta', loss='MSE')
regressor.summary()


# In[80]:


for epoch in range(100):
    my_hist=regressor.fit(x=train_final, y=train_target, epochs=1,batch_size=50,
                validation_data=[val_final,val_target],
                verbose=1,
                shuffle=True
                )
    if epoch>0 and epoch%3==0:
        val_loss=my_hist.history['val_loss'][0]
        save_name='my_regressor_'+str(val_loss)+'_'+str(epoch)+'.h5'
        regressor.save(save_name)
