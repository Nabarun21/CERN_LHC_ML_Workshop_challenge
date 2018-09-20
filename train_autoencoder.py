
# coding: utf-8

# In[1]:


from keras import regularizers,optimizers
from keras.layers import Reshape,Flatten,Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,ELU,Dropout,BatchNormalization,concatenate
from keras.models import Model
from keras import backend as K
from keras import callbacks
import helpers

helpers.mask_busy_gpus()

# In[3]:


import numpy as np
train_consts=np.load('train_consts.npz')['arr_0']
print(train_consts.shape)

val_consts=np.load('val_consts.npz')['arr_0']
print(val_consts.shape)

def apply_gaussian_noise(X,sigma=0.1):
    """
    adds noise from standard normal distribution with standard deviation sigma
    :param X: image tensor of shape [batch,height,width,3]
    Returns X + noise.
    """
    noise = np.random.standard_normal(X.shape)*sigma### YOUR CODE HERE ###
    return X + noise

# In[4]:


#load or build the model

input_img=Input(shape=(1,8,130))

#encoder



x1=Conv2D(16,(8,1),padding='valid',data_format='channels_first',name='conva',kernel_initializer='he_uniform')(input_img)
x1=ELU(alpha=0.3)(x1)
print(x1.shape)
x1=Flatten()(x1)
print(x1.shape)
x1=Dense(100,name='densea',kernel_initializer='he_uniform')(x1)
x1=ELU(alpha=0.3)(x1)
x1=Dropout(0.3)(x1)
print(x1.shape)

x2=Conv2D(24,(1,130),padding='valid',data_format='channels_first',name='convb',kernel_initializer='he_uniform')(input_img)
x2=ELU()(x2)
print(x2.shape)
x2=Flatten()(x2)
print(x2.shape)
x2=Dense(300,name='denseb',kernel_initializer='he_uniform')(x2)
x2=ELU(alpha=0.3)(x2)
x2=Dropout(0.4)(x2)
print(x2.shape)

x=concatenate([x1,x2],axis=1)

print(x.shape)
encoded=Dense(100,name='densec',kernel_initializer='he_uniform')(x)
x=ELU(alpha=0.3)(encoded)
x=Dropout(0.1)(x)
print(x.shape)
x=Dense(400,name='densef',kernel_initializer='he_uniform')(x)
x=ELU(alpha=0.4)(x)
x=Dropout(0.4)(x)
print(x.shape)
x=Dense(8*130,name='denseg',kernel_initializer='he_uniform',activation='linear')(x)
print(x.shape)
decoded=Reshape((1,8,130))(x)

encoder=Model(input_img,encoded)
autoencoder=Model(input_img, decoded)

encoder.compile(optimizer='adadelta', loss='MSE')
autoencoder.compile(optimizer='adadelta', loss='MSE')
autoencoder.summary()


# In[5]:

for epoch in range(200):
    train_consts_noise=apply_gaussian_noise(train_consts,0.2)
    val_consts_noise=apply_gaussian_noise(val_consts,0.2)

    my_hist=autoencoder.fit(x=train_consts_noise, y=train_consts, epochs=1,batch_size=100,
                validation_data=[val_consts_noise,val_consts],shuffle=True,
                verbose=1
                )
    if epoch>10 and epoch%3==0:
        val_loss=my_hist.history['val_loss'][0]
        save_name='my_encoder_'+str(val_loss)+'_'+str(epoch)+'.h5'
        encoder.save(save_name)
