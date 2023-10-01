import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D as Conv2D
from tensorflow.keras.layers import Conv1D as Conv1D
from tensorflow.keras.layers import MaxPooling2D as Maxpool
from tensorflow.keras.layers import MaxPooling1D as Maxpool1D
from tensorflow.keras.layers import UpSampling2D as Upsample
from tensorflow.keras.layers import UpSampling1D as Upsample1D
from tensorflow.keras.layers import Reshape as Reshape
from tensorflow.keras.layers import Activation as Activation
from tensorflow.keras.activations import exponential as Exp
from tensorflow.keras.layers import Flatten as Flatten
from tensorflow.keras.layers import Dense as Dense
from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.layers import BatchNormalization as BatchNormalization
from tensorflow.keras.layers import ZeroPadding2D as ZP
from tensorflow.keras.layers import Lambda as Lambda




def Fully_connected_matrix(HEIGHT_in,WIDTH_in,NUM_frames):
    
  initializer = tf.random_uniform_initializer(minval=0., maxval=1.)
  # initializer = tf.random_normal_initializer(0.,0.02)
  T=tf.Variable(initial_value=initializer(shape=[NUM_frames,HEIGHT_in,WIDTH_in,1], dtype=tf.float32), trainable=True,name='real')
  
  return T


def forward_model_base(output_size ,HEIGHT_in = 50,WIDTH_in = 50,mean_firing_rate = 0.05):
  # inputs= tf.keras.layers.Input(shape=[1,HEIGHT_in,WIDTH_in])

  

  blocks_1 = [
            Conv2D(filters=4, kernel_size=21,padding='valid',data_format='channels_first', use_bias=False,kernel_regularizer=l2(1e-5)),
            
            Reshape(( 4,(HEIGHT_in-21+1)*(WIDTH_in-21+1))) ]

  blocks_2=[
            
            Reshape(( 4,(HEIGHT_in-21+1),(WIDTH_in-21+1))),
            # tf.keras.layers.GaussianNoise(sigma),
            Activation('relu'),
            Conv2D(filters=4, kernel_size=15,padding='valid',use_bias=True, data_format='channels_first', kernel_regularizer=l2(1e-5)),
            # tf.keras.layers.GaussianNoise(sigma),
            Activation('relu'),
            Flatten(),
            Dense(output_size,kernel_initializer='normal',use_bias=True,bias_initializer=tf.keras.initializers.Constant(value=(mean_firing_rate)),kernel_regularizer=l2(1e-5),activity_regularizer=None),
            Activation('softplus'),
  ]
  
    
  blocks_1.append(Lambda( lambda x: tf.transpose(x,perm=[2,1,0])))
 

  blocks_1.append(Conv1D(filters=4, kernel_size=40,padding='same',data_format='channels_first', groups=4, use_bias=True,kernel_regularizer=l2(1e-5)))
  

  blocks_1.append(Lambda( lambda x: tf.transpose(x,perm=[2,1,0])))
  
  blocks_2.append(Lambda( lambda x: x[tf.newaxis,...]))
  

  return tf.keras.Sequential(blocks_1+blocks_2)



def forward_model_noisy(output_size ,HEIGHT_in = 50,WIDTH_in = 50,mean_firing_rate = 0.05):
  # inputs= tf.keras.layers.Input(shape=[1,HEIGHT_in,WIDTH_in])

  

  blocks_1 = [
            Conv2D(filters=4, kernel_size=21,padding='valid',data_format='channels_first', use_bias=False,kernel_regularizer=l2(1e-5)),
            GaussianNoise(0.1),
            Reshape(( 4,(HEIGHT_in-21+1)*(WIDTH_in-21+1))) ]

  blocks_2=[
            
            Reshape(( 4,(HEIGHT_in-21+1),(WIDTH_in-21+1))),

            Activation('relu'),
            Conv2D(filters=4, kernel_size=15,padding='valid',use_bias=True, data_format='channels_first', kernel_regularizer=l2(1e-5)),
            GaussianNoise(0.1),
            Activation('relu'),
            Flatten(),
            Dense(output_size,kernel_initializer='normal',use_bias=True,bias_initializer=tf.keras.initializers.Constant(value=(mean_firing_rate)),kernel_regularizer=l2(1e-5),activity_regularizer=l1(1e-5)),
            Activation('softplus'),
  ]
  
    
  blocks_1.append(Lambda( lambda x: tf.transpose(x,perm=[2,1,0])))
 

  blocks_1.append(Conv1D(filters=4, kernel_size=40,padding='same',data_format='channels_first', groups=4, use_bias=True,kernel_regularizer=l2(1e-5)))
  

  blocks_1.append(Lambda( lambda x: tf.transpose(x,perm=[2,1,0])))
  
  blocks_2.append(Lambda( lambda x: x[tf.newaxis,...]))
  

  return tf.keras.Sequential(blocks_1+blocks_2)
  

def forward_model_noisy_big(output_size ,HEIGHT_in = 50,WIDTH_in = 50,mean_firing_rate = 0.05):
  # inputs= tf.keras.layers.Input(shape=[1,HEIGHT_in,WIDTH_in])

  

  initializer_mean = tf.keras.initializers.RandomNormal(mean=0., stddev=0.001)
  # initializer_log_var = tf.keras.initializers.RandomNormal(mean=-2.5, stddev=0.001)
  blocks_1 = [Reshape((1,HEIGHT_in,WIDTH_in)),            
              Conv2D(filters=64, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
              Activation('relu'),
              Maxpool((2,2),data_format='channels_first'),
              Conv2D(filters=32, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
              Activation('relu'),
              Maxpool((2,2),data_format='channels_first'),
              Conv2D(filters=16, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
              Activation('relu'),
              Flatten(),
              Dense(2*15,kernel_initializer=initializer_mean,use_bias=False,kernel_regularizer=None,activity_regularizer=None)
               ]
              
  blocks_2 =[

            Dense(12*12*16,kernel_initializer='normal',use_bias=True,kernel_regularizer=None,activity_regularizer=None),
            Reshape((16,12,12)),
            Activation('relu'),
            Conv2D(filters=32, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
            Activation('relu'),
            Upsample((2,2),data_format='channels_first'),            
            ZP(((1,0),(1,0)),data_format='channels_first'),          
            Conv2D(filters=64, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
            Activation('relu'),
            Upsample((2,2),data_format='channels_first'),
            Conv2D(filters=1, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
            Activation('sigmoid'),
  ]


  blocks_3 = [
            Conv2D(filters=4, kernel_size=21,padding='valid',data_format='channels_first', use_bias=False,kernel_regularizer=l2(1e-5)),
            GaussianNoise(0.1),
            Reshape(( 4,(HEIGHT_in-21+1)*(WIDTH_in-21+1))) ]

  blocks_4=[
            
            Reshape(( 4,(HEIGHT_in-21+1),(WIDTH_in-21+1))),

            Activation('relu'),
            Conv2D(filters=4, kernel_size=15,padding='valid',use_bias=True, data_format='channels_first', kernel_regularizer=l2(1e-5)),
            GaussianNoise(0.1),
            Activation('relu'),
            Flatten(),
            Dense(output_size,kernel_initializer='normal',use_bias=True,bias_initializer=tf.keras.initializers.Constant(value=(mean_firing_rate)),kernel_regularizer=l2(1e-5),activity_regularizer=None),
            Activation('softplus'),
  ]
  
    
  blocks_3.append(Lambda( lambda x: tf.transpose(x,perm=[2,1,0])))
 

  blocks_3.append(Conv1D(filters=4, kernel_size=40,padding='same',data_format='channels_first', groups=4, use_bias=True,kernel_regularizer=l2(1e-5)))
  

  blocks_3.append(Lambda( lambda x: tf.transpose(x,perm=[2,1,0])))
  

  return tf.keras.Sequential(blocks_1+blocks_2+blocks_3+blocks_4)



def encoder_disjoint_cnn(latent_dims, multiplier=2, HEIGHT_in = 50,WIDTH_in = 50):
    
  # inputs= tf.keras.layers.Input(shape=[1,HEIGHT_in,WIDTH_in])

  initializer_mean = tf.keras.initializers.RandomNormal(mean=0., stddev=0.001)
  # initializer_log_var = tf.keras.initializers.RandomNormal(mean=-2.5, stddev=0.001)
  blocks_1 = [Reshape((1,HEIGHT_in,WIDTH_in)),            
              Conv2D(filters=64, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
              Activation('relu'),
              Maxpool((2,2),data_format='channels_first'),
              Conv2D(filters=32, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
              Activation('relu'),
              Maxpool((2,2),data_format='channels_first'),
              Conv2D(filters=16, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
              Activation('relu'),
              Flatten(),
              Dense(multiplier*latent_dims,kernel_initializer=initializer_mean,use_bias=False,kernel_regularizer=None,activity_regularizer=None),
              
            ]
  blocks_1.append(Lambda( lambda x: x[tf.newaxis,...]))

  return tf.keras.Sequential(blocks_1)





def decoder_disjoint_cnn():

  
  blocks_1=[(Lambda( lambda x: tf.squeeze(x))),
            Dense(12*12*16,kernel_initializer='normal',use_bias=True,kernel_regularizer=None,activity_regularizer=None),
            Reshape((16,12,12)),
            Activation('relu'),
            Conv2D(filters=32, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
            Activation('relu'),
            Upsample((2,2),data_format='channels_first'),            
            ZP(((1,0),(1,0)),data_format='channels_first'),          
            Conv2D(filters=64, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
            Activation('relu'),
            Upsample((2,2),data_format='channels_first'),
            Conv2D(filters=1, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
            Activation('sigmoid'),
  ]

  return tf.keras.Sequential(blocks_1)



def encoder_joint_cnn_base(latent_dims,multiplier=2, HEIGHT_in = 50,WIDTH_in = 50):
    
  # inputs= tf.keras.layers.Input(shape=[1,HEIGHT_in,WIDTH_in])

  initializer_mean = tf.keras.initializers.RandomNormal(mean=0., stddev=0.001)
  # initializer_log_var = tf.keras.initializers.RandomNormal(mean=-2.5, stddev=0.001)
  
  blocks_1 = [BatchNormalization(axis=2),
              Reshape((1,HEIGHT_in,WIDTH_in)),            
              Conv2D(filters=64, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
              Activation('relu'),
              Maxpool((2,2),data_format='channels_first'),
              Flatten(),
              BatchNormalization(axis=1),
              Reshape((64,25,25)),            
              Conv2D(filters=32, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
              Activation('relu'),
              Maxpool((2,2),data_format='channels_first'),
              Flatten(),
              BatchNormalization(axis=1),
              Reshape((32,12,12)),            
              Conv2D(filters=16, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
              Activation('relu'),
              Flatten(),
              BatchNormalization(axis=1),
              Reshape((16,12*12)),  
              Lambda( lambda x: tf.transpose(x,perm=[2,1,0])),
              Conv1D(filters=16, kernel_size=40,padding='same',data_format='channels_first', groups=16, use_bias=True,kernel_regularizer=l2(1e-5)),
              Activation('relu'),
              Maxpool1D(2,data_format='channels_first'),
              Lambda( lambda x: tf.transpose(x,perm=[2,1,0])),
              Flatten(),
              BatchNormalization(axis=1),
              Reshape((16,12,12)),  


              Flatten(),
              Dense(multiplier*latent_dims,kernel_initializer=initializer_mean,use_bias=False,kernel_regularizer=None,activity_regularizer=None),
              
            ]
  blocks_1.append(Lambda( lambda x: x[tf.newaxis,...]))

  return tf.keras.Sequential(blocks_1)


def decoder_joint_cnn_base():

  
  blocks_1=[
            Lambda( lambda x: tf.squeeze(x)),
            Dense(12*12*16,kernel_initializer='normal',use_bias=True,kernel_regularizer=None,activity_regularizer=None),
            Activation('relu'),
            BatchNormalization(axis=1),
            Reshape((16,12*12)),  
            Lambda( lambda x: tf.transpose(x,perm=[2,1,0])),
            Conv1D(filters=16, kernel_size=40,padding='same',data_format='channels_first', groups=16, use_bias=True,kernel_regularizer=l2(1e-5)),
            Activation('relu'),
            Lambda( lambda x: tf.transpose(x,perm=[0,2,1])),
            Upsample1D(size=2),
            Lambda( lambda x: tf.transpose(x,perm=[1,2,0])),
            Flatten(),
            BatchNormalization(axis=1),
            Reshape((16,12,12)),
            Conv2D(filters=32, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
            Activation('relu'),
            Flatten(),
            BatchNormalization(axis=1),
            Reshape((32,12,12)),  
            Upsample((2,2),data_format='channels_first'),            
            ZP(((1,0),(1,0)),data_format='channels_first'),          
            Conv2D(filters=64, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
            Activation('relu'),
            Flatten(),
            BatchNormalization(axis=1),
            Reshape((64,25,25)), 
            Upsample((2,2),data_format='channels_first'),
            Conv2D(filters=1, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
            Activation('sigmoid'),
            Flatten(),
            BatchNormalization(axis=1),
            Reshape((1,50,50)), 

  ]

  return tf.keras.Sequential(blocks_1)


def actor_cnn(latent_dims, multiplier=2, HEIGHT_in = 50,WIDTH_in = 50):
    
  # inputs= tf.keras.layers.Input(shape=[1,HEIGHT_in,WIDTH_in])

  initializer_mean = tf.keras.initializers.RandomNormal(mean=0., stddev=0.001)
  # initializer_log_var = tf.keras.initializers.RandomNormal(mean=-2.5, stddev=0.001)
  blocks_1 = [Reshape((1,HEIGHT_in,WIDTH_in)),            
              Conv2D(filters=64, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
              Activation('relu'),
              Maxpool((2,2),data_format='channels_first'),
              Conv2D(filters=32, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
              Activation('relu'),
              Maxpool((2,2),data_format='channels_first'),
              Conv2D(filters=16, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
              Activation('relu'),
              Flatten(),
              Dense(multiplier*latent_dims,kernel_initializer=initializer_mean,use_bias=False,kernel_regularizer=None,activity_regularizer=None),
              
            ]

  
  blocks_2=[
            Dense(12*12*16,kernel_initializer='normal',use_bias=True,kernel_regularizer=None,activity_regularizer=None),
            Reshape((16,12,12)),
            Activation('relu'),
            Conv2D(filters=32, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
            Activation('relu'),
            Upsample((2,2),data_format='channels_first'),            
            ZP(((1,0),(1,0)),data_format='channels_first'),          
            Conv2D(filters=64, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
            Activation('relu'),
            Upsample((2,2),data_format='channels_first'),
            Conv2D(filters=1, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
            Activation('sigmoid'),
  ]

  return tf.keras.Sequential(blocks_1+blocks_2)



def perception_network(output_size ,HEIGHT_in = 30,WIDTH_in = 9):
    
  # inputs= tf.keras.layers.Input(shape=[1,HEIGHT_in,WIDTH_in])

  initializer_mean = tf.keras.initializers.RandomNormal(mean=0., stddev=0.001)
  # initializer_log_var = tf.keras.initializers.RandomNormal(mean=-2.5, stddev=0.001)

  blocks_1=[ 
            Dense(output_size, kernel_initializer='normal',use_bias=True,kernel_regularizer=None,activity_regularizer=None),
            Activation('sigmoid'),
  ]

  return tf.keras.Sequential(blocks_1)


