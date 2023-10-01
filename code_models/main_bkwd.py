import tensorflow as tf
import os
import time
import numpy as np
from matplotlib import pyplot as plt
from IPython import display
from math import floor
from RGCDataset import load_data
from train_test_fwd import train_backward
from dim_reduction_utils import tSNE_transform
from Fwd_Models import ML_perception
import logging
tf.get_logger().setLevel(logging.ERROR)


smoothed_data=True

if smoothed_data:
    DIR_fw_main='../data/dataset-smoothed/'
    # DIR_fw_main='../data/dataset-wn/'
    
else:

        DIR_fw_main='../data/dataset/'
        # DIR_fw_main='../data/dataset-wn/'

# _____________________
num_of_epochs = 200
num_of_epochs_actor=1
frame_rate=0.01
mean_firing_rate=0.05

n_iter=1
beta_vae=500
latent_dims=15
noise=tf.random.normal(shape=[1000,latent_dims])
# _____________________


height=50
width=50
num_neurons=9



BATCH_SIZE=287
BATCH_SIZE_val=72
BATCH_SIZE_test=5


# -------------------------original data complete size------------------

DIR_fw_main_test = DIR_fw_main
loader_fw_main=load_data(DIR_fw_main,DIR_fw_main_test,'white_noise',height=height,width=width,num_neurons=num_neurons)
train_dataset_fw_main,val_dataset_fw_main,test_dataset_fw_main=loader_fw_main.make_dataset()
train_dataset_fw_main=train_dataset_fw_main.batch(1)
val_dataset_fw_main=val_dataset_fw_main.batch(1)
test_dataset_fw_main=test_dataset_fw_main.batch(1)


WIDTH_spikes = 1
HIGHT_spikes = loader_fw_main.num_neurons # nb_cells
WIDTH_stimuli = loader_fw_main.width
HEIGHT_stimuli = loader_fw_main.height
NUM_NEURONS= loader_fw_main.num_neurons # nb_cells



TIME_LENGTH=1000
WINDOW_LENGTH=30
Y_DATA_DIM=WIDTH_stimuli*HEIGHT_stimuli
X_DATA_DIM=WIDTH_spikes*HIGHT_spikes

WIDTH_transformed=3
HEIGHT_transformed=3

# ------------------------------- main forward model---------------------------------
checkpoint_dir_fw_main='./training_checkpoints/'
latest = tf.train.latest_checkpoint(checkpoint_dir_fw_main)


model= ML_perception( X_DATA_DIM,Y_DATA_DIM, TIME_LENGTH,WINDOW_LENGTH)
model.build_model()
model_sub_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4,beta_1=0.9)


print('latest checlpoint: {}'.format(latest))

checkpoint_fw_main = tf.train.Checkpoint(model_sub_optimizer=model_sub_optimizer,
                                             model=model)

status=checkpoint_fw_main.restore(latest)

model=checkpoint_fw_main.model


manager_fwd_main_model=tf.train.CheckpointManager(
    checkpoint_fw_main, checkpoint_dir_fw_main, max_to_keep=5)
# -------------------------------------------------





def fit(train_ds, val_ds,test_ds,model_sub_local,
                        model_sub_optimizer_local,num_of_epochs,iter_i):

    # Keep track of the best model
    manager_fwd_main_model.save()
    best_loss = 1e8
    best_corr=-1e8

    # Track the train and validation loss
    train_losses = []
    val_losses = []
    test_losses = []
    gen_losses = []
    
    train_corrs=[]
    val_corrs=[]
    test_corrs=[]
    gen_corrs=[]
    for epoch in range(num_of_epochs):
        print(epoch)
        # for phase in ['train', 'val', 'test','generation']:
        for phase in ['train', 'val', 'test']:    
            # track the running loss over batches
          running_loss = 0
          running_corr = 0
          running_size = 0
          if phase =='train':
            for n, (stimuli, spikes) in train_ds.enumerate():
               # print(n.numpy())
               
               
               stimuli=tf.transpose(stimuli,perm=[1,0,2])
               # spikes=tf.squeeze(spikes,axis=[0])
               y_pred_t,stimuli_t,loss_t,corr_t= train_backward(stimuli, spikes, epoch + num_of_epochs*(iter_i),'train',
                                                        model_sub_local,
                                                        model_sub_optimizer_local
                                                        )
               # print(loss_t.numpy())
               assert tf.math.is_finite(loss_t)
               running_loss += loss_t.numpy()
               running_corr +=corr_t.numpy()
               running_size += 1
          elif phase == 'val':
              
            for n, (stimuli, spikes) in val_ds.enumerate():
               stimuli=tf.transpose(stimuli,perm=[1,0,2])
               # spikes=tf.squeeze(spikes,axis=[0])
               y_pred_t,stimuli_t,loss_t,corr_t= train_backward(stimuli, spikes,epoch + num_of_epochs*(iter_i),'val',
                                                        model_sub_local,
                                                        model_sub_optimizer_local
                                                        )
               running_loss += loss_t.numpy()
               running_corr +=corr_t.numpy()
               running_size += 1
          elif phase == 'test':
              
            for n, (stimuli, spikes) in test_ds.enumerate():
               stimuli=tf.transpose(stimuli,perm=[1,0,2])
               # spikes=tf.squeeze(spikes,axis=[0])
               y_pred_t,stimuli_t,loss_t,corr_t= train_backward(stimuli, spikes, epoch + num_of_epochs*(iter_i),'test',
                                                        model_sub_local,
                                                        model_sub_optimizer_local
                                                       )
               assert tf.math.is_finite(loss_t)
               running_loss += loss_t.numpy()
               running_corr +=corr_t.numpy()
               running_size += 1


            # compute the train/validation loss and update the best
            # model parameters if this is the lowest validation loss yet
          running_loss /= running_size
          running_corr/=running_size
          
          if phase == "train":
                train_losses.append(running_loss)
                train_corrs.append(running_corr)
 
          elif phase == 'val':
                val_losses.append(running_loss)
                print('running loss: '+ str(running_loss)+'best loss: '+str(best_loss))
                val_corrs.append(running_corr)
                if running_loss < best_loss:
                # if running_corr > best_corr:
                    best_loss = running_loss
                    best_corr = running_corr
                    manager_fwd_main_model.save()
          elif phase == 'test':
                test_losses.append(running_loss)
                test_corrs.append(running_corr)
                
          # elif phase == 'generation':
          #       gen_losses.append(running_loss)
          #       gen_corrs.append(running_corr)



    plt.figure()
    plt.plot(range(len(train_losses)),train_losses)
    plt.plot(range(len(val_losses)),val_losses)
    plt.plot(range(len(test_losses)),test_losses)
    plt.plot(range(len(gen_losses)),gen_losses)
    plt.title('fwd loss iter {}'.format(iter_i))
    
    plt.figure()
    
    plt.plot(range(len(train_corrs)),train_corrs)
    plt.plot(range(len(val_corrs)),val_corrs)
    plt.plot(range(len(test_corrs)),test_corrs)
    plt.plot(range(len(gen_corrs)),gen_corrs)
    plt.title('fwd corr iter {}'.format(iter_i))
    

    return [train_losses,train_corrs],[val_losses,val_corrs],[test_losses,test_corrs]





def backward_model_iter(n_iter,model,model_sub_optimizer):

    iter_train_ds=train_dataset_fw_main
    iter_val_ds=val_dataset_fw_main

        
        # # # # input('press something')
    train_fwd, val_fwd, test_fwd = fit(iter_train_ds, iter_val_ds,test_dataset_fw_main,
                                                                model,
                                                                model_sub_optimizer,num_of_epochs,0)

    return train_fwd, val_fwd,  test_fwd




 

train_fwd, val_fwd,  test_fwd = backward_model_iter(n_iter,model,model_sub_optimizer)


np.squeeze(np.array(train_fwd)).tofile('./train_fwd')
np.squeeze(np.array(val_fwd)).tofile('./val_fwd')
np.squeeze(np.array(test_fwd)).tofile('./test_fwd')


