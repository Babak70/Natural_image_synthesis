import tensorflow as tf
import shutil
import numpy as np
from matplotlib import pyplot as plt
from RGCDataset import load_data
from train_test import train_vae, adaptive_train_fwd, adaptive_train_actor
from test_evals import test_forward
from Generative_Models import VAE,GP_VAE, BandedJointEncoder
from Fwd_Models import ML_fwd, ML_actor
from latent_analysis import reconstruct_stimuli,z_traverse, beta_traverse_ablation,response_plot, response_plot_actor, beta_traverse_emb,z_cumulative

import sys
import logging
tf.get_logger().setLevel(logging.ERROR)
import os

import gc
from tensorflow.keras import backend as k
MAIN_DIR='../data/dataset/'
import argparse

# num_of_epochs = 100
# num_of_epochs_actor = 60
# n_iter = 3
# beta_vae = 700
# latent_dims = 15

# _____________________
latent_dims_actor = 15
stim_type = 'natural'
dataset_smoothed = True
TIME_LENGTH =1000
latent_TIME_LENGTH=500
# _____________________

BATCH_NUM = 287
BATCH_NUM_val = 72
BATCH_NUM_test = 5
mean_firing_rate = 0.05

BATCH_size = 1
BATCH_size_val = 1
BATCH_size_test = 1

curr_dir = os.getcwd()
if dataset_smoothed:
    
    DIR_fw_main =   '../data/dataset-smoothed/'
    DIR_fw_sub0 =    '../data/dataset-smoothed-iteration/0-3ed-original/'
    DIR_fw_sub =   '../data/dataset-smoothed-iteration/0-3ed/'
    
else:
    DIR_fw_main =   '../data/dataset/'
    DIR_fw_sub0 =  '../data/dataset-iteration/0-3ed-original/'
    DIR_fw_sub =   '../data/dataset-iteration/0-3ed/'

# -------------------------original data complete size------------------

DIR_fw_main_test = DIR_fw_main
loader_fw_main=load_data(DIR_fw_main, DIR_fw_main_test, stim_type)
train_dataset_fw_main,val_dataset_fw_main,test_dataset_fw_main=loader_fw_main.make_dataset()
train_dataset_fw_main=train_dataset_fw_main.batch(BATCH_size)
val_dataset_fw_main=val_dataset_fw_main.batch(BATCH_size_val)
test_dataset_fw_main=test_dataset_fw_main.batch(BATCH_size_test)

# # -------------------------original data, one third size-----------------

DIR_fw_sub_test0 = DIR_fw_main_test
loader_fw_sub0 = load_data(DIR_fw_sub0,DIR_fw_sub_test0, stim_type)
train_dataset_fw_sub0, val_dataset_fw_sub0, _= loader_fw_sub0.make_dataset()
train_dataset_fw_sub0 = train_dataset_fw_sub0.batch(1)
val_dataset_fw_sub0 = val_dataset_fw_sub0.batch(1)

# # -------------------------------------------------

DIR_fw_sub_test = DIR_fw_main_test
loader_fw_sub = load_data(DIR_fw_sub,DIR_fw_sub_test, stim_type)
train_dataset_fw_sub, val_dataset_fw_sub, _= loader_fw_sub.make_dataset()
train_dataset_fw_sub = train_dataset_fw_sub.batch(1)
val_dataset_fw_sub = val_dataset_fw_sub.batch(1)
# test_dataset_fw_sub=test_dataset_fw_sub.batch(1)
# # # -------------------------------------------------



WIDTH_spikes = 1
HIGHT_spikes = loader_fw_main.num_neurons # nb_cells
WIDTH_stimuli = loader_fw_main.width
HEIGHT_stimuli = loader_fw_main.height
NUM_NEURONS= loader_fw_main.num_neurons # nb_cells




X_DATA_DIM=WIDTH_stimuli*HEIGHT_stimuli
Y_DATA_DIM=WIDTH_spikes*HIGHT_spikes


# -------------------------------------------------

def fit(train_ds, val_ds,test_ds,model_local,
                        model_optimizer_local,num_of_epochs,iter_i, train_model, manager,second_model=None, TV=None,alpha=None):

    # Keep track of the best model
    manager.save()
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
    
    train_nll = []
    val_nll = []
    test_nll = []

    
    train_kl = []
    val_kl = []
    test_kl = []
 
    
    train_mse = []
    val_mse = []
    test_mse = []

    
    for epoch in range(num_of_epochs):
        print(epoch)

        for phase in ['train', 'val', 'test']:    
            # track the running loss over batches
          running_loss = 0
          running_corr = 0
          running_nll = 0
          running_kl = 0
          running_mse = 0
          running_size = 0
          if phase =='train':
            gc.collect()
            k.clear_session()
            for n, (stimuli, spikes) in train_ds.enumerate():
               # print(n.numpy())
               
               
               stimuli=tf.transpose(stimuli,perm=[1,0,2])
               if ((n.numpy()) == BATCH_NUM//BATCH_size) and (BATCH_NUM%BATCH_size!=0) :
                   multiplyer = BATCH_size-(BATCH_NUM%BATCH_size)
                   stimuli=tf.concat([stimuli,tf.tile(stimuli[:,-1,:][:,tf.newaxis,:],[1,multiplyer,1])],axis=1)
                   spikes=tf.concat([spikes,tf.tile(spikes[-1,:,:][tf.newaxis,...],[multiplyer,1,1])],axis=0)
               # spikes=tf.squeeze(spikes,axis=[0])
               loss_t,nll_t,kl_t,mse_t,corr_t = train_model(stimuli, spikes, epoch + num_of_epochs*(iter_i),'train',
                                                        model_local,
                                                        model_optimizer_local,
                                                        beta_vae, BATCH_size,
                                                        second_model=second_model,
                                                        TV=TV,alpha=alpha)
               # print(loss_t.numpy())
               assert tf.math.is_finite(loss_t)
               running_loss += loss_t.numpy()
               running_corr +=corr_t.numpy()
               running_nll += nll_t.numpy()
               running_kl +=kl_t.numpy()
               running_mse +=mse_t.numpy()
               running_size += 1
          elif phase == 'val':
            gc.collect()
            k.clear_session()
            for n, (stimuli, spikes) in val_ds.enumerate():
               stimuli=tf.transpose(stimuli,perm=[1,0,2])
               # spikes=tf.squeeze(spikes,axis=[0])
               loss_t,nll_t,kl_t,mse_t,corr_t = train_model(stimuli, spikes,epoch + num_of_epochs*(iter_i),'val',
                                                        model_local,
                                                        model_optimizer_local,
                                                        beta_vae, BATCH_size,
                                                        second_model=second_model,
                                                        TV=TV,alpha=alpha)
               assert tf.math.is_finite(loss_t) 
               running_loss += loss_t.numpy()
               running_corr +=corr_t.numpy()
               running_nll += nll_t.numpy()
               running_kl +=kl_t.numpy()
               running_mse +=mse_t.numpy()  
               running_size += 1
          elif phase == 'test':
            gc.collect()
            k.clear_session()             
            for n, (stimuli, spikes) in test_ds.enumerate():
               stimuli=tf.transpose(stimuli,perm=[1,0,2])
               # spikes=tf.squeeze(spikes,axis=[0])
               loss_t,nll_t,kl_t,mse_t,corr_t = train_model(stimuli, spikes, epoch + num_of_epochs*(iter_i),'test',
                                                        model_local,
                                                        model_optimizer_local,
                                                        beta_vae, BATCH_size,
                                                        second_model=second_model,
                                                        TV=TV,alpha=alpha)
               assert tf.math.is_finite(loss_t)
               running_loss += loss_t.numpy()
               running_corr +=corr_t.numpy()
               running_nll += nll_t.numpy()
               running_kl +=kl_t.numpy()
               running_mse +=mse_t.numpy()               
               running_size += 1
            # compute the train/validation loss and update the best
            # model parameters if this is the lowest validation loss yet

          running_loss /= running_size
          running_corr/=running_size
          
          if phase == "train":
                train_losses.append(running_loss)
                train_corrs.append(running_corr)
                train_nll.append(running_nll)
                train_kl.append(running_kl)
                train_mse.append(running_mse) 
                
          elif phase == 'val':
                val_losses.append(running_loss)
                print('running loss:'+ str(running_loss)+'best loss:'+str(best_loss))
                val_corrs.append(running_corr)
                val_nll.append(running_nll)
                val_kl.append(running_kl)
                val_mse.append(running_mse)                 
                if running_loss < best_loss:
                # if running_corr > best_corr:
                    best_loss = running_loss
                    best_corr = running_corr
                    manager.save()
          elif phase == 'test':
                test_losses.append(running_loss)
                test_corrs.append(running_corr)
                test_nll.append(running_nll)
                test_kl.append(running_kl)
                test_mse.append(running_mse)               


    plt.figure()
    plt.plot(range(len(train_losses)),train_losses)
    plt.plot(range(len(val_losses)),val_losses)
    plt.plot(range(len(test_losses)),test_losses)
    plt.plot(range(len(gen_losses)),gen_losses)
    if second_model ==None:
        plt.title('fwd loss iter {}'.format(iter_i))
    else:
        plt.title('actor loss iter {}'.format(iter_i))
    plt.show()
    plt.savefig('./loss progress.png')
    plt.figure()
    
    plt.plot(range(len(train_corrs)),train_corrs)
    plt.plot(range(len(val_corrs)),val_corrs)
    plt.plot(range(len(test_corrs)),test_corrs)
    plt.plot(range(len(gen_corrs)),gen_corrs)
    if second_model ==None:
        plt.title('fwd loss iter {}'.format(iter_i))
    else:
        plt.title('actor loss iter {}'.format(iter_i))
    plt.show()
    plt.savefig('./corr progress.png')
    return [train_losses,train_corrs,train_nll,train_kl,train_mse],[val_losses,val_corrs,val_nll,val_kl,val_mse],[test_losses,test_corrs,test_nll,test_kl,test_mse]


def vae_training(model_type):
    
    # ------------------------------- main forward model---------------------------------
    checkpoint_dir_fw_main='./training_checkpoints/'
    latest = tf.train.latest_checkpoint(checkpoint_dir_fw_main)
    
    if model_type == "IB-Disjoint":
        vae = VAE(latent_dims, X_DATA_DIM,Y_DATA_DIM, TIME_LENGTH, beta=beta_vae)
    elif model_type == "IB-GP":
        # vae= GP_VAE(latent_dims, X_DATA_DIM,Y_DATA_DIM, TIME_LENGTH, beta=beta_vae)
        vae= GP_VAE(latent_dims, X_DATA_DIM,Y_DATA_DIM, TIME_LENGTH, beta=beta_vae, encoder=BandedJointEncoder,\
            length_scale=1, kernel_scales=1,latent_time_length=latent_TIME_LENGTH)
        # vae= GP_VAE(latent_dims, X_DATA_DIM,Y_DATA_DIM, TIME_LENGTH, beta=beta_vae, encoder=BandedJointEncoder,\
                                                      # kernel="diffusion", length_scale=0.4,  latent_time_length=latent_TIME_LENGTH)
    else:
        sys.exit("Model_type not correct. Choose IB-Disjoint or IB-GP")
        
    vae.build_model() #importnat. To avoid exceptions thrown in tfp with non-eager mode
    vae_sub_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4,beta_1=0.9)
    print('latest checlpoint: {}'.format(latest))

    checkpoint_fw_main = tf.train.Checkpoint(vae_sub_optimizer=vae_sub_optimizer,
                                             vae=vae)
    status=checkpoint_fw_main.restore(latest)
    vae=checkpoint_fw_main.vae
    vae_sub_optimizer=checkpoint_fw_main.vae_sub_optimizer

    manager_fwd_main_vae=tf.train.CheckpointManager(
    checkpoint_fw_main, checkpoint_dir_fw_main, max_to_keep=5)
    for i in range(0,1):
        
        iter_train_ds=train_dataset_fw_main
        iter_val_ds=val_dataset_fw_main
       
        

        train_fwd, val_fwd, test_fwd = fit(iter_train_ds, iter_val_ds, test_dataset_fw_main,
                                                                vae, vae_sub_optimizer, num_of_epochs, i, train_vae,manager_fwd_main_vae)

    np.squeeze(np.array(train_fwd)).tofile('./train_fwd')
    np.squeeze(np.array(val_fwd)).tofile('./val_fwd')
    np.squeeze(np.array(test_fwd)).tofile('./test_fwd')
    return train_fwd, val_fwd,  test_fwd


def vae_training_iter(n_iter):
    MAIN_dir_fwd_true = './adaptive-optimization/fwd-model-true'
    MAIN_dir_actor = './adaptive-optimization/actor-model'
    MAIN_dir_fwd = './adaptive-optimization/fwd-model'
# clean up

    shutil.rmtree('./Z', ignore_errors=True)

    shutil.rmtree('./opt-stimuli', ignore_errors=True)

    # shutil.rmtree(MAIN_dir_fwd + '/training_checkpoints', ignore_errors=True)

    shutil.rmtree(MAIN_dir_actor + '/training_checkpoints', ignore_errors=True)


    checkpoint_dir_fwd_true =  MAIN_dir_fwd_true + '/training_checkpoints/'
    
    
    # model_true= VAE(latent_dims, X_DATA_DIM,Y_DATA_DIM, TIME_LENGTH, beta=700)
    # # model_true= GP_VAE(latent_dims, X_DATA_DIM,Y_DATA_DIM, TIME_LENGTH, beta=beta_vae, encoder=BandedJointEncoder,\
    # #         length_scale=1, kernel_scales=1,latent_time_length=latent_TIME_LENGTH)
    
    # checkpoint_fwd_true = tf.train.Checkpoint(vae=model_true)
    # latest = tf.train.latest_checkpoint(checkpoint_dir_fwd_true)
    # print('latest checkpoint: {}'.format(latest))
    # status= checkpoint_fwd_true.restore(latest)
    # model_true = checkpoint_fwd_true.vae
    
    model_true= ML_fwd( X_DATA_DIM,Y_DATA_DIM, TIME_LENGTH)
    checkpoint_fwd_true = tf.train.Checkpoint(model=model_true)
    latest = tf.train.latest_checkpoint(checkpoint_dir_fwd_true)
    print('latest checkpoint: {}'.format(latest))
    status= checkpoint_fwd_true.restore(latest)
    model_true = checkpoint_fwd_true.model
    
    
    
    corr_AC=[]
    loss_AC=[]

    
    checkpoint_dir_fwd =  MAIN_dir_fwd + '/training_checkpoints/'


    # model= VAE(latent_dims, X_DATA_DIM,Y_DATA_DIM, TIME_LENGTH, beta=beta_vae)
    model = GP_VAE(latent_dims, X_DATA_DIM,Y_DATA_DIM, TIME_LENGTH, beta=beta_vae, encoder=BandedJointEncoder,\
            length_scale=1, kernel_scales=1,latent_time_length=latent_TIME_LENGTH)
        
    model.build_model()
        
    model_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4,beta_1=0.9)
    checkpoint_fwd= tf.train.Checkpoint(vae=model, vae_sub_optimizer=model_optimizer)
    
    manager_fwd=tf.train.CheckpointManager(
        checkpoint_fwd, checkpoint_dir_fwd, max_to_keep=5)
    
    
    checkpoint_dir_actor =  MAIN_dir_actor + '/training_checkpoints/'
    
    actor= ML_actor(X_DATA_DIM,latent_dims_actor, TIME_LENGTH)

            
    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4,beta_1=0.9)
    checkpoint_actor = tf.train.Checkpoint(actor=actor,actor_optimizer=actor_optimizer)

    manager_actor=tf.train.CheckpointManager(
        checkpoint_actor, checkpoint_dir_actor, max_to_keep=5)

    for i in range(0, n_iter):
        

        # load best model weights
        latest = tf.train.latest_checkpoint(checkpoint_dir_fwd)
        print('latest checkpoint: {}'.format(latest))
        status= checkpoint_fwd.restore(latest)
        model = checkpoint_fwd.vae
        model_optimizer =  checkpoint_fwd.vae_sub_optimizer
        

        # load best model weights
        latest = tf.train.latest_checkpoint(checkpoint_dir_actor)
        print('latest checkpoint: {}'.format(latest))
        status= checkpoint_actor.restore(latest)
        actor = checkpoint_actor.actor
        actor_optimizer =  checkpoint_actor.actor_optimizer
        

        
        if i == 0:
            iter_train_ds = train_dataset_fw_main
            iter_val_ds = val_dataset_fw_main
        else:
            
            
            iter_train_ds = train_dataset_fw_sub
            iter_val_ds = val_dataset_fw_sub


        # train_losses_fwd, val_losses_fwd, test_losses_fwd = fit(iter_train_ds, iter_val_ds, test_dataset_fw_main, 
        #                                                                   model, model_optimizer, num_of_epochs, i,
        #                                                                   adaptive_train_fwd, manager_fwd)
        
        
        
        train_losses_actor, val_losses_actor, test_losses_actor = fit(train_dataset_fw_main, val_dataset_fw_main,
                                                                          test_dataset_fw_main, 
                                                                          actor, actor_optimizer, num_of_epochs_actor,i,
                                                                          adaptive_train_actor,manager_actor,
                                                                          second_model=model,TV=TV,alpha=alpha)


        write_dir = [DIR_fw_sub + 'train_stimuli.bin',DIR_fw_sub + 'train_spikes.bin']
        t1, t2, _, _,_, _ = test_forward(train_dataset_fw_main, actor, model_true, write_dir = write_dir)
        
        write_dir = [DIR_fw_sub + 'val_stimuli.bin',DIR_fw_sub + 'val_spikes.bin']
        tt1, tt2, _, _,_, _ = test_forward(val_dataset_fw_main, actor, model_true,  write_dir = write_dir)
        
        
        write_dir = [DIR_fw_sub + 'test_stimuli.bin',DIR_fw_sub + 'test_spikes.bin']
        ttt1, ttt2,ttt3,ttt4,ttt5,ttt6 = test_forward(test_dataset_fw_main, actor, model_true, write_dir = write_dir)
        
        np.savetxt('./loss_test_iter-{}.txt'.format(i), np.array([ttt1]), delimiter=',')
        np.savetxt('./corr_test_iter-{}.txt'.format(i), np.array([ttt2]), delimiter=',')
        
        np.savetxt('./2DTV_tnsf_test_iter-{}.txt'.format(i), np.array([ttt3]), delimiter=',')
        np.savetxt('./2DTV_stimuli_test_iter-{}.txt'.format(i), np.array([ttt4]), delimiter=',')
        np.savetxt('./1DTV_tnsf_test_iter-{}.txt'.format(i), np.array([ttt5]), delimiter=',')
        np.savetxt('./1DTV_stimuli_test_iter-{}.txt'.format(i), np.array([ttt6]), delimiter=',')
        
        
        loss_AC.append([t1,tt1,ttt1])
        corr_AC.append([t2,tt2,ttt2])

 
    plt.figure()
    plt.plot(range(len(np.array(loss_AC)[:,0])),np.array(loss_AC)[:,0],'*')
    plt.plot(range(len(np.array(loss_AC)[:,1])),np.array(loss_AC)[:,1],'*')
    plt.plot(range(len(np.array(loss_AC)[:,2])),np.array(loss_AC)[:,2],'*')
    plt.title('loss progress')
    plt.show()
    plt.savefig('./loss progress.png')
    plt.figure()
    plt.plot(range(len(np.array(corr_AC)[:,0])),np.array(corr_AC)[:,0],'*')
    plt.plot(range(len(np.array(corr_AC)[:,1])),np.array(corr_AC)[:,1],'*')
    plt.plot(range(len(np.array(corr_AC)[:,2])),np.array(corr_AC)[:,2],'*')
    plt.title('corr progress')
    plt.show()
    plt.savefig('./corr progress.png')
    
    np.squeeze(np.array(loss_AC)).tofile('./loss_actor')
    np.squeeze(np.array(corr_AC)).tofile('./corr_actor')

    return loss_AC, corr_AC



def main(num_of_epochs, num_of_epochs_actor,n_iter,beta_vae,latent_dims,task,model_type,alpha,TV):

    if   task == "test_forward":
        
        model_types = ['IB-GP','IB-Disjoint','feedforwardCNN']
        response_plot(model_types, X_DATA_DIM, Y_DATA_DIM, TIME_LENGTH,beta_vae,latent_TIME_LENGTH,latent_dims,test_dataset_fw_main)

    elif   task == "test_forward_actor":
        
        model_types = ['IB-GP']
        response_plot_actor(model_types, X_DATA_DIM, Y_DATA_DIM, TIME_LENGTH,beta_vae,latent_TIME_LENGTH,latent_dims, latent_dims_actor,test_dataset_fw_main)
    
    elif task == "train_forward":
        
        train_fwd, val_fwd, test_fwd = vae_training(model_type)
    
    elif task == "adaptive_train":
        
        loss_actor,corr_actor = vae_training_iter(n_iter)
        
    elif task == "reconstruct_stimuli":
        
        reconstruct_stimuli(X_DATA_DIM, Y_DATA_DIM, TIME_LENGTH,beta_vae,latent_TIME_LENGTH,latent_dims,test_dataset_fw_main )
        
    elif task == "traversal":

        
        z_traverse(model_type, X_DATA_DIM, Y_DATA_DIM, TIME_LENGTH,beta_vae,latent_TIME_LENGTH,latent_dims,test_dataset_fw_main)
        
    elif task == "beta_traverse":
        
        beta_vals = [1, 3, 5, 50, 100, 300, 700, 1000, 10000]
        mask_indx =[
                     [a for a in range(latent_dims) ],
                     [a for a in range(latent_dims) if a!=0 and a!=5 and a!=8 and a!=11],
                     [a for a in range(latent_dims) if  a!=5],
                     [a for a in range(latent_dims) if  a!=1],
                     [a for a in range(latent_dims) if  a!=3 and a!=5],
                     [a for a in range(latent_dims) if  a!=8 and a!=12],
                     [a for a in range(latent_dims) if  a!=1 and a!=3 and a!=5 and a!=10 and a!=13],
                     [a for a in range(latent_dims) if  a!=1 and a!=3 and a!=4 and a!=10 and a!=12 and a!=14],
                     [a for a in range(latent_dims) if  a!=1 and a!=3 and a!=4 and a!=6 and a!=8 and a!=9 and a!=10 and a!=14],
                     ]
        
        beta_traverse_ablation(model_type, beta_vals,mask_indx, X_DATA_DIM, Y_DATA_DIM, TIME_LENGTH,beta_vae,latent_TIME_LENGTH,latent_dims,test_dataset_fw_main)

    elif task == "latent_traverse_corr":
        
        model_types = ['IB-GP', 'IB-Disjoint']
        dataset_type = 'natural'
        beta_vals = [5, 20, 30, 50]
        # beta_vals = [100,300,700]
        
        z_cumulative(model_types,  beta_vals, dataset_type, X_DATA_DIM, Y_DATA_DIM, TIME_LENGTH,beta_vae,latent_TIME_LENGTH,latent_dims,test_dataset_fw_main)
        
    elif task == "latent_analyze":
        model_type= ["IB-GP","IB-Disjoint"]
        beta_vals = [1]
        mask_indx= []
        beta_traverse_emb(model_type, beta_vals,mask_indx, X_DATA_DIM, Y_DATA_DIM, TIME_LENGTH,beta_vae,latent_TIME_LENGTH,latent_dims,test_dataset_fw_main)
    
    else:
        
        sys.exit('task not found')



parser = argparse.ArgumentParser()
parser.add_argument('-num_of_epochs', '--num_epochs', type=int, default=200)
parser.add_argument('-num_of_epochs_actor', '--num_epochs_actor',type=int, default=40)
parser.add_argument('-n_iter', '--adaptive_iteration_num', default=1)
parser.add_argument('-beta_vae', '--beta', type=float, default=20)
parser.add_argument('-latent_dims', '--latent_dims', type=int, default=15)
parser.add_argument('-task', '--task', default="test_forward")
parser.add_argument('-model_type', '--model_type', default="IB-GP")
parser.add_argument('-alpha', '--alpha', type=float, default=0)
parser.add_argument('-TV', '--TV', type=float, default=0)

args = parser.parse_args()


num_of_epochs=args.num_epochs
num_of_epochs_actor=args.num_epochs_actor
n_iter=args.adaptive_iteration_num
beta_vae=args.beta
latent_dims=args.latent_dims
task=args.task
model_type=args.model_type
alpha=args.alpha
TV=args.TV

if __name__ == "__main__":
    main(num_of_epochs, num_of_epochs_actor,n_iter,beta_vae,latent_dims,task,model_type,alpha,TV)


