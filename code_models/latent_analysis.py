import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.transforms as transforms


from Generative_Models import VAE,GP_VAE, BandedJointEncoder,PoissonDecoder_DecodeOnly
from Fwd_Models import ML_fwd, ML_actor

from test_evals import test_forward, test_forward_all, test_traverse, test_beta_traverse,test_ablation, test_response_plot,test_response_plot_actor, test_beta_traverse_emb,test_cumulative

import tikzplotlib 
import os

import sys
import itertools
from decimal import Decimal


    

def reconstruct_stimuli( X_DATA_DIM, Y_DATA_DIM, TIME_LENGTH,beta_vae,latent_TIME_LENGTH,latent_dims,test_dataset_fw_main):
    
    num_of_repeats = 1
    mask_indx = []
    main_dir = './models'
    
    loss_tnsf_GP = np.ndarray(num_of_repeats)
    corr_tnsf_GP = np.ndarray(num_of_repeats)
    loss_tnsf_GP_fwd = np.ndarray(num_of_repeats)
    corr_tnsf_GP_fwd = np.ndarray(num_of_repeats)

    loss_tnsf_Dis = np.ndarray(num_of_repeats)
    corr_tnsf_Dis = np.ndarray(num_of_repeats)
    loss_tnsf_Dis_fwd = np.ndarray(num_of_repeats)
    corr_tnsf_Dis_fwd = np.ndarray(num_of_repeats)
    
    
    loss_tnsf_GP_1 = np.ndarray(num_of_repeats)
    corr_tnsf_GP_1 = np.ndarray(num_of_repeats)
    loss_tnsf_GP_fwd_1 = np.ndarray(num_of_repeats)
    corr_tnsf_GP_fwd_1 = np.ndarray(num_of_repeats)

    loss_tnsf_Dis_1 = np.ndarray(num_of_repeats)
    corr_tnsf_Dis_1 = np.ndarray(num_of_repeats)
    loss_tnsf_Dis_fwd_1 = np.ndarray(num_of_repeats)
    corr_tnsf_Dis_fwd_1 = np.ndarray(num_of_repeats)
    
    loss_pca_fwd = np.ndarray(num_of_repeats)
    corr_pca_fwd = np.ndarray(num_of_repeats)
    loss_fwd = np.ndarray(num_of_repeats)
    corr_fwd = np.ndarray(num_of_repeats)
    loss_random = np.ndarray(num_of_repeats)
    corr_random = np.ndarray(num_of_repeats)
 

    
    checkpoint_dir_vae = main_dir + '/IB-GP/stim-recons/' + '/training_checkpoints/'
       
    vae= GP_VAE(latent_dims, X_DATA_DIM,Y_DATA_DIM, TIME_LENGTH, beta=beta_vae, encoder=BandedJointEncoder,\
                      length_scale=1, kernel_scales=1,latent_time_length=latent_TIME_LENGTH)
    checkpoint_vae = tf.train.Checkpoint(vae=vae)
            # load best model weights
    latest = tf.train.latest_checkpoint(checkpoint_dir_vae)
    print('latest checkpoint: {}'.format(latest))
    if latest is None:
            sys.exit("Model not found. Put the model in the correct path")
    status=checkpoint_vae.restore(latest)
    model_GP_vae=checkpoint_vae.vae
        
        

    checkpoint_dir_vae = main_dir + '/IB-Disjoint/stim-recons/' + '/training_checkpoints/'
    vae= VAE(latent_dims, X_DATA_DIM,Y_DATA_DIM, TIME_LENGTH, beta=beta_vae)

    checkpoint_vae = tf.train.Checkpoint(vae=vae)
            # load best model weights
    latest = tf.train.latest_checkpoint(checkpoint_dir_vae)
    print('latest checkpoint: {}'.format(latest))
    if latest is None:
            sys.exit("Model not found. Put the model in the correct path")
    status=checkpoint_vae.restore(latest)
    model_vae=checkpoint_vae.vae
    
    
    for i in range(num_of_repeats):
    
        checkpoint_dir_fwd = main_dir +'/feedforward-CNN/' + str(int(i+1)) + '/training_checkpoints/'
        model= ML_fwd(X_DATA_DIM,Y_DATA_DIM, TIME_LENGTH)
        checkpoint_fwd = tf.train.Checkpoint(model=model)
        # load best model weights
        latest = tf.train.latest_checkpoint(checkpoint_dir_fwd)
        print('latest checkpoint: {}'.format(latest))
        if latest is None:
            sys.exit("Model not found. Put the model in the correct path")
        status= checkpoint_fwd.restore(latest)
        model_fwd= checkpoint_fwd.model
        
    

    # # IB-GP
        loss_tnsf_GP[i],corr_tnsf_GP[i],loss_tnsf_GP_fwd[i],corr_tnsf_GP_fwd[i] \
        = test_forward_all(test_dataset_fw_main,model_fwd,model=model_GP_vae, task='IB-GP')
    # # IB-Disjoint
        loss_tnsf_Dis[i],corr_tnsf_Dis[i],loss_tnsf_Dis_fwd[i],corr_tnsf_Dis_fwd[i] \
        = test_forward_all(test_dataset_fw_main,model_fwd,model=model_vae, task='IB-Disjoint')

    # # IB-GP-1latent
        mask_indx=[0,2]
        mask = np.zeros((1,model_GP_vae.latent_dim, model_GP_vae.latent_time_length),dtype=np.float32)
        mask[:,mask_indx,:]=1.0
        mask = tf.convert_to_tensor(mask)
        loss_tnsf_GP_1[i],corr_tnsf_GP_1[i],loss_tnsf_GP_fwd_1[i],corr_tnsf_GP_fwd_1[i] \
        = test_forward_all(test_dataset_fw_main,model_fwd,model=model_GP_vae, task='IB-GP',mask=mask)
        
    # # IB-Disjoint-1latent
        mask_indx=[10]
        mask = np.zeros((1,model_vae.time_length//2, model_vae.latent_dim),dtype=np.float32)
        mask[:,:, mask_indx]=1.0
        mask = tf.convert_to_tensor(mask)
        loss_tnsf_Dis_1[i],corr_tnsf_Dis_1[i],loss_tnsf_Dis_fwd_1[i],corr_tnsf_Dis_fwd_1[i] \
        = test_forward_all(test_dataset_fw_main,model_fwd,model=model_vae, task='IB-Disjoint',mask=mask)

    # # PCA:
        loss_pca_fwd[i], corr_pca_fwd[i]\
            =test_forward_all(test_dataset_fw_main,model_fwd,task='PCA')
    
    # forward-model
        loss_fwd[i],corr_fwd[i]\
            =test_forward_all(test_dataset_fw_main,model_fwd,task='forward-model')
    
    # Random uniform
        loss_random[i],corr_random[i]\
            =test_forward_all(test_dataset_fw_main,model_fwd,task='random')
    
    # stim-visual-all three 
        # test_forward_all(test_dataset_fw_main,model_fwd,model=model_GP_vae,task='stimuli-vis',mask_indx)


    plt.figure()
    x = [1, 2, 3, 4, 5, 6]
    
    height = [loss_tnsf_GP_fwd.mean(),loss_tnsf_Dis_fwd.mean(),
             loss_tnsf_GP_fwd_1.mean(),loss_tnsf_Dis_fwd_1.mean(),
             loss_pca_fwd.mean(), loss_random.mean()]
    
    yerr =   [loss_tnsf_GP_fwd.std(),loss_tnsf_Dis_fwd.std(),
             loss_tnsf_GP_fwd_1.std(),loss_tnsf_Dis_fwd_1.std(),
              loss_pca_fwd.std(), loss_random.std()]
    
    plt.bar(x, height,yerr=yerr, width=0.5, bottom=[0]*len(x), align='center', alpha=0.5,
                                                    tick_label=['IB-GP','IB-Dis','IB-GP-one-Latent','IB-Dis-one-Latent','PCA','Random'],
                                                    color =['c', 'm', 'y', 'r', 'g', 'k'],
                                                    ecolor='black',capsize=5)
    x = [0,2,3,4,5,6.25]
    plt.plot(x,[loss_fwd.mean()]*len(x),'--k')
    y_l = loss_fwd.mean()-loss_fwd.std()
    y_u = loss_fwd.mean()+loss_fwd.std()
    plt.fill_between(np.array(x), np.array([y_l]*len(x)) , np.array([y_u]*len(x)),facecolor='#C5C9C7')
    plt.xlim(0.7,6.3)
    plt.ylabel('Poisson loss')
    plt.xticks(rotation = 45)
    plt.tight_layout()
    if os.path.isdir('./figures')==False:
        os.mkdir('./figures')
    plt.savefig('./figures/Reconstructed-stimuli-response-perfprmance-loss.png')
    tikzplotlib.save('./figures/Reconstructed-stimuli-response-perfprmance-loss.tex')


    plt.figure()
    x = [1, 2, 3, 4, 5, 6.25]
    height = [corr_tnsf_GP_fwd.mean(), corr_tnsf_Dis_fwd.mean(),
              corr_tnsf_GP_fwd_1.mean(), corr_tnsf_Dis_fwd_1.mean(),
             corr_pca_fwd.mean(), abs(corr_random.mean())*2]
    yerr = [corr_tnsf_GP_fwd.std(), corr_tnsf_Dis_fwd.std(),
            corr_tnsf_GP_fwd_1.std(), corr_tnsf_Dis_fwd_1.std(),
            corr_pca_fwd.std(),  0*corr_random.std()]

    plt.bar(x, height, yerr=yerr, width=0.5, bottom=[0]*len(x), align='center',alpha=0.5,
                                                       color =['c', 'm', 'y', 'r', 'g', 'k'],
                                                     tick_label=['IB-GP','IB-Dis','IB-GP-one-Latent','IB-Dis-one-Latent','PCA','Random'],
                                                      ecolor='black',capsize=5)
    x = [0,2,3,4,5,6.25]
    plt.plot(x,[corr_fwd.mean()]*len(x),'--k')
    y_l = corr_fwd.mean()-corr_fwd.std()
    y_u = corr_fwd.mean()+corr_fwd.std()
    plt.fill_between(x, np.array([y_l]*len(x)), np.array([y_u]*len(x)),facecolor='#C5C9C7')
    plt.xlim(0.7,6.3)
    plt.ylabel('Pearson correlation')
    plt.xticks(rotation = 45)
    plt.tight_layout()
    if os.path.isdir('./figures')==False:
        os.mkdir('./figures')
    plt.savefig('./figures/Reconstructed-stimuli-response-perfprmance-corr.png')
    tikzplotlib.save('./figures/Reconstructed-stimuli-response-perfprmance-corr.tex')

    print()
    print('IB-GP: {}'.format(corr_tnsf_GP_fwd.mean()))
    print()
    print('IB-Disjoint: {}'.format(corr_tnsf_Dis_fwd.mean()))
    print()
    print('IB-GP-1: {}'.format(corr_tnsf_GP_fwd_1.mean()))
    print()
    print('IB-Disjoint-1: {}'.format(corr_tnsf_Dis_fwd_1.mean()))
    print()
    print('PCA: {}'.format(corr_pca_fwd.mean()))
    print()
    print('random: {}'.format(corr_random.mean()))
    print()
    print('fwd-model: {}'.format(corr_fwd.mean()))
    print()
    print('tnsf: {}'.format(corr_tnsf_GP.mean()))
    
    print('tnsf: {}'.format(corr_tnsf_Dis.mean()))

    plt.figure()
    plt.plot(loss_tnsf_GP.mean(), '*', label = 'tnsf-GP')
    plt.plot(loss_tnsf_GP_fwd.mean() , '*', label = 'tnsf_fwd-GP')
    plt.plot(loss_tnsf_Dis.mean(), '*', label = 'tnsf-Dis')
    plt.plot(loss_tnsf_Dis_fwd.mean() , '*', label = 'tnsf_fwd-Dis')
    plt.plot(loss_tnsf_GP_1.mean(), '*', label = 'tnsf-GP-1')
    plt.plot(loss_tnsf_GP_fwd_1.mean() , '*', label = 'tnsf_fwd-GP-1')
    plt.plot(loss_tnsf_Dis_1.mean(), '*', label = 'tnsf-Dis-1')
    plt.plot(loss_tnsf_Dis_fwd_1.mean() , '*', label = 'tnsf_fwd-Dis-1')
    plt.plot(loss_fwd.mean(), '*', label = 'fwd')
    plt.plot(loss_pca_fwd.mean() , '*', label = 'pca_fwd')
    plt.plot(loss_random.mean(), '*', label = 'random')
    plt.title('loss')
    plt.legend(loc="upper right")
    plt.show()
    
    
    plt.figure()
    plt.plot(corr_tnsf_GP.mean(), '*', label = 'tnsf-GP')
    plt.plot(corr_tnsf_GP_fwd.mean(), '*', label = 'tnsf_fwd-GP')
    plt.plot(corr_tnsf_Dis.mean(), '*', label = 'tnsf-Dis')
    plt.plot(corr_tnsf_Dis_fwd.mean(), '*', label = 'tnsf_fwd-Dis')
    plt.plot(corr_tnsf_GP_1.mean(), '*', label = 'tnsf-GP-1')
    plt.plot(corr_tnsf_GP_fwd_1.mean(), '*', label = 'tnsf_fwd-GP-1')
    plt.plot(corr_tnsf_Dis_1.mean(), '*', label = 'tnsf-Dis-1')
    plt.plot(corr_tnsf_Dis_fwd_1.mean(), '*', label = 'tnsf_fwd-Dis-1')
    plt.plot(corr_fwd.mean(), '*', label = 'fwd')
    plt.plot(corr_pca_fwd.mean(), '*', label = 'pca_fwd')
    plt.plot(corr_random.mean(), '*', label = 'random')
    plt.title('corr')
    plt.legend(loc="upper right")
    plt.show()
    
    
def z_cumulative(model_types, beta_vals,dataset_type, X_DATA_DIM, Y_DATA_DIM, TIME_LENGTH,beta_vae,latent_TIME_LENGTH,latent_dims,test_dataset_fw_main):
    
    mask_indx = []
    
    markers = itertools.cycle(( '1', 'X', 's', 'd','4','>','<','x','v'))
    colors = itertools.cycle(('b', 'g', 'k', 'r', 'y'))
    main_dir = './paper-results/beta-variation'
    
    corr_orig_avg = np.ndarray((len(beta_vals),2,1))
    corr_keep_avg = np.ndarray((len(beta_vals),2,latent_dims))
    corr_orig_std = np.ndarray((len(beta_vals),2,1))
    corr_keep_std = np.ndarray((len(beta_vals),2,latent_dims))

    nll_orig_avg = np.ndarray((len(beta_vals),2,1))
    nll_keep_avg = np.ndarray((len(beta_vals),2,latent_dims))
    nll_orig_std = np.ndarray((len(beta_vals),2,1))
    nll_keep_std = np.ndarray((len(beta_vals),2,latent_dims))
    
    
    
    fig0, ax0 = plt.subplots(1,1)
    fig1, ax1 = plt.subplots(1,1)
    for i in range(len(beta_vals)):
        color=next(colors)
        for j in range(len(model_types)):
            model_type = model_types[j]
    
            if model_type=='IB-GP':
                checkpoint_dir_vae = main_dir +'/'+ dataset_type +'/IB-GP/' + str(beta_vals[i]) + '/training_checkpoints/'
            
                vae= GP_VAE(latent_dims, X_DATA_DIM,Y_DATA_DIM, TIME_LENGTH, beta=beta_vae, encoder=BandedJointEncoder,\
                          length_scale=1, kernel_scales=1,latent_time_length=latent_TIME_LENGTH)
            elif model_type=='IB-Disjoint':
                checkpoint_dir_vae = main_dir + '/'+ dataset_type +'/IB-Disjoint/' + str(beta_vals[i]) +'/training_checkpoints/'
                vae= VAE(latent_dims, X_DATA_DIM,Y_DATA_DIM, TIME_LENGTH, beta=beta_vae)
            else:
                sys.exit('Model not found in the folder')
                
                
            checkpoint_vae = tf.train.Checkpoint(vae=vae)
                # load best model weights
            latest = tf.train.latest_checkpoint(checkpoint_dir_vae)
            print('latest checkpoint: {}'.format(latest))
            if latest is None:
                    sys.exit("Model not found. Put the model in the correct path")
            status=checkpoint_vae.restore(latest)
            model_vae=checkpoint_vae.vae

            a,b,c,d, e,f,g,h = test_cumulative(test_dataset_fw_main, model_vae, mask_indx)
            corr_orig_avg [i,j,:] = a
            corr_orig_std [i,j,:] = b
            corr_keep_avg [i,j,:] = c
            corr_keep_std [i,j,:] = d
            
            nll_orig_avg [i,j,:] = e
            nll_orig_std [i,j,:] = f
            nll_keep_avg [i,j,:] = g
            nll_keep_std [i,j,:] = h
            
            # plt.plot(corr_keep_avg[1,0,:])
            # plt.plot(corr_keep_avg[1,1,:])
            # plt.show()
            
            # marker = next(markers)
            if model_type =="IB-GP":
                linestyle='solid'
            else:
                linestyle='dashed'
            marker = next(markers)
            ax0.errorbar([ii for ii in range(1,latent_dims+1)], nll_keep_avg[i,j,:],yerr= nll_keep_std [i,j,:],color=color, linewidth=1.5,label = r'$\beta$'+" = {:0.3f}, {}".format(Decimal(1/beta_vals[i]),model_types[j]),linestyle=linestyle)
            ax1.errorbar([ii for ii in range(1,latent_dims+1)], corr_keep_avg[i,j,:],yerr= corr_keep_std [i,j,:],color=color,linewidth=1.5, label = r'$\beta$'+" = {:0.3f}, {}".format(Decimal(1/beta_vals[i]),model_types[j]),linestyle=linestyle)
    x = [aa for aa in range(1,16)]
    ax1.plot(x,[0.43]*len(x),'--y',label="Feedforward CNN",linewidth= 3)
    y_l = 0.43-0.02
    y_u = 0.43+0.02
    # ax1.fill_between(x, np.array([y_l]*len(x)), np.array([y_u]*len(x)),facecolor='#C5C9C7')
    
    x = [aa for aa in range(1,16)]
    ax0.plot(x,[0.095]*len(x),'--y' ,label="Feedforward CNN",linewidth= 3)
    y_l = 0.095-0.001
    y_u = 0.095+0.001
    ax0.legend(frameon=False, fontsize=9)
    # ax0.fill_between(x, np.array([y_l]*len(x)), np.array([y_u]*len(x)),facecolor='#C5C9C7')

    
    
    ax0.set_ylabel('Negative Log Likelihood')
    ax0.set_xlabel('Number of latents')
    plt.tight_layout()
    # plt.show()
    fig0.savefig('./figures/cumulative/z-cumulative-nll.png')
    tikzplotlib.save('./figures/cumulative/z-cumulative-nll.tex',figure=fig0)
    plt.close(fig0)
    
    
    ax1.set_ylabel('Pearson Correlation')
    ax1.set_xlabel('Number of latents')
    # ax1.legend(frameon=False)
    plt.tight_layout()
    # plt.show()
    fig1.savefig('./figures/cumulative/z-cumulative-corr.png', figure=fig1)
    tikzplotlib.save('./figures/cumulative/z-cumulative-corr.tex')
    plt.close(fig1)

    return 1
            
            

def z_traverse(model_type,X_DATA_DIM, Y_DATA_DIM, TIME_LENGTH,beta_vae,latent_TIME_LENGTH,latent_dims,test_dataset_fw_main):
    
    mask_indx=[]

    main_dir = './models'
    
    if model_type=='IB-GP':
        checkpoint_dir_vae = main_dir + '/IB-GP/traverse/' + '/training_checkpoints/'
    
        vae= GP_VAE(latent_dims, X_DATA_DIM,Y_DATA_DIM, TIME_LENGTH, beta=beta_vae, encoder=BandedJointEncoder,\
                  length_scale=1, kernel_scales=1,latent_time_length=latent_TIME_LENGTH)
    elif model_type=='IB-Disjoint':
        checkpoint_dir_vae = main_dir + '/IB-Disjoint/traverse/' + '/training_checkpoints/'
        vae= VAE(latent_dims, X_DATA_DIM,Y_DATA_DIM, TIME_LENGTH, beta=beta_vae)
    else:
        sys.exit('Model not found in the folder')
        
        
    checkpoint_vae = tf.train.Checkpoint(vae=vae)
        # load best model weights
    latest = tf.train.latest_checkpoint(checkpoint_dir_vae)
    print('latest checkpoint: {}'.format(latest))
    if latest is None:
            sys.exit("Model not found. Put the model in the correct path")
    status=checkpoint_vae.restore(latest)
    model_vae=checkpoint_vae.vae
# IB
    test_traverse(test_dataset_fw_main, model_vae, mask_indx)
    # loss_tnsf.append(a)
    # corr_tnsf.append(b)
    # loss_tnsf_fwd.append(c)
    # corr_tnsf_fwd.append(d)
    
    
def response_plot(model_types,X_DATA_DIM, Y_DATA_DIM, TIME_LENGTH,beta_vae,latent_TIME_LENGTH,latent_dims,test_dataset_fw_main):
    
    
    main_dir = './models'
    models=[]
    for i in range(len(model_types)):
        model_type=model_types[i]
        
        
        if model_type=='IB-GP':
            checkpoint_dir_vae = main_dir + '/IB-GP/response_plot/' + '/training_checkpoints/'
        
            vae = GP_VAE(latent_dims, X_DATA_DIM,Y_DATA_DIM, TIME_LENGTH, beta=beta_vae, encoder=BandedJointEncoder,\
                      length_scale=1, kernel_scales=1,latent_time_length=latent_TIME_LENGTH)
            checkpoint_vae = tf.train.Checkpoint(vae=vae)
            # load best model weights
            latest = tf.train.latest_checkpoint(checkpoint_dir_vae)
            print('latest checkpoint: {}'.format(latest))
            if latest is None:
                sys.exit("Model not found. Put the model in the correct path")
            status=checkpoint_vae.restore(latest)
            model = checkpoint_vae.vae
            
        elif model_type=='IB-Disjoint':
            checkpoint_dir_vae = main_dir + '/IB-Disjoint/response-plot/' + '/training_checkpoints/'
            vae = VAE(latent_dims, X_DATA_DIM,Y_DATA_DIM, TIME_LENGTH, beta=beta_vae)
            
            checkpoint_vae = tf.train.Checkpoint(vae=vae)
            # load best model weights
            latest = tf.train.latest_checkpoint(checkpoint_dir_vae)
            print('latest checkpoint: {}'.format(latest))
            if latest is None:
                sys.exit("Model not found. Put the model in the correct path")
            status=checkpoint_vae.restore(latest)
            model = checkpoint_vae.vae
        elif model_type=='feedforwardCNN':
            
            checkpoint_dir_fwd = main_dir +'/feedforward-CNN/1/' + '/training_checkpoints/'
            model= ML_fwd(X_DATA_DIM,Y_DATA_DIM, TIME_LENGTH)
            checkpoint_fwd = tf.train.Checkpoint(model=model)
            # load best model weights
            latest = tf.train.latest_checkpoint(checkpoint_dir_fwd)
            print('latest checkpoint: {}'.format(latest))
            if latest is None:
                sys.exit("Model not found. Put the model in the correct path")
            status= checkpoint_fwd.restore(latest)
            model = checkpoint_fwd.model
        
        else:
            sys.exit('Model not found in the folder')
            
        models.append(model)


    test_response_plot(test_dataset_fw_main, models,model_types)
    
    
def response_plot_actor(model_types,X_DATA_DIM, Y_DATA_DIM, TIME_LENGTH,beta_vae,latent_TIME_LENGTH,latent_dims,latent_dims_actor,test_dataset_fw_main):
    MAIN_dir_fwd_true = './adaptive-optimization/fwd-model-true'
    MAIN_dir_actor = './adaptive-optimization/actor-model'
    MAIN_dir_fwd = './adaptive-optimization/fwd-model'
    
    checkpoint_dir_fwd_true =  MAIN_dir_fwd_true + '/training_checkpoints/'
    
    model_true= VAE(latent_dims, X_DATA_DIM,Y_DATA_DIM, TIME_LENGTH, beta=700)
    # model_true= GP_VAE(latent_dims, X_DATA_DIM,Y_DATA_DIM, TIME_LENGTH, beta=beta_vae, encoder=BandedJointEncoder,\
    #         length_scale=1, kernel_scales=1,latent_time_length=latent_TIME_LENGTH)
        
    checkpoint_fwd_true = tf.train.Checkpoint(vae=model_true)
    latest = tf.train.latest_checkpoint(checkpoint_dir_fwd_true)
    print('latest checkpoint: {}'.format(latest))
    status= checkpoint_fwd_true.restore(latest)
    model_true = checkpoint_fwd_true.vae
    

    for i in range(len(model_types)):
        model_type=model_types[i]
        
        
        if model_type=='IB-GP':
            checkpoint_dir_vae = MAIN_dir_fwd + '/training_checkpoints/'
        
            vae = GP_VAE(latent_dims, X_DATA_DIM,Y_DATA_DIM, TIME_LENGTH, beta=beta_vae, encoder=BandedJointEncoder,\
                      length_scale=1, kernel_scales=1,latent_time_length=latent_TIME_LENGTH)
            checkpoint_vae = tf.train.Checkpoint(vae=vae)
            # load best model weights
            latest = tf.train.latest_checkpoint(checkpoint_dir_vae)
            print('latest checkpoint: {}'.format(latest))
            if latest is None:
                sys.exit("Model not found. Put the model in the correct path")
            status=checkpoint_vae.restore(latest)
            model = checkpoint_vae.vae
            
        elif model_type=='IB-Disjoint':
            checkpoint_dir_vae = MAIN_dir_fwd +  '/training_checkpoints/'
            vae = VAE(latent_dims, X_DATA_DIM,Y_DATA_DIM, TIME_LENGTH, beta=beta_vae)
            
            checkpoint_vae = tf.train.Checkpoint(vae=vae)
            # load best model weights
            latest = tf.train.latest_checkpoint(checkpoint_dir_vae)
            print('latest checkpoint: {}'.format(latest))
            if latest is None:
                sys.exit("Model not found. Put the model in the correct path")
            status=checkpoint_vae.restore(latest)
            model = checkpoint_vae.vae

        
        else:
            sys.exit('Model fsnot found in the folder')
            

#  load actor
        checkpoint_dir_actor =  MAIN_dir_actor + '/training_checkpoints/'
        actor= ML_actor(X_DATA_DIM,latent_dims_actor, TIME_LENGTH)            
        checkpoint_actor = tf.train.Checkpoint(actor=actor)
        # load best model weights
        latest = tf.train.latest_checkpoint(checkpoint_dir_actor)
        print('latest checkpoint: {}'.format(latest))
        status= checkpoint_actor.restore(latest)
        actor = checkpoint_actor.actor


    test_response_plot_actor(test_dataset_fw_main, model,actor,model_true)

def beta_traverse_ablation(model_type,beta_vals,mask_indx, 
                  X_DATA_DIM, Y_DATA_DIM, TIME_LENGTH,beta_vae,latent_TIME_LENGTH,latent_dims,test_dataset_fw_main):
    
    corr = []
    Ixz  = []
    Iyz  = []  
    corr_ablated = []
    main_dir = './models'
    
    for i in range(len(beta_vals)):
        
        if model_type=='IB-GP':
                checkpoint_dir_vae = main_dir + '/IB-GP/beta-traverse/' + str(int(i+1))+ '/training_checkpoints/'
            
                vae= GP_VAE(latent_dims, X_DATA_DIM,Y_DATA_DIM, TIME_LENGTH, beta=beta_vae, encoder=BandedJointEncoder,\
                          length_scale=1, kernel_scales=1,latent_time_length=latent_TIME_LENGTH)
        elif model_type=='IB-Disjoint':
                checkpoint_dir_vae = main_dir + '/IB-Disjoint/beta-traverse/' + str(int(i+1))+ '/training_checkpoints/'
                vae= VAE(latent_dims, X_DATA_DIM,Y_DATA_DIM, TIME_LENGTH, beta=beta_vae)
        else:
                sys.exit('Model-type not correct')
                
                
        checkpoint_vae = tf.train.Checkpoint(vae=vae)
        # load best model weights
        latest = tf.train.latest_checkpoint(checkpoint_dir_vae)
        print('latest checkpoint: {}'.format(latest))
        
        if latest is None:
            sys.exit("Model not found. Put the model in the correct path")
            
        status=checkpoint_vae.restore(latest)
        model_vae=checkpoint_vae.vae
        
        a, b, c = test_beta_traverse(test_dataset_fw_main, model_vae)
        corr.append(a)
        Ixz.append(b)
        Iyz.append(c)
        
        
        _, a = test_ablation(test_dataset_fw_main, model_vae,mask_indx[i])
        corr_ablated.append(a)


        
    fig, ax1 = plt.subplots()
    trans = transforms.blended_transform_factory(ax1.transData, ax1.transData)
    ax1.plot([round(1/x, 2) for x in beta_vals], corr, lw =2, color= 'tab:blue',label='Original')
    ax1.plot([round(1/x, 2) for x in beta_vals], corr_ablated, lw =2,color= 'tab:green', label='Ablated')
    ax1.set_xlabel(r"$\beta$")
    ax1.set_ylabel('Pearson correlation')
    ax1.set_xscale('log')
    text = [str(len(mask_indx[i])) for i in range(len(mask_indx))]
    
    for i in range(len(beta_vals)):
        x = 1/beta_vals[i]
        y = corr_ablated[i]
        text= str(int(model_vae.latent_dim-len(mask_indx[i])))
        
        if i<2:
            continue
        elif i>4:
            break
        
        ax1.text(x,y,text,transform=trans)
    plt.savefig('./figures/beta-var-corrs.png')
    tikzplotlib.save('./figures/beta-var-corrs.tex')
    

    fig, ax1 = plt.subplots()
    
    color = 'tab:blue'
    ax1.set_xlabel(r"$\beta$")
    ax1.set_yscale('log')
    ax1.set_ylabel('KL', color=color)
    ax1.plot([round(1/x, 2) for x in beta_vals], Ixz,lw =2, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:green'
    ax2.set_ylabel('Log-likelihood', color=color)  # we already handled the x-label with ax1
    ax2.plot([round(1/x, 2) for x in beta_vals], Iyz, lw =2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('./figures/beta-var-Ixyz.png')
    tikzplotlib.save('./figures/beta-var-Ixyz.tex')
    
    plt.figure()
    plt.plot(Ixz, [i for i  in Iyz])
    

def beta_traverse_emb(model_types,beta_vals,mask_indx, 
                  X_DATA_DIM, Y_DATA_DIM, TIME_LENGTH,beta_vae,latent_TIME_LENGTH,latent_dims,test_dataset_fw_main):

    main_dir = './models'

    for i in range(len(beta_vals)):
        models= []
        for j in range(len(model_types)):
            model_type = model_types[j]
            
            if model_type=='IB-GP':
                    checkpoint_dir_vae = main_dir + '/IB-GP/beta-traverse/' + str(int(i+1))+ '/training_checkpoints/'
                
                    vae= GP_VAE(latent_dims, X_DATA_DIM,Y_DATA_DIM, TIME_LENGTH, beta=beta_vae, encoder=BandedJointEncoder,\
                              length_scale=1, kernel_scales=1,latent_time_length=latent_TIME_LENGTH)
            elif model_type=='IB-Disjoint':
                    checkpoint_dir_vae = main_dir + '/IB-Disjoint/beta-traverse/' + str(int(i+1))+ '/training_checkpoints/'
                    vae= VAE(latent_dims, X_DATA_DIM,Y_DATA_DIM, TIME_LENGTH, beta=beta_vae)
            else:
                    sys.exit('Model-type not correct')
                    
                    
            checkpoint_vae = tf.train.Checkpoint(vae=vae)
            # load best model weights
            latest = tf.train.latest_checkpoint(checkpoint_dir_vae)
            print('latest checkpoint: {}'.format(latest))
            
            if latest is None:
                sys.exit("Model not found. Put the model in the correct path")
            
            status=checkpoint_vae.restore(latest)
            model_vae=checkpoint_vae.vae
            models.append(model_vae)
    emb = test_beta_traverse_emb(test_dataset_fw_main, models)


        

    
