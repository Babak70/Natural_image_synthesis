from matplotlib import pyplot as plt
import tensorflow as tf
from dim_reduction_utils import pca_transform, tSNE_transform,single_img_svd
import numpy as np
import imageio
from metrics import fraction_of_explained_variance,correlation_coefficient
import tensorflow.keras.backend as K
import os
import gc
from scipy import stats
import tikzplotlib 
from numpy.random import default_rng
from sklearn.decomposition import PCA
from scipy import signal
import matplotlib.transforms as mtransforms
from decimal import Decimal
from Generative_Models import VAE,GP_VAE
import time
import matplotlib
def save_img(stimulii,names,direct):
    
  plt.rcParams.update({'font.size':20})
  for i in range(stimulii[0].shape[0]):
      
      fig,axs=plt.subplots(1, len(names),figsize=(10, 10))
      
      for j in range(len(names)):
          
          stimuli=stimulii[j]
          name=names[j]

          axs[j].imshow(stimuli[i,:,:])
          axs[j].title.set_text(name)
          axs[j].axis('off')
          fig.savefig( direct+'image {}.png'.format(i+1))
          plt.close('all')


def plot_density(samples, xlim, name):
    
      bins = np.linspace(-xlim, xlim, 30)
      
      histogram, bins = np.histogram(samples, bins=bins, density=True)
      bin_centers = 0.5*(bins[1:] + bins[:-1])

      plt.figure(figsize=(6, 4))
      plt.plot(bin_centers, histogram, label=name)
      # pdf = stats.norm.pdf(bin_centers)
      # plt.plot(bin_centers, pdf, label="PDF-average")
      plt.legend()
      plt.show()
    
    
    
    
def make_anim(anim_file, direct, num_imgs):
    

        with imageio.get_writer(anim_file,mode='I') as writer:
            filenames = [direct +'image {}.png'.format(i+1) for i in range(num_imgs)]

            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
                image = imageio.imread(filename)
                writer.append_data(image)

        # embed.embed_file(anim_file)

def test_ablation(test_ds,model,mask_indx):
                     
    running_corr_ablated = 0
    running_corr_orig = 0
    
    running_size = 0


    for n, (stimuli, spikes) in test_ds.enumerate():
               
               stimuli=tf.transpose(stimuli, perm=[1,0,2])
               obs_rate = spikes

               est_rate_orig, est_rate_ablated, tsnf_stimuli_orig, tsnf_stimuli_ablated = model.ablation(stimuli, obs_rate, mask_indx)
               
               corr_orig = K.mean(correlation_coefficient(tf.squeeze(obs_rate),tf.squeeze(est_rate_orig)))
               
               corr_ablated = K.mean(correlation_coefficient(tf.squeeze(obs_rate),tf.squeeze(est_rate_ablated)))


               running_corr_orig += corr_orig.numpy()
               running_corr_ablated += corr_ablated.numpy()
               running_size += 1
               

    

    running_corr_orig /= running_size
    running_corr_ablated /= running_size

    return corr_orig, corr_ablated



def test_cumulative (test_ds,model,mask_indx):
    
    if os.path.isdir('./figures')==False:
        os.mkdir('./figures')
    else:
        if os.path.isdir('./figures/cumulative')==False:
            os.mkdir('./figures/cumulative')

    k_fold = 3
    corr_orig_avg = np.ndarray((k_fold,1))
    corr_keep_avg = np.ndarray((k_fold,model.latent_dim))
    nll_orig_avg = np.ndarray((k_fold,1))
    nll_keep_avg = np.ndarray((k_fold,model.latent_dim))

    
    
    for i in range(k_fold):
        corr_orig_temp = []
        corr_keep_temp = []
        nll_orig_temp = []
        nll_keep_temp = []
        
        for n, (stimuli, spikes) in test_ds.enumerate():
            stimuli=tf.transpose(stimuli,perm=[1,0,2])
            obs_rate=spikes
            corr_orig,corr_keep, nll_orig, nll_keep = model.cumulative_latent(stimuli, obs_rate,batch_num=n.numpy())
            corr_orig_temp.append(corr_orig)
            corr_keep_temp.append(corr_keep)
            
            nll_orig_temp.append(nll_orig)
            nll_keep_temp.append(nll_keep)
            
        corr_orig_temp = np.array(corr_orig_temp)
        corr_keep_temp = np.array(corr_keep_temp)
        nll_orig_temp = np.array(nll_orig_temp)
        nll_keep_temp = np.array(nll_keep_temp)
        
        corr_orig_avg[i,:] = np.mean(corr_orig_temp,axis=0)
        corr_keep_avg[i,:] = np.mean(corr_keep_temp,axis=0)
        nll_orig_avg[i,:] = np.mean(nll_orig_temp,axis=0)
        nll_keep_avg[i,:] = np.mean(nll_keep_temp,axis=0)
        
    Corr_orig_avg = np.mean(corr_orig_avg)
    Corr_orig_std = np.std (corr_orig_avg)
    Corr_keep_avg = np.mean (corr_keep_avg,axis=0)
    Corr_keep_std = np.std (corr_keep_avg,axis=0)

    NLL_orig_avg = np.mean(nll_orig_avg)
    NLL_orig_std = np.std (nll_orig_avg)
    NLL_keep_avg = np.mean (nll_keep_avg,axis=0)
    NLL_keep_std = np.std (nll_keep_avg,axis=0)
    
    return Corr_orig_avg,Corr_orig_std,Corr_keep_avg,Corr_keep_std, NLL_orig_avg,NLL_orig_std,NLL_keep_avg,NLL_keep_std


def test_traverse (test_ds,model,mask_indx):
    
    if os.path.isdir('./figures')==False:
        os.mkdir('./figures')
    else:
        if os.path.isdir('./figures/traverse')==False:
            os.mkdir('./figures/traverse')
            os.mkdir('./figures/traverse/batch-0')
            os.mkdir('./figures/traverse/batch-1')
            os.mkdir('./figures/traverse/batch-2')
            os.mkdir('./figures/traverse/batch-3')
            os.mkdir('./figures/traverse/batch-4')
                     
    corr_orig_avg = []
    corr_std_avg = []
    kl_traverse_avg  = []

    for n, (stimuli, spikes) in test_ds.enumerate():
        # if n.numpy()==1:
        #     break
        #     1
               
        stimuli=tf.transpose(stimuli,perm=[1,0,2])
        obs_rate=spikes

        corr_traverse, corr_orig, kl_traverse = model.traverse(stimuli, obs_rate,batch_num=n.numpy())
        
        plt.figure()
        for i in range(7):
            plt.plot(range(corr_traverse.shape[1]), corr_traverse[i,:],'*')
        plt.plot(range(corr_traverse.shape[1]), [corr_orig]*corr_traverse.shape[1])
        plt.title('batch {} corrs'.format(n.numpy()))
        plt.savefig('./figures/traverse/batch-{}/corr-all.png'.format(n.numpy()))
        
        corr_traverse_std = np.sqrt(np.sum(np.square(corr_traverse-corr_orig),axis=0)/7)
        corr_std_avg.append(corr_traverse_std)
        corr_orig_avg.append(corr_orig)
        kl_traverse_avg.append(kl_traverse)
        
        plt.figure()
        plt.plot(range(corr_traverse.shape[1]),corr_traverse_std)
        plt.plot(range(corr_traverse.shape[1]), [corr_orig]*corr_traverse.shape[1])
        plt.title('batch {} corr_std'.format(n.numpy()))
        plt.savefig('./figures/traverse/batch-{}/corr-std_all.png'.format(n.numpy()))
        
        make_anim('./figures/traverse/batch {} traverse.gif'.format(n.numpy()),'./figures/traverse/batch-{}/'.format(n.numpy()),7*model.latent_dim)
        
    plt.figure()
    plt.plot(range(corr_traverse.shape[1]),sum(corr_std_avg)/len(corr_std_avg))
    plt.plot(range(corr_traverse.shape[1]), [(sum(corr_orig_avg)/len(corr_orig_avg))]*corr_traverse.shape[1])
    plt.title('corr_std_avg')
    plt.savefig('./figures/traverse/corr-std_avg.png'.format(n.numpy()))
    
    
    kl_traverse_avg = np.array(kl_traverse_avg)
    if isinstance(model, VAE):
        kl_traverse_avg /= (model.time_length*model.latent_dim)
    else:
        kl_traverse_avg /= (2*model.time_length*model.latent_dim)
    corr_std_avg = np.array(corr_std_avg)
    corr_orig_avg = np.array(corr_orig_avg)
    
    fig, ax1 = plt.subplots()
    x = [_ for _ in range(corr_traverse.shape[1])]
    
   
    y = corr_orig_avg.mean(axis=0)
    # y_l = y-corr_orig_avg.std(axis=0)
    # y_u = y+corr_orig_avg.std(axis=0)
    ax1.plot(x, [y]*corr_traverse.shape[1],lw=3, color ='#0000FF')
    # ax1.fill_between(x, y_l , y_u, facecolor='#C5C9C7', alpha = 0.1)
    y_std = corr_std_avg.mean(axis=0)
    y_l = y-y_std*1.
    y_u = y+y_std*0.0
    # ax1.plot(x,y, lw =2, color = '#0000FF')
    ax1.fill_between(x, y_l , y_u, facecolor='#0343DF', alpha =0.2)
    ax1.set_ylabel('Pearson correlation', color='#0000FF')
    ax1.set_xlabel('i-th latent')
    ax1.tick_params(axis='y', labelcolor='#0000FF')
    
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    y =    kl_traverse_avg.mean(axis=0)
    y_l = y-kl_traverse_avg.std(axis=0)
    y_u = y+kl_traverse_avg.std(axis=0)
    ax2.plot(x,y, lw=2, color='#008000')
    ax2.fill_between(x, y_l , y_u, facecolor='#15B01A', alpha =0.3)
    ax2.set_ylabel('KL (normalized)', color='#008000')  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor='#008000')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.legend(frameon=False, fontsize=9)
    ax2.legend(frameon=False, fontsize=9)
    plt.savefig('./figures/traverse/KL-corr-std_all.png')
    tikzplotlib.save('./figures/traverse/KL-corr-std_all.tex')

    
    return 0

def test_response_plot (test_ds,models,model_types):
    neuron_number=0
    line_types = ['-','--','-.']
    line_colors = ['b', 'g', 'y']
    time_len = 100
    
    if os.path.isdir('./figures')==False:
        os.mkdir('./figures')
    else:
        if os.path.isdir('./figures/responses')==False:
            os.mkdir('./figures/responses')
            os.mkdir('./figures/responses/batch-0')
            os.mkdir('./figures/responses/batch-1')
            os.mkdir('./figures/responses/batch-2')
            os.mkdir('./figures/responses/batch-3')
            os.mkdir('./figures/responses/batch-4')
                     

    for n, (stimuli, spikes) in test_ds.enumerate():
      stimuli = tf.transpose(stimuli,perm=[1,0,2])
      obs_rate = spikes

      for j in range(len(models)):
        model = models[j]
        if model_types[j]=='feedforwardCNN':
            
            est_rate = model.forward(stimuli,obs_rate)
        else:
            est_rate,_,_ = model.forward(stimuli,obs_rate)
            
        # x = np.arange(0, model.time_length/100, 0.01)
        # plt.figure()
        # plt.plot(x,tf.squeeze(obs_rate)[:,neuron_number].numpy(), '-k', lw=3, label="true rate")
        # plt.plot(x,tf.squeeze(est_rate)[:,neuron_number].numpy(), line_colors[j] , linestyle= line_types[j], lw=2, label=model_types[j])
        # plt.legend()
        # plt.ylabel('Rate(spikes/sec)')
        # plt.xlabel('time(sec)')
        # plt.savefig('./figures/responses/batch-{}/response-{}.png'.format(n.numpy(),model_types[j]))
        # tikzplotlib.save('./figures/responses/batch-{}/response-{}.tex'.format(n.numpy(),model_types[j]))
        # plt.close()
        
        fig, axs = plt.subplots(3, 3)
        x = np.arange(0, model.time_length/100, 0.01)
        for i, ax in enumerate(axs.flat):
                
                slice_i = slice((i)*time_len,(i+1)*time_len)
                ax.plot(x[slice_i], tf.squeeze(obs_rate)[:,neuron_number].numpy()[slice_i], lw=2, label="true rate", color ='#808080')
                ax.plot(x[slice_i], tf.squeeze(est_rate)[:,neuron_number].numpy()[slice_i], lw=2, label="true rate", color ='#008000',alpha=0.85)
                corr = K.mean(correlation_coefficient(tf.squeeze(obs_rate)[:,neuron_number][slice_i],tf.squeeze(est_rate)[:,neuron_number][slice_i]))
                trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
                ax.text(0.3, 1.0, str(round(corr.numpy(),2)), transform=ax.transAxes + trans,fontsize='medium', verticalalignment='top')
                # x0,x1 = ax.get_xlim()
                # y0,y1 = ax.get_ylim()
                # ax.set_aspect((x1-x0)/(y1-y0))
                ax.tick_params(axis='x', which='major', labelsize=10)
                ax.tick_params(axis='y', which='major', labelsize=10)
                ax.set_ylim((0.00, 0.7))
                if i not  in [0,3,6]:
                    ax.set_yticklabels([])
                # if i not in [6,7,8]:
                #     ax.set_xticklabels([])
        fig.text(0.5, -0.001, 'Time (s)', ha='center')
        fig.text(-0.001, 0.5, 'Rate (spikes/s)', va='center', rotation='vertical')
        
        # fig.tight_layout()
        # plt.show()
        fig.savefig('./figures/responses/batch-{}/response-{}.eps'.format(n.numpy(),model_types[j]),format='eps')
        fig.savefig('./figures/responses/batch-{}/response-{}.png'.format(n.numpy(),model_types[j]))
        tikzplotlib.save('./figures/responses/batch-{}/response-{}.tex'.format(n.numpy(),model_types[j]),figure=fig)
        plt.close(fig)
        

      

    return 0

def test_response_plot_actor (test_ds,model,actor,model_true):
    neuron_number=0
    line_types = ['-','-','.']
    line_colors = ['b', 'g', 'y']
    
    if os.path.isdir('./figures')==False:
        os.mkdir('./figures')
    else:
        if os.path.isdir('./figures/responses')==False:
            os.mkdir('./figures/responses')
            os.mkdir('./figures/responses/batch-0')
            os.mkdir('./figures/responses/batch-1')
            os.mkdir('./figures/responses/batch-2')
            os.mkdir('./figures/responses/batch-3')
            os.mkdir('./figures/responses/batch-4')
                     
    corr_orig_avg = []
    corr_std_avg = []
    for n, (stimuli, spikes) in test_ds.enumerate():
      stimuli = tf.transpose(stimuli,perm=[1,0,2])
      obs_rate = spikes

      corrs = []

      est_rate_true,_,_ = model_true.forward(stimuli,obs_rate)
      optimized_stimuli = actor.forward(stimuli)
      est_rate,_,_ = model.forward(tf.reshape(optimized_stimuli,[model.time_length,-1,model.x_data_dim]),obs_rate)
      
      corr_true = K.mean(correlation_coefficient(tf.squeeze(obs_rate),tf.squeeze(est_rate_true)))
      corr = K.mean(correlation_coefficient(tf.squeeze(obs_rate),tf.squeeze(est_rate)))
      corrs.append(corr)
      
      plt.figure()
      plt.plot(np.arange(0, 10, 0.01),tf.squeeze(obs_rate)[:,neuron_number].numpy(), '-k', lw=3, label="true rate")
      plt.plot(np.arange(0, 10, 0.01),tf.squeeze(est_rate_true)[:,neuron_number].numpy(), 'b-' , lw=2, label='High-res. response, corr: {:.2f}'.format(round(corr_true.numpy(),2)))
      plt.plot(np.arange(0, 10, 0.01),tf.squeeze(est_rate)[:,neuron_number].numpy(), 'g--' , lw=2, label='Synthesized response, corr: {:.2f}'.format(round(corr.numpy(),2)))
      plt.legend()
      plt.ylabel('Rate(spikes/sec)')
      plt.xlabel('time(sec)')
      plt.savefig('./figures/responses/batch-{}/response.png'.format(n.numpy()))
      tikzplotlib.save('./figures/responses/batch-{}.tex'.format(n.numpy()))
      

    return 0

def test_beta_traverse  (test_ds,model):
    
    if os.path.isdir('./figures')==False:
        os.mkdir('./figures')

                     
    running_corr = 0
    running_Ixz = 0
    running_Iyz = 0
    running_size = 0

    for n, (stimuli, spikes) in test_ds.enumerate():
               
               stimuli=tf.transpose(stimuli, perm=[1,0,2])
               obs_rate = spikes

               Iyz,Ixz, est_rate = model.mutual_info(stimuli, obs_rate)
               corr = K.mean(correlation_coefficient(tf.squeeze(obs_rate),tf.squeeze(est_rate)))
               

               running_corr += corr.numpy()
               running_Ixz += Ixz.numpy()
               running_Iyz += Iyz.numpy()
               running_size += 1
               

    running_corr /= running_size
    running_Ixz /= running_size
    running_Iyz /= running_size

    return  running_corr, running_Ixz, running_Iyz
    
def emb_and_plot(z, numbers, emb_type,early_exaggeration=12, perplexity=30, marker='.',alpha=0.4, color = 'blue'):
    
    
   if emb_type =="PCA":
           pca=PCA(2).fit(z) #learning the manifold
           emb_i=pca.transform(z) # projection of high res samples onto the low-dim basis vectors
   else:
           emb_i = tSNE_transform(z,n_components=2,early_exaggeration=early_exaggeration, perplexity=perplexity)
           
   plt.scatter(emb_i[numbers,0],emb_i[numbers,1],edgecolors=None, marker=marker,alpha=alpha, color =color)

   return emb_i
def auto_corr_f(xsignal, auto_corr_type, cell_num=None, do_resample=True):
       if cell_num==None:
           ysignal = np.mean(np.squeeze(xsignal.numpy()),axis=1)
       else:
           ysignal = np.squeeze(xsignal.numpy())[:,cell_num]
           
       if do_resample:
           
           ysignal =signal.resample(ysignal,500)

       return np.array(signal.correlate(ysignal,ysignal,auto_corr_type))
       
    
def test_beta_traverse_emb (test_ds,models, task="auto-correlation", return_corr=False):
    
    if os.path.isdir('./figures')==False:
        os.mkdir('./figures')
    model_1=models[0]
    model_2=[]
    if len(models)>1:
        model_2=models[1]


    if task == "embedding":
        
        
        intvl_TSNE=50
        early_exaggeration = 25
        perplexity = 25
        rng = default_rng(seed=1)
        n_samples = 100
        numbers = rng.choice(n_samples, size=100, replace=False)
        
        if os.path.isdir('./figures/emb')==False:
            os.mkdir('./figures/emb')
            
        Z_IB_GP =   np.empty([n_samples,model_1.latent_time_length,model_1.latent_dim])
        Z_IB_Disjoint = np.empty([n_samples,model_1.latent_time_length,model_1.latent_dim])

        for n, (stimuli, spikes) in test_ds.enumerate():
          stimuli=tf.transpose(stimuli, perm=[1,0,2])
          obs_rate = spikes
    
          if n.numpy()==0:
            # IB_GP model
            for j in range(n_samples):
                            __, z, _ = model_1.forward(stimuli)
                            z = tf.squeeze(z)
                            z = tf.transpose(z)
                            Z_IB_GP[j,:,:]=z
                            print(j)
                            __, z_2, _ = model_2.forward(stimuli)
                            z_2 = tf.squeeze(z_2)
                            Z_IB_Disjoint[j,:,:] =z_2


            plt.figure()
            emb_and_plot(tf.reshape(np.take(Z_IB_GP,[3],axis=2),[-1,1*n_samples]), numbers,"TSNE",early_exaggeration=early_exaggeration,
                                                                     perplexity=perplexity,marker='o', color = '#008000')
            plt.xlim([-intvl_TSNE,intvl_TSNE])
            plt.ylim([-intvl_TSNE,intvl_TSNE])
            plt.xlabel('TSNE dimension 1')
            plt.ylabel('TSNE dimension 2')
            plt.savefig('./figures/emb/IB-GP-informative-latent.png')
            tikzplotlib.save('./figures/emb/IB-GP-informative-latent.tex')
            # plt.show()
            
            plt.figure()
            emb_and_plot(tf.reshape(np.take(Z_IB_Disjoint,[0],axis=2),[-1,1*n_samples]), numbers,"TSNE",early_exaggeration=early_exaggeration,
                                                                     perplexity=perplexity,marker='o', color = '#008000')
            plt.xlim([-intvl_TSNE,intvl_TSNE])
            plt.ylim([-intvl_TSNE,intvl_TSNE])
            plt.xlabel('TSNE dimension 1')
            plt.ylabel('TSNE dimension 2')
            plt.savefig('./figures/emb/IB-Disjoint-informative-latent.png')
            tikzplotlib.save('./figures/emb/IB-Disjoint-informative-latent.tex')
            # plt.show()

    elif task == "covariance_matrix":
        if os.path.isdir('./figures/covariance_matrix')==False:
            os.mkdir('./figures/covariance_matrix')
        model=model_1
        latent_indx =[13,11,2]
        for n, (stimuli, spikes) in test_ds.enumerate():
            if n.numpy() != 0:
                            continue
            else:
                             
                         stimuli=tf.transpose(stimuli, perm=[1,0,2])
                         obs_rate = spikes
            
                         est_rate, z, loc, scale = model.forward(stimuli, return_param = True)
                         z = tf.squeeze(z)
                         loc = tf.squeeze(loc)
                         scale = tf.squeeze(scale)
                         for j in range(len(latent_indx)):
                             scale_i = scale[latent_indx[j],:,:].numpy()
                             scale_i[scale_i<0] = np.finfo(float).eps
                             cov_matrix= scale_i @ np.transpose(scale_i)
                             matplotlib.rc('image', cmap='seismic')
                             plt.figure()
                             # plt.imshow(np.log10(np.tril(scale_i + np.finfo(float).eps)),extent=[0,10,10,0])
                             plt.imshow(np.log10(cov_matrix + np.finfo(float).eps),extent=[0,10,10,0])
                             if j==0: plt.title('Most informative latent')
                             elif j==1:plt.title('Second most informative latent')
                             elif j==2:plt.title('Least informative latent')
                             # if j ==(len(latent_indx)-1):
                             #     plt.colorbar()
                             plt.savefig('./figures/covariance_matrix/covariance_matrix-{}.eps'.format(j),format='eps')
                             plt.savefig('./figures/covariance_matrix/covariance_matrix-{}.png'.format(j))
                             plt.show()


    elif task == "auto-correlation":
        
        model = model_1
        auto_corr_type = "same"
        if os.path.isdir('./figures/auto_correlation')==False:
            os.mkdir('./figures/auto_correlation')
        latent_num_IB_GP = [13]
        latent_num_IB_Disjoint = [1]
        latent_num = latent_num_IB_Disjoint
        auto_corr_latent_all = []
        auto_corr_data_all = []
        auto_corr_model_all = []
        
        auto_corr_cells_data_all = []
        auto_corr_cells_model_all = []
        for n, (stimuli, spikes) in test_ds.enumerate():
            if n.numpy() != 0:
                          continue
            else:
                         stimuli=tf.transpose(stimuli, perm=[1,0,2])
                         
                         
                         est_rate, z, loc, scale = model.forward(stimuli, return_param = True)
                         z = tf.squeeze(z)
                         loc = tf.squeeze(loc)
                         scale = tf.squeeze(scale)
                         
                         if z.shape[1]>z.shape[0]:
                            z = tf.transpose(z)
                            loc = tf.transpose(loc)
                            scale = tf.transpose(scale)
                            latent_num = latent_num_IB_GP
                         # Latent
                         z_signal = np.squeeze(z.numpy()[:,latent_num])
                         auto_corr_latent = signal.correlate(z_signal,z_signal,auto_corr_type)
                         auto_corr_latent = auto_corr_latent/np.max(auto_corr_latent)
                         auto_corr_latent_all.append(auto_corr_latent)

                         # Model
                         auto_corr = auto_corr_f(est_rate, auto_corr_type, cell_num=None, do_resample=True)
                         auto_corr = auto_corr/np.max(auto_corr)
                         auto_corr_model_all.append(auto_corr)

                         # Groundtruth
                         auto_corr = auto_corr_f(spikes, auto_corr_type, cell_num=None, do_resample=True)
                         auto_corr = auto_corr/np.max(auto_corr)
                         auto_corr_data_all.append(auto_corr)


                         auto_corr_cells_data =[None]*model.y_data_dim
                         auto_corr_cells_model =[None]*model.y_data_dim
                         
                         for k in range(model.y_data_dim):

                              # plt.plot(auto_corr_latent)
                              auto_corr = auto_corr_f(est_rate,auto_corr_type, cell_num=k, do_resample=True)
                              auto_corr = (auto_corr)/np.max((auto_corr))
                              auto_corr_cells_model [k] = auto_corr

                             
                              auto_corr = auto_corr_f(spikes, auto_corr_type, cell_num=k, do_resample=True)
                              auto_corr = (auto_corr)/np.max((auto_corr))
                              auto_corr_cells_data [k] = auto_corr

                              plt.title(k)

                         auto_corr_cells_data_all.append(auto_corr_cells_data)
                         auto_corr_cells_model_all.append(auto_corr_cells_model)
    
     
        auto_corr_latent_all =  np.array(auto_corr_latent_all)
        auto_corr_data_all =  np.array(auto_corr_data_all)
        auto_corr_model_all =  np.array(auto_corr_model_all)
        
        auto_corr_cells_data_all =  np.array(auto_corr_cells_data_all)
        auto_corr_cells_model_all =  np.array(auto_corr_cells_model_all)
    
    
        plt.figure()
        x = [_/100 for _ in range(-len(auto_corr)//2,(len(auto_corr)//2))]

        y = np.mean(auto_corr_latent_all,axis=0)
        plt.plot(x, y, linewidth=1.5,label="Latent", color = "#0000FF", alpha = 0.7)
        
        y = np.mean(auto_corr_model_all,axis=0)
        plt.plot(x, y,linewidth=1.5,label="Model response",color = "#008000")
        y_l = y-np.std(auto_corr_model_all,axis=0)
        y_u = y+np.std(auto_corr_model_all,axis=0)
        plt.fill_between(x, y_l, y_u,facecolor='#15B01A', alpha=0.4)
        
        y = np.mean(auto_corr_data_all,axis=0)
        plt.plot(x,y, linewidth=1.5 ,label="RGC response", color = "#808080")
        y_l = y-np.std(auto_corr_data_all,axis=0)
        y_u = y+np.std(auto_corr_data_all,axis=0)
        plt.fill_between(x, y_l, y_u,facecolor='#C5C9C7', alpha=0.4)
        plt.xlabel("Time lag (s)")
        plt.ylabel("Autocorrelation")
        plt.legend(frameon=False, fontsize=9)
        plt.savefig('./figures/auto_correlation/auto-corr-avg-cells.png')
        tikzplotlib.save('./figures/auto_correlation/auto-corr-avg-cells.tex')
        for k in range(model.y_data_dim):
            plt.figure()
            y = np.mean(auto_corr_latent_all,axis=0)
            plt.plot(x,y,linewidth=1.5,label="Latent", color = "#0000FF", alpha = 0.9)

            
            y = np.mean(auto_corr_cells_model_all,axis=0)[k,:]
            plt.plot(x,y,linewidth=1.5,label="Model response", color = "#008000")
            
            y_l = y-np.std(auto_corr_cells_model_all,axis=0)[k,:]
            y_u = y+np.std(auto_corr_cells_model_all,axis=0)[k,:]
            plt.fill_between(x, y_l, y_u,facecolor='#15B01A', alpha=0.4)
            
            y = np.mean(auto_corr_cells_data_all,axis=0)[k,:]
            plt.plot(x,y,linewidth=1.5,label="RGC response", color= "#808080")
            
            y_l = y-np.std(auto_corr_cells_data_all,axis=0)[k,:]
            y_u = y+np.std(auto_corr_cells_data_all,axis=0)[k,:]
            plt.fill_between(x, y_l, y_u,facecolor ='#C5C9C7', alpha=0.4)
            plt.title(k)
            plt.xlabel("Time lag (s)")
            plt.ylabel("Autocorrelation")
            plt.legend(frameon=False, fontsize=9)
            plt.savefig('./figures/auto_correlation/auto-corr-cell-{}.png'.format(k))
            tikzplotlib.save('./figures/auto_correlation/auto-corr-cell-{}.tex'.format(k))
            
            
            
    elif task == "param_density":
                         model=model_2
                         if os.path.isdir('./figures/param_density')==False:
                             os.mkdir('./figures/param_density')
                         est_rate, z, loc, scale = model.forward(stimuli, return_param = True)
                         z = tf.squeeze(z)
                         loc = tf.squeeze(loc)
                         scale = tf.squeeze(scale)

                         if z.shape[1]>z.shape[0]:
                            z = tf.transpose(z)
                            loc = tf.transpose(loc)
                            scale = tf.transpose(scale)
 
                         latent_num = [0,12]
                         xlim = 5
                         samples_mu = np.mean(np.squeeze(loc.numpy())[:,latent_num],axis=1)
                         plot_density(samples_mu, xlim, "mu")
                         xlim = 2
                         samples_logvar = np.mean(2*np.log10(np.squeeze(scale.numpy()))[:,latent_num],axis=1)
                         plot_density(samples_logvar, xlim, "logvar")
                         
    else:
         print("Task not found")
         exit()
                         


def test_forward_all(test_ds,model_fwd,model=None,model_decoder=None, task=None,mask=None):
                     
    running_corr_tnsf = 0
    running_loss_tnsf=0
    
    running_corr_fwd = 0
    running_loss_fwd=0
    
    running_corr_tnsf_fwd = 0
    running_loss_tnsf_fwd=0
    
    running_corr_pca_fwd = 0
    running_loss_pca_fwd=0
    
    running_size = 0
    
    direct_main='./Reconstructed-stimuli/'
    if os.path.isdir('./Reconstructed-stimuli')==False:
        os.mkdir('./Reconstructed-stimuli')

    if task=='IB-GP':

       for n, (stimuli, spikes) in test_ds.enumerate():
    
        
            obs_rate = spikes
            stimuli=tf.transpose(stimuli,perm=[1,0,2])
            
            loss_tnsf_t,  px_z_tnsf, z, stimuli_tnsf = model.forward(stimuli,obs_rate, return_loss=True, mask = mask)
            
            # z = tf.transpose(z, perm=[0,2,1])
            # stimuli_tnsf =  model_decoder(z)

                
            stimuli_tnsf = tf.reshape(stimuli_tnsf,[model.time_length, -1 ,model.x_data_dim])
            loss_tnsf_fwd_t, px_z_tnsf_fwd  = model_fwd.forward(stimuli_tnsf, obs_rate, return_loss=True)
            corr_tnsf = K.mean(correlation_coefficient(tf.squeeze(obs_rate), tf.squeeze(px_z_tnsf)))
            corr_tnsf_fwd = K.mean(correlation_coefficient(tf.squeeze(obs_rate), tf.squeeze(px_z_tnsf_fwd)))
    
            running_loss_tnsf += loss_tnsf_t.numpy()
            running_loss_tnsf_fwd += loss_tnsf_fwd_t.numpy()
    
            running_corr_tnsf += corr_tnsf.numpy()
            running_corr_tnsf_fwd += corr_tnsf_fwd.numpy()
            running_size += 1
            # if mask == None:
            #     if n.numpy()==0:
            #         stimulii=[tf.reshape(tf.squeeze(stimuli),[-1, 50,50])]
            #         stimulii.append(tf.reshape(tf.squeeze(stimuli_tnsf),[-1, 50,50]))
            #         names=['Orig-Stimuli', 'IB']
            #         write_dirs='IB-GP'
            #         if os.path.isdir(direct_main + write_dirs)==False:
            #             os.mkdir(direct_main + write_dirs)
            #             os.mkdir(direct_main + write_dirs +'/images')
            #         direct= direct_main + write_dirs + '/images/'
            #         save_img(stimulii,names,direct)
            #         gc.collect
                    
            #     else:
            #         continue
            
            
            
       running_corr_tnsf /= running_size
       running_corr_tnsf_fwd /= running_size
       running_loss_tnsf /=running_size
       running_loss_tnsf_fwd /=running_size
       return  running_loss_tnsf,running_corr_tnsf,running_loss_tnsf_fwd,running_corr_tnsf_fwd

    if task=='IB-Disjoint':

       for n, (stimuli, spikes) in test_ds.enumerate():
        
        
            obs_rate = spikes
            stimuli=tf.transpose(stimuli,perm=[1,0,2])
            
            loss_tnsf_t,  px_z_tnsf, z, stimuli_tnsf = model.forward(stimuli,obs_rate, return_loss=True,mask=mask)
            
            # stimuli_tnsf = model_decoder(z)

            stimuli_tnsf = tf.reshape(stimuli_tnsf,[model.time_length, -1 ,model.x_data_dim])
                   
            loss_tnsf_fwd_t, px_z_tnsf_fwd  = model_fwd.forward(stimuli_tnsf, obs_rate, return_loss=True)
            corr_tnsf = K.mean(correlation_coefficient(tf.squeeze(obs_rate), tf.squeeze(px_z_tnsf)))
            corr_tnsf_fwd = K.mean(correlation_coefficient(tf.squeeze(obs_rate), tf.squeeze(px_z_tnsf_fwd)))
    
            running_loss_tnsf += loss_tnsf_t.numpy()
            running_loss_tnsf_fwd += loss_tnsf_fwd_t.numpy()
    
            running_corr_tnsf += corr_tnsf.numpy()
            running_corr_tnsf_fwd += corr_tnsf_fwd.numpy()
            running_size += 1
            
            # if n.numpy()==0:
            #     stimulii=[tf.reshape(tf.squeeze(stimuli),[-1, 50,50])]
            #     stimulii.append(tf.reshape(tf.squeeze(stimuli_tnsf),[-1, 50,50]))
            #     names=['Orig-Stimuli', 'IB']
            #     write_dirs='IB-GP'
            #     if os.path.isdir(direct_main + write_dirs)==False:
            #         os.mkdir(direct_main + write_dirs)
            #         os.mkdir(direct_main + write_dirs +'/images')
            #     direct= direct_main + write_dirs + '/images/'
            #     save_img(stimulii,names,direct)
            #     gc.collect
                
            # else:
            #     continue
            
            
            
       running_corr_tnsf /= running_size
       running_corr_tnsf_fwd /= running_size
       running_loss_tnsf /=running_size
       running_loss_tnsf_fwd /=running_size
       return  running_loss_tnsf,running_corr_tnsf,running_loss_tnsf_fwd,running_corr_tnsf_fwd
   
    elif task=='forward-model':
        for n, (stimuli, spikes) in test_ds.enumerate():
            
               obs_rate = spikes
               stimuli=tf.transpose(stimuli,perm=[1,0,2])
               loss_fwd_t, px_z_fwd  = model_fwd.forward(stimuli, obs_rate, return_loss=True)
               corr_fwd = K.mean(correlation_coefficient(tf.squeeze(obs_rate), tf.squeeze(px_z_fwd)))
               running_corr_fwd += corr_fwd.numpy()               
               running_loss_fwd += loss_fwd_t.numpy()               
               running_size += 1
               
        running_corr_fwd /= running_size
        running_loss_fwd /=running_size
        return running_loss_fwd,running_corr_fwd
        
        
    elif task=='PCA':
        all_stimuli = []
        for n, (stimuli, _) in test_ds.enumerate():

               stimuli = tf.transpose(stimuli,perm=[1,0,2])
               all_stimuli.append(stimuli)
               
        all_stimuli_concat = tf.concat(all_stimuli,axis=0)
        
        # all_images = []
        # all_images  = np.ndarray((5000,2500))
        # for ii in range(0,len(all_stimuli_concat),50):
        #     print(ii)
        #     img_rec = pca_transform(tf.squeeze(all_stimuli_concat[ii:(ii+50),:]).numpy(), 1)
        #     if ii % 100 == 0:
        #         plt.figure()
        #         plt.imshow(np.reshape(img_rec[0,...],[50,50]))
        #         plt.show()
        #     all_images[ii:(ii+50),:] = img_rec
        # all_stimuli_pca = (np.squeeze(all_images))
        # all_stimuli_pca=tf.squeeze(tf.convert_to_tensor(all_stimuli_pca))
        
        # all_images = []
        # all_images  = np.ndarray((5000,2500))
        # for ii in range(0,len(all_stimuli_concat),1):
        #     print(ii)
        #     img_rec = single_img_svd(np.reshape(tf.squeeze(all_stimuli_concat[ii:(ii+1),:]).numpy(),[50,50]), 1)
        #     img_rec = np.reshape(img_rec,[1,2500])
        #     if ii % 100 == 0:
        #         plt.figure()
        #         plt.imshow(np.reshape(img_rec[0,...],[50,50]))
        #         plt.show()
        #     all_images[ii:(ii+1),:] = img_rec
        # all_stimuli_pca = (np.squeeze(all_images))
        # all_stimuli_pca=tf.squeeze(tf.convert_to_tensor(all_stimuli_pca))
        
        
        
        all_stimuli_concat = tf.concat(all_stimuli,axis=0)
        all_stimuli_pca=pca_transform(tf.squeeze(all_stimuli_concat).numpy(), 0.1)
        all_stimuli_pca=tf.squeeze(tf.convert_to_tensor(all_stimuli_pca))
        
        
        for n, (_, spikes) in test_ds.enumerate():
               print(n.numpy())
               obs_rate = spikes
               bias = n.numpy()*spikes.shape[1]
               stimuli_pca=all_stimuli_pca[(0+bias):(spikes.shape[1]+bias),:]
               stimuli_pca=tf.expand_dims(stimuli_pca, axis=0)
               stimuli_pca=tf.transpose(stimuli_pca,perm=[1,0,2])
               loss_pca_fwd_t, px_z_pca_fwd  = model_fwd.forward(stimuli_pca, obs_rate, return_loss=True)
               corr_pca_fwd = K.mean(correlation_coefficient(tf.squeeze(obs_rate), tf.squeeze(px_z_pca_fwd)))
               running_loss_pca_fwd += loss_pca_fwd_t.numpy()
               running_corr_pca_fwd += corr_pca_fwd.numpy()
               running_size += 1
                           

               # if n.numpy()==0:
               #      stimulii=[tf.reshape(tf.squeeze(stimuli),[-1, 50,50])]
               #      stimulii.append(tf.reshape(tf.squeeze(stimuli_pca),[-1, 50,50]))
               #      names=['Orig-Stimuli', 'PCA']
               #      write_dirs='PCA'
               #      if os.path.isdir(direct_main + write_dirs)==False:
               #          os.mkdir(direct_main + write_dirs)
               #          os.mkdir(direct_main + write_dirs +'/images')
               #      direct = direct_main + write_dirs + '/images/'
               #      save_img(stimulii,names,direct)
               #      gc.collect
               # else:
               #      continue
                
        running_corr_pca_fwd /= running_size
        running_loss_pca_fwd /=running_size
        return running_loss_pca_fwd, running_corr_pca_fwd

               
               
    elif task=='random':
        for n, (_, spikes) in test_ds.enumerate():
               stimuli = tf.random.uniform(shape=_.shape)
               obs_rate = spikes
               stimuli=tf.transpose(stimuli,perm=[1,0,2])
               loss_fwd_t, px_z_fwd  = model_fwd.forward(stimuli, obs_rate, return_loss=True)
               corr_fwd = K.mean(correlation_coefficient(tf.squeeze(obs_rate), tf.squeeze(px_z_fwd)))
               running_corr_fwd += corr_fwd.numpy()               
               running_loss_fwd += loss_fwd_t.numpy()               
               running_size += 1
               
        running_corr_fwd /= running_size
        running_loss_fwd /=running_size
        return running_loss_fwd,running_corr_fwd
        
    elif task=='stimuli-vis':
        for n, (stimuli, spikes) in test_ds.enumerate():
            if n.numpy()==4:
                obs_rate = spikes
                stimuli=tf.transpose(stimuli,perm=[1,0,2])
                _, _, _,stimuli_tnsf = model.forward(stimuli,obs_rate, return_loss=True,mask=mask)
                stimuli_pca=pca_transform(tf.squeeze(stimuli).numpy(), 1)
                stimuli_pca=tf.convert_to_tensor(stimuli_pca)
    
                stimulii=[tf.reshape(tf.squeeze(stimuli),[-1, 50,50])]
                stimulii.append(tf.reshape(tf.squeeze(stimuli_tnsf),[-1, 50,50]))
                stimulii.append(tf.reshape(tf.squeeze(stimuli_pca),[-1, 50,50]))
                names=['Orig-Stimuli', 'IB', 'PCA']
                write_dirs='PCA-IBGP'
                if os.path.isdir(direct_main + write_dirs)==False:
                    os.mkdir(direct_main + write_dirs)
                    os.mkdir(direct_main + write_dirs +'/images')
                direct= direct_main + write_dirs + '/images/'

                save_img(stimulii, names, direct)
                gc.collect
                make_anim(direct_main + write_dirs +'batch_{}.gif'.format(n.numpy()), direct, 1000)
            else:
                    continue


    else:
        print('undefined task')
               



def test_forward(data_ds, actor, model_true, mask_indx=None, write_dir=None, noise=None):
    

    running_corr_tnsf_fwd = 0
    running_loss_tnsf_fwd=0
    running_2DTV_tnsf=0
    running_2DTV_stimuli=0
    running_1DTV_tnsf=0
    running_1DTV_stimuli=0
    running_size = 0


    with open(write_dir[0], "wb") as f_stimuli, open(write_dir[1],"wb") as f_spikes,open('./Z',"ab") as f_latent, open('./opt-stimuli',"ab") as f_opt_stim:

        for n, (stimuli, spikes) in data_ds.enumerate():
            
               obs_rate = spikes
               stimuli = tf.transpose(stimuli,perm=[1,0,2])
               stimuli_tnsf = actor.forward(stimuli)
               
               TV2D_loss_tnsf_avg = tf.reduce_sum(tf.image.total_variation(tf.transpose(stimuli_tnsf,perm=[0,2,3,1])))
               TV2D_loss_stimuli_avg = tf.reduce_sum(tf.image.total_variation(tf.transpose(tf.reshape(stimuli,[-1,1,50,50]),perm=[0,2,3,1])))
               TV2D_loss_tnsf_all = tf.image.total_variation(tf.transpose(stimuli_tnsf,perm=[0,2,3,1]))
               TV2D_loss_stimuli_all = tf.image.total_variation(tf.transpose(stimuli_tnsf,perm=[0,2,3,1]))

               # np.savetxt('./2DTV_tnsf_all-{}.txt'.format(n.numpy()), TV2D_loss_tnsf_all.numpy(), delimiter=',')
               # np.savetxt('./2DTV_stimuli_all-{}.txt'.format(n.numpy()), TV2D_loss_stimuli_all.numpy(), delimiter=',')
               
               stimuli_tnsf.numpy().astype(np.float32).tofile(f_stimuli)
               stimuli_tnsf.numpy().astype(np.float32).tofile(f_opt_stim)
               stimuli_tnsf = tf.reshape(stimuli_tnsf,[actor.time_length, -1 ,actor.x_data_dim])
               
               TV1D_loss_tnsf_avg = np.mean(np.abs(np.diff(tf.reduce_sum(stimuli_tnsf,axis=[1,2]))))
               TV1D_loss_stimuli_avg = np.mean(np.abs(np.diff(tf.reduce_sum(stimuli,axis=[1,2]))))
               # loss_tnsf_fwd_t, px_z_tnsf_fwd,z, _  = model_true.forward(stimuli_tnsf, obs_rate,return_loss=True)               
               # z.numpy().astype(np.float32).tofile(f_latent)
               
               loss_tnsf_fwd_t, px_z_tnsf_fwd  = model_true.forward(stimuli_tnsf, obs_rate,return_loss=True)
               tf.squeeze(px_z_tnsf_fwd).numpy().astype(np.float32).tofile(f_spikes)
               corr_tnsf_fwd = K.mean(correlation_coefficient(tf.squeeze(obs_rate), tf.squeeze(px_z_tnsf_fwd)))

               assert  tf.math.is_finite(loss_tnsf_fwd_t)
               running_loss_tnsf_fwd += loss_tnsf_fwd_t.numpy()
               running_corr_tnsf_fwd += corr_tnsf_fwd.numpy()
               
               running_2DTV_tnsf += TV2D_loss_tnsf_avg.numpy()
               running_2DTV_stimuli += TV2D_loss_stimuli_avg.numpy()
               
               running_1DTV_tnsf += TV1D_loss_tnsf_avg
               running_1DTV_stimuli += TV1D_loss_stimuli_avg
               running_size += 1

        running_corr_tnsf_fwd /= running_size
        running_loss_tnsf_fwd /=running_size
        running_2DTV_tnsf     /= running_size
        running_2DTV_stimuli  /= running_size
        running_1DTV_tnsf     /= running_size
        running_1DTV_stimuli  /= running_size
        
    return running_loss_tnsf_fwd, running_corr_tnsf_fwd, running_2DTV_tnsf,running_2DTV_stimuli, running_1DTV_tnsf, running_1DTV_stimuli




