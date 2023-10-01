import tensorflow as tf
import tensorflow.keras.backend as K
import datetime
import matplotlib.pyplot as plt
from metrics import fraction_of_explained_variance,correlation_coefficient
from Generative_Models import VAE,GP_VAE
import numpy as np



log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


def save_img(stimuli,tsnf_stimuli_orig,tsnf_stimuli):
    
    
  for i in range(tsnf_stimuli_orig.shape[0]):
      fig,axs=plt.subplots(1, 3,figsize=(5, 5))
      axs[0].imshow(stimuli[i,0,:,:])
      axs[0].title.set_text('stimuli')
      axs[1].imshow(tsnf_stimuli_orig[i,0,:,:])
      axs[1].title.set_text('original')
      axs[2].imshow(tsnf_stimuli[i,0,:,:])
      axs[2].title.set_text('ablated')
      fig.savefig('./images/images{}.png'.format(i+1))
      plt.close('all')

@tf.function
def train_vae(stimuli, spikes, epoch,train_what, model, vae_optimizer,beta, BATCH_size,actor=None, second_model=None, TV=None, alpha=None):
   obs_rate=spikes
   

   if train_what=='train':
       
       
       gen_total_loss_m = [None]*BATCH_size
       nll = [None]*BATCH_size
       kl = [None]*BATCH_size
       mse = [None]*BATCH_size
       nll_norm = [None]*BATCH_size
       kl_norm = [None]*BATCH_size
       mse_norm = [None]*BATCH_size
       est_rate = [None]*BATCH_size
       corr = [None]*BATCH_size
      
       
      
       with tf.GradientTape() as vae_tape:
           gen_total_loss=0
          
           for m in range(BATCH_size):
          
               gen_total_loss_m[m],nll[m],kl[m],mse[m], nll_norm[m],kl_norm[m], mse_norm[m], est_rate[m] = model.compute_loss(stimuli[:,m,:][:,tf.newaxis,:],obs_rate[m,:,:][tf.newaxis,...],return_parts=True)
               gen_total_loss += gen_total_loss_m[m]
           gen_total_loss /=BATCH_size
          

       vae_gradients = vae_tape.gradient(gen_total_loss,
                                               model.trainable_variables)
       vae_optimizer.apply_gradients(zip(vae_gradients,
                                               model.trainable_variables))

       for m in range(BATCH_size):
          
           corr[m]=K.mean(correlation_coefficient(tf.squeeze(obs_rate[m,...]),tf.squeeze(est_rate[m])))
      
       nll=sum(nll)/BATCH_size
       nll_norm=sum(nll_norm)/BATCH_size
       mse=sum(mse)/BATCH_size
       mse_norm=sum(mse_norm)/BATCH_size
       corr=sum(corr)/BATCH_size
       kl=sum(kl)/BATCH_size
       kl_norm=sum(kl_norm)/BATCH_size


      #  with tf.name_scope('vae/train'):
      #   with summary_writer.as_default():
      #       tf.summary.scalar('loss-total-train', gen_total_loss, step=epoch)
      #       tf.summary.scalar('loss-rec-train', nll, step=epoch)
      #       tf.summary.scalar('loss-rec-norm-train', nll_norm, step=epoch)
      #       tf.summary.scalar('loss-mse-norm-train', mse_norm, step=epoch)
      #       tf.summary.scalar('loss-mse-train', mse, step=epoch)
      #       tf.summary.scalar('loss-KL-train', kl, step=epoch)
      #       tf.summary.scalar('loss-KL-norm-train', kl_norm, step=epoch)
            
      #       tf.summary.scalar('corr-train',corr,step=epoch)
      #       tf.summary.image('stimuli_train', tf.transpose(tf.reshape(stimuli,[model.time_length, -1,tf.math.sqrt(model.x_data_dim +0.),tf.math.sqrt(model.x_data_dim +0.) ]), perm=[0,2,3,1]),step=epoch)
      #       # tf.summary.image('stimuli_tnsf_train', tf.transpose(tf.reshape(stimuli_tnsf,[model.time_length, -1,tf.math.sqrt(model.x_data_dim +0.),tf.math.sqrt(model.x_data_dim +0.) ]), perm=[0,2,3,1]),step=epoch)

        
   elif train_what=='val':

      gen_total_loss,nll,kl,mse, nll_norm,kl_norm, mse_norm, est_rate = model.compute_loss(stimuli,obs_rate,return_parts=True)
      
      corr=K.mean(correlation_coefficient(tf.squeeze(obs_rate),tf.squeeze(est_rate)))

      # with tf.name_scope('vae/val'):
      #   with summary_writer.as_default():
      #       tf.summary.scalar('loss-total-val', gen_total_loss, step=epoch)
      #       tf.summary.scalar('loss-rec-val', nll, step=epoch)
      #       tf.summary.scalar('loss-rec-norm-val', nll_norm, step=epoch)
      #       tf.summary.scalar('loss-mse-norm-val', mse_norm, step=epoch)
      #       tf.summary.scalar('loss-mse-val', mse, step=epoch)
      #       tf.summary.scalar('loss-KL-val', kl, step=epoch)
      #       tf.summary.scalar('loss-KL-norm-val', kl_norm, step=epoch)
            
      #       tf.summary.scalar('corr-val',corr,step=epoch)
      #       tf.summary.image('stimuli_val', tf.transpose(tf.reshape(stimuli,[model.time_length, -1,tf.math.sqrt(model.x_data_dim +0.),tf.math.sqrt(model.x_data_dim +0.) ]), perm=[0,2,3,1]),step=epoch)
      #       # tf.summary.image('stimuli_tnsf_val', tf.transpose(tf.reshape(stimuli_tnsf,[model.time_length, -1,tf.math.sqrt(model.x_data_dim +0.),tf.math.sqrt(model.x_data_dim +0.) ]), perm=[0,2,3,1]),step=epoch)



   elif train_what=='test':

       
      gen_total_loss,nll,kl,mse, nll_norm,kl_norm, mse_norm, est_rate = model.compute_loss(stimuli,obs_rate,return_parts=True)
      
      corr=K.mean(correlation_coefficient(tf.squeeze(obs_rate),tf.squeeze(est_rate)))



      with tf.name_scope('vae/test'):
        with summary_writer.as_default():
            tf.summary.scalar('loss-total-test', gen_total_loss, step=epoch)
            tf.summary.scalar('loss-rec-test', nll, step=epoch)
            tf.summary.scalar('loss-rec-norm-test', nll_norm, step=epoch)
            tf.summary.scalar('loss-mse-val', mse, step=epoch)
            tf.summary.scalar('loss-mse-norm-test', mse_norm, step=epoch)
            tf.summary.scalar('loss-KL-test', kl, step=epoch)
            tf.summary.scalar('loss-KL-norm-test', kl_norm, step=epoch)
            
            tf.summary.scalar('corr-test',corr,step=epoch)
            tf.summary.image('stimuli_test', tf.transpose(tf.reshape(stimuli,[model.time_length, -1,tf.math.sqrt(model.x_data_dim +0.),tf.math.sqrt(model.x_data_dim +0.) ]), perm=[0,2,3,1]),step=epoch)
            # tf.summary.image('stimuli_test', tf.transpose(tf.reshape(stimuli_tnsf,[model.time_length, -1,tf.math.sqrt(model.x_data_dim +0.),tf.math.sqrt(model.x_data_dim +0.) ]), perm=[0,2,3,1]),step=epoch)


   return gen_total_loss,nll_norm,kl_norm,mse_norm, corr



@tf.function
def adaptive_train_fwd(stimuli, spikes, epoch,train_what, model, model_optimizer,beta, BATCH_size,  
                                                                                   second_model=None,TV=None, alpha=None):
   obs_rate=spikes
   a,b,c= model.forward(stimuli)

   if train_what=='train':
       
       
       gen_total_loss_m = [None]*BATCH_size

       est_rate = [None]*BATCH_size
       corr = [None]*BATCH_size

       with tf.GradientTape() as IB_tape:
           gen_total_loss=0
          
           for m in range(BATCH_size):
               
               gen_total_loss_m[m], est_rate[m] = model.compute_loss(stimuli[:,m,:][:,tf.newaxis,:],obs_rate[m,:,:][tf.newaxis,...])
               gen_total_loss += gen_total_loss_m[m]
               
           gen_total_loss /=BATCH_size
          

       IB_gradients = IB_tape.gradient(gen_total_loss,
                                               model.trainable_variables)
       model_optimizer.apply_gradients(zip(IB_gradients,
                                               model.trainable_variables))

       for m in range(BATCH_size):
          
           corr[m]=K.mean(correlation_coefficient(tf.squeeze(obs_rate[m,...]),tf.squeeze(est_rate[m])))

       corr=sum(corr)/BATCH_size


   elif train_what=='val':
       

      gen_total_loss, est_rate = model.compute_loss(stimuli,obs_rate)
           
      corr=K.mean(correlation_coefficient(tf.squeeze(obs_rate),tf.squeeze(est_rate)))


   elif train_what=='test':

       

      gen_total_loss, est_rate = model.compute_loss(stimuli,obs_rate)
      corr=K.mean(correlation_coefficient(tf.squeeze(obs_rate),tf.squeeze(est_rate)))



      with tf.name_scope('vae/test'):
        with summary_writer.as_default():
            tf.summary.scalar('loss-total-test', gen_total_loss, step=epoch)


            
            tf.summary.scalar('corr-test',corr,step=epoch)
            tf.summary.image('stimuli_test', tf.transpose(tf.reshape(stimuli,[model.time_length, -1,tf.math.sqrt(model.x_data_dim +0.),tf.math.sqrt(model.x_data_dim +0.) ]), perm=[0,2,3,1]),step=epoch)
            # tf.summary.image('stimuli_test', tf.transpose(tf.reshape(stimuli_tnsf,[model.time_length, -1,tf.math.sqrt(model.x_data_dim +0.),tf.math.sqrt(model.x_data_dim +0.) ]), perm=[0,2,3,1]),step=epoch)


   return gen_total_loss,tf.constant(1., dtype=tf.float32),tf.constant(1., dtype=tf.float32),tf.constant(1., dtype=tf.float32), corr


@tf.function
def adaptive_train_actor(stimuli, spikes, epoch,train_what, actor, actor_optimizer,beta, BATCH_size,
                                                                                      second_model=None,TV=None,alpha=None):
   obs_rate=spikes
   model=second_model
   mask_indx=[_ for _ in range(15)]
   if isinstance(model,GP_VAE):
      mask = np.zeros((1,model.latent_dim, model.latent_time_length),dtype=np.float32)
      mask[:,mask_indx,:]=1.0
   else:
      mask = np.zeros((1,model.time_length//2, model.latent_dim),dtype=np.float32)
      mask[:,:, mask_indx]=1.0
      
   mask = tf.convert_to_tensor(mask)

   if train_what=='train':

       with tf.GradientTape() as actor_tape:

                   optimized_stimuli = actor.forward(stimuli)
                   TV_loss = tf.reduce_sum(tf.image.total_variation(tf.transpose(optimized_stimuli,perm=[0,2,3,1])))
                   y_loss, est_rate = model.compute_loss(tf.reshape(optimized_stimuli,[model.time_length,-1,model.x_data_dim]),obs_rate, mask=mask)
                   mse = tf.reduce_sum(tf.math.square(tf.reshape(optimized_stimuli,[model.time_length,-1,model.x_data_dim])- stimuli), axis = [0, 1, 2])
                   gen_total_loss=y_loss + (mse*alpha) + TV*TV_loss

       actor_gradients = actor_tape.gradient(gen_total_loss,
                                               actor.trainable_variables)
       actor_optimizer.apply_gradients(zip(actor_gradients,
                                               actor.trainable_variables))

       corr=K.mean(correlation_coefficient(tf.squeeze(obs_rate),tf.squeeze(est_rate)))

   elif train_what=='val':
       

       
      optimized_stimuli = actor.forward(stimuli)
      TV_loss = tf.reduce_sum(tf.image.total_variation(tf.transpose(optimized_stimuli,perm=[0,2,3,1])))
      y_loss, est_rate = model.compute_loss(tf.reshape(optimized_stimuli,[model.time_length,-1,model.x_data_dim]),obs_rate, mask=mask)
      mse = tf.reduce_sum(tf.math.square(tf.reshape(optimized_stimuli,[model.time_length,-1,model.x_data_dim])- stimuli), axis = [0, 1, 2])
      gen_total_loss = y_loss + (mse*alpha) + (TV*TV_loss)
           
      corr=K.mean(correlation_coefficient(tf.squeeze(obs_rate),tf.squeeze(est_rate)))


   elif train_what=='test':

      optimized_stimuli = actor.forward(stimuli)
      TV_loss = tf.reduce_sum(tf.image.total_variation(tf.transpose(optimized_stimuli,perm=[0,2,3,1])))
      y_loss, est_rate = model.compute_loss(tf.reshape(optimized_stimuli,[model.time_length,-1,model.x_data_dim]),obs_rate, mask=mask)
      mse = tf.reduce_sum(tf.math.square(tf.reshape(optimized_stimuli,[model.time_length,-1,model.x_data_dim])- stimuli), axis = [0, 1, 2])
      gen_total_loss=y_loss + (mse*alpha) + (TV*TV_loss)
           
      corr=K.mean(correlation_coefficient(tf.squeeze(obs_rate),tf.squeeze(est_rate)))


      with tf.name_scope('vae-actor/test'):
        with summary_writer.as_default():
            tf.summary.scalar('loss-total-test', gen_total_loss, step=epoch)
            tf.summary.scalar('loss-TV-test', TV_loss, step=epoch)
            tf.summary.scalar('loss-poisson-test', y_loss, step=epoch)
            tf.summary.scalar('loss-mse-test', mse, step=epoch)
            tf.summary.scalar('corr-test',corr,step=epoch)
            tf.summary.image('stimuli_test', tf.transpose(tf.reshape(stimuli,[model.time_length, -1,tf.math.sqrt(model.x_data_dim +0.),tf.math.sqrt(model.x_data_dim +0.) ]), perm=[0,2,3,1]),step=epoch)
            # tf.summary.image('stimuli_test', tf.transpose(tf.reshape(stimuli_tnsf,[model.time_length, -1,tf.math.sqrt(model.x_data_dim +0.),tf.math.sqrt(model.x_data_dim +0.) ]), perm=[0,2,3,1]),step=epoch)

   return gen_total_loss,tf.constant(1., dtype=tf.float32),tf.constant(1., dtype=tf.float32),tf.constant(1., dtype=tf.float32), corr



@tf.function
def train_vae_FwdOnly(stimuli, spikes, epoch,train_what, model, vae_optimizer,beta):
   obs_rate=spikes

   if train_what=='train':
      
         
      with tf.GradientTape() as vae_tape:
          
          gen_total_loss, nll, nll_norm, est_rate = model.compute_loss(stimuli,obs_rate,return_parts=True, FwdOnly=True)

      
      vae_gradients = vae_tape.gradient(gen_total_loss,
                                               model.decoder_fwd.trainable_variables)
      vae_optimizer.apply_gradients(zip(vae_gradients,
                                               model.decoder_fwd.trainable_variables))

      corr=K.mean(correlation_coefficient(tf.squeeze(obs_rate),tf.squeeze(est_rate)))


      with tf.name_scope('fwd/train'):
        with summary_writer.as_default():
            tf.summary.scalar('loss-total-train', gen_total_loss, step=epoch)
            tf.summary.scalar('loss-rec-train', nll, step=epoch)
            tf.summary.scalar('loss-rec-norm-train', nll_norm, step=epoch)
            tf.summary.scalar('corr-train',corr,step=epoch)
            tf.summary.image('stimuli_train', tf.transpose(tf.reshape(stimuli,[model.time_length, -1,tf.math.sqrt(model.x_data_dim +0.),tf.math.sqrt(model.x_data_dim +0.) ]), perm=[0,2,3,1]),step=epoch)

        
   elif train_what=='val':

      gen_total_loss, nll, nll_norm, est_rate = model.compute_loss(stimuli,obs_rate,return_parts=True, FwdOnly=True)
      
      corr=K.mean(correlation_coefficient(tf.squeeze(obs_rate),tf.squeeze(est_rate)))

      with tf.name_scope('fwd/val'):
        with summary_writer.as_default():
            tf.summary.scalar('loss-total-val', gen_total_loss, step=epoch)
            tf.summary.scalar('loss-rec-val', nll, step=epoch)
            tf.summary.scalar('loss-rec-norm-val', nll_norm, step=epoch)
            tf.summary.scalar('corr-val',corr,step=epoch)
            tf.summary.image('stimuli_val', tf.transpose(tf.reshape(stimuli,[model.time_length, -1,tf.math.sqrt(model.x_data_dim +0.),tf.math.sqrt(model.x_data_dim +0.) ]), perm=[0,2,3,1]),step=epoch)



   elif train_what=='test':

       
      gen_total_loss, nll, nll_norm, est_rate = model.compute_loss(stimuli,obs_rate,return_parts=True, FwdOnly=True)

      corr=K.mean(correlation_coefficient(tf.squeeze(obs_rate),tf.squeeze(est_rate)))



      with tf.name_scope('fwd/test'):
        with summary_writer.as_default():
            tf.summary.scalar('loss-total-test', gen_total_loss, step=epoch)
            tf.summary.scalar('loss-rec-test', nll, step=epoch)
            tf.summary.scalar('loss-rec-norm-test', nll_norm, step=epoch)
            tf.summary.scalar('corr-test',corr,step=epoch)
            tf.summary.image('stimuli_test', tf.transpose(tf.reshape(stimuli,[model.time_length, -1,tf.math.sqrt(model.x_data_dim +0.),tf.math.sqrt(model.x_data_dim +0.) ]), perm=[0,2,3,1]),step=epoch)

   return est_rate,obs_rate,gen_total_loss,corr



@tf.function
def train_vae_DecodeFwd(stimuli, spikes, epoch,train_what, model, vae_optimizer,beta):
   obs_rate=spikes
   n_avg=1
   if train_what=='train':
      
         
      with tf.GradientTape() as vae_tape:
          
            gen_total_loss,nll,kl,mse, nll_norm,kl_norm, mse_norm, est_rate = model.compute_loss(stimuli,obs_rate,return_parts=True)

      
      vae_gradients = vae_tape.gradient(gen_total_loss,
                                              model.decoder_decode.trainable_variables)
      vae_optimizer.apply_gradients(zip(vae_gradients,
                                              model.decoder_decode.trainable_variables))
      est_rate_expected=tf.zeros_like(est_rate)
      
      for i in range(n_avg):
           _, stimuli_tnsf, est_rate= model.forward(stimuli)
           est_rate_expected += est_rate
           
      est_rate_expected/=n_avg
      corr=K.mean(correlation_coefficient(tf.squeeze(obs_rate),tf.squeeze(est_rate_expected)))



      with tf.name_scope('vae_iter/train'):
        with summary_writer.as_default():
            tf.summary.scalar('loss-total-train', gen_total_loss, step=epoch)
            tf.summary.scalar('loss-rec-train', nll, step=epoch)
            tf.summary.scalar('loss-rec-norm-train', nll_norm, step=epoch)
            tf.summary.scalar('loss-mse-norm-train', mse_norm, step=epoch)
            tf.summary.scalar('loss-mse-train', mse, step=epoch)
            tf.summary.scalar('loss-KL-train', kl, step=epoch)
            tf.summary.scalar('loss-KL-norm-train', kl_norm, step=epoch)

            tf.summary.scalar('corr-train',corr,step=epoch)
            tf.summary.image('stimuli_train', tf.transpose(tf.reshape(stimuli,[model.time_length, -1,tf.math.sqrt(model.x_data_dim +0.),tf.math.sqrt(model.x_data_dim +0.) ]), perm=[0,2,3,1]),step=epoch)
            tf.summary.image('stimuli_tnsf_train', tf.transpose(tf.reshape(stimuli_tnsf,[model.time_length, -1,tf.math.sqrt(model.x_data_dim +0.),tf.math.sqrt(model.x_data_dim +0.) ]), perm=[0,2,3,1]),step=epoch)

        
   elif train_what=='val':

      gen_total_loss,nll,kl,mse, nll_norm,kl_norm, mse_norm, est_rate = model.compute_loss(stimuli,obs_rate,return_parts=True)
      
      est_rate_expected=tf.zeros_like(est_rate)
      for i in range(n_avg):
           _, stimuli_tnsf, est_rate= model.forward(stimuli)
           est_rate_expected+=est_rate
           
      est_rate_expected/=n_avg
      corr=K.mean(correlation_coefficient(tf.squeeze(obs_rate),tf.squeeze(est_rate_expected)))

      with tf.name_scope('vae_iter/val'):
        with summary_writer.as_default():
            tf.summary.scalar('loss-total-val', gen_total_loss, step=epoch)
            tf.summary.scalar('loss-rec-val', nll, step=epoch)
            tf.summary.scalar('loss-rec-norm-val', nll_norm, step=epoch)
            tf.summary.scalar('loss-mse-norm-val', mse_norm, step=epoch)
            tf.summary.scalar('loss-mse-val', mse, step=epoch)
            tf.summary.scalar('loss-KL-val', kl, step=epoch)
            tf.summary.scalar('loss-KL-norm-val', kl_norm, step=epoch)
            
            tf.summary.scalar('corr-val',corr,step=epoch)
            tf.summary.image('stimuli_val', tf.transpose(tf.reshape(stimuli,[model.time_length, -1,tf.math.sqrt(model.x_data_dim +0.),tf.math.sqrt(model.x_data_dim +0.) ]), perm=[0,2,3,1]),step=epoch)
            tf.summary.image('stimuli_tnsf_val', tf.transpose(tf.reshape(stimuli_tnsf,[model.time_length, -1,tf.math.sqrt(model.x_data_dim +0.),tf.math.sqrt(model.x_data_dim +0.) ]), perm=[0,2,3,1]),step=epoch)



   elif train_what=='test':

       
      gen_total_loss,nll,kl,mse, nll_norm,kl_norm, mse_norm, est_rate = model.compute_loss(stimuli,obs_rate,return_parts=True)
      est_rate_expected=tf.zeros_like(est_rate)
      for i in range(n_avg):
           _, stimuli_tnsf, est_rate= model.forward(stimuli)
           est_rate_expected+=est_rate
           
      est_rate_expected/=n_avg
      corr=K.mean(correlation_coefficient(tf.squeeze(obs_rate),tf.squeeze(est_rate_expected)))



      with tf.name_scope('vae_iter/test'):
        with summary_writer.as_default():
            tf.summary.scalar('loss-total-test', gen_total_loss, step=epoch)
            tf.summary.scalar('loss-rec-test', nll, step=epoch)
            tf.summary.scalar('loss-rec-norm-test', nll_norm, step=epoch)
            tf.summary.scalar('loss-mse-val', mse, step=epoch)
            tf.summary.scalar('loss-mse-norm-test', mse_norm, step=epoch)
            tf.summary.scalar('loss-KL-test', kl, step=epoch)
            tf.summary.scalar('loss-KL-norm-test', kl_norm, step=epoch)
            
            tf.summary.scalar('corr-test',corr,step=epoch)
            tf.summary.image('stimuli_test', tf.transpose(tf.reshape(stimuli,[model.time_length, -1,tf.math.sqrt(model.x_data_dim +0.),tf.math.sqrt(model.x_data_dim +0.) ]), perm=[0,2,3,1]),step=epoch)
            tf.summary.image('stimuli_test', tf.transpose(tf.reshape(stimuli_tnsf,[model.time_length, -1,tf.math.sqrt(model.x_data_dim +0.),tf.math.sqrt(model.x_data_dim +0.) ]), perm=[0,2,3,1]),step=epoch)


   return est_rate,obs_rate,gen_total_loss,corr
