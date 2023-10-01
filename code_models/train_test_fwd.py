import tensorflow as tf
import tensorflow.keras.backend as K
import datetime
import matplotlib.pyplot as plt
from metrics import fraction_of_explained_variance,correlation_coefficient

import numpy as np



log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# @tf.function

# @tf.function
def windowed_data(x_train, w_size=30,shift=1 ):
    x_train = tf.squeeze(x_train)
    
    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    dataset = dataset.window(size = w_size, shift = shift, drop_remainder = True)

    return dataset.flat_map(create_seqeunce_ds)

@tf.function
def train_backward(stimuli, spikes, epoch,train_what, model, model_optimizer):
    

   w_size = model.window_length
   obs_rate = spikes
   n_avg=1
   shift=1
   
   BATCH_size = stimuli.shape[0] - w_size + 1
   gen_total_loss_n = []
   y_pred_n = []

   corr_n = []
   
   
   if train_what=='train':
       
     # windowed_dataset = windowed_data(obs_rate, w_size=w_size,shift=1 )
     windowed_dataset = tf.data.Dataset.from_tensor_slices(tf.squeeze(obs_rate))
     windowed_dataset = windowed_dataset.window(size = w_size, shift = shift, drop_remainder = True)
     
     def create_seqeunce_ds(chunk, batch_size=30):
        return chunk.batch(batch_size, drop_remainder=True)
    
     windowed_dataset = windowed_dataset.flat_map(create_seqeunce_ds)
     counter = 0
     with tf.GradientTape() as model_tape:
              gen_total_loss = 0.0
              # for n, window in windowed_dataset.enumerate():
                  
              #           stimuli_n = stimuli[counter,0,:][tf.newaxis,...]
              #           temp_loss, temp_y_pred = model.compute_loss(window, stimuli_n, return_parts=True)
              #           y_pred_n.append(temp_y_pred)
              #           gen_total_loss_n.append(temp_loss)
              #           gen_total_loss += temp_loss
              #           # print(n)
              #           counter +=1
              
              for n in range(BATCH_size):
                  
                        stimuli_n = stimuli[counter,0,:][tf.newaxis,...]
                        window = obs_rate[0,(n):(w_size+n),:]
                        temp_loss, temp_y_pred = model.compute_loss(window, stimuli_n, return_parts=True)
                        y_pred_n.append(temp_y_pred)
                        gen_total_loss_n.append(temp_loss)
                        gen_total_loss += temp_loss
                        # print(n)
                        counter +=1
              # stimuli_n = stimuli[counter,0,:][tf.newaxis,...]
              # gen_total_loss, temp_y_pred = model.compute_loss(obs_rate[0,0:30,:], stimuli_n, return_parts=True)
              gen_total_loss /=BATCH_size
    
              
     model_gradients = model_tape.gradient(gen_total_loss,
                                                      model.trainable_variables)
     model_optimizer.apply_gradients(zip(model_gradients, model.trainable_variables))
     for m in range(BATCH_size):
                  
                   corr_n.append(K.mean(correlation_coefficient(tf.squeeze(stimuli[m, 0, :]),tf.squeeze(y_pred_n[m]))))
              
     corr=sum(corr_n)/BATCH_size
     with tf.name_scope('fwd/train'):
                with summary_writer.as_default():
                    tf.summary.scalar('loss-total-train', gen_total_loss, step=epoch)
                    tf.summary.scalar('corr',corr,step=epoch)
                    tf.summary.image('stimuli_GT', tf.transpose(tf.reshape(stimuli[0:BATCH_size,0,:],[BATCH_size, -1,tf.math.sqrt(model.y_data_dim +0.),tf.math.sqrt(model.y_data_dim +0.) ]), perm=[0,2,3,1]),step=epoch)
                    tf.summary.image('stimuli_pred', tf.transpose(tf.reshape(y_pred_n[0:BATCH_size],[BATCH_size, -1,tf.math.sqrt(model.y_data_dim +0.),tf.math.sqrt(model.y_data_dim +0.) ]), perm=[0,2,3,1]),step=epoch)

   elif train_what=='val':
      print('val')
       
      windowed_dataset = windowed_data(obs_rate, w_size=w_size,shift=1 )
      gen_total_loss = 0
      counter = 0
      for n, window in windowed_dataset.enumerate():
                        stimuli_n = stimuli[counter,0,:][tf.newaxis,...]
                        gen_total_loss_n[counter], y_pred_n[counter] = model.compute_loss(window, stimuli_n, return_parts=True)
                        gen_total_loss += gen_total_loss_n[counter]
                        counter +=1
        
      gen_total_loss /=BATCH_size
              

           
      for m in range(BATCH_size):
                   corr_n[m]=K.mean(correlation_coefficient(tf.squeeze(stimuli[m, 0,:]),tf.squeeze(y_pred_n[m])))
                   
      corr=sum(corr_n)/BATCH_size
      with tf.name_scope('fwd/val'):
        with summary_writer.as_default():
            tf.summary.scalar('loss-total-val', gen_total_loss, step=epoch)
            tf.summary.scalar('corr',corr,step=epoch)
            tf.summary.image('stimuli_GT', tf.transpose(tf.reshape(stimuli[0:BATCH_size, 0,:],[BATCH_size, -1,tf.math.sqrt(model.y_data_dim +0.),tf.math.sqrt(model.y_data_dim +0.) ]), perm=[0,2,3,1]),step=epoch)
            tf.summary.image('stimuli_pred', tf.transpose(tf.reshape(y_pred_n[0:BATCH_size],[BATCH_size, -1,tf.math.sqrt(model.y_data_dim +0.),tf.math.sqrt(model.y_data_dim +0.) ]), perm=[0,2,3,1]),step=epoch)


   elif train_what=='test':
      print('val')
      windowed_dataset = windowed_data(obs_rate, w_size=w_size,shift=1 )
      gen_total_loss = 0
      counter = 0
      for n, window in windowed_dataset.enumerate():
                        stimuli_n = stimuli[counter,0,:][tf.newaxis,...]
                        gen_total_loss_n[counter], y_pred_n[counter] = model.compute_loss(window, stimuli_n, return_parts=True)
                        gen_total_loss += gen_total_loss_n[counter]
                        counter +=1
        
      gen_total_loss /=BATCH_size
              
      for m in range(BATCH_size):
                   corr_n[m]=K.mean(correlation_coefficient(tf.squeeze(stimuli[m, 0, :]),tf.squeeze(y_pred_n[m])))
                   
      corr=sum(corr_n)/BATCH_size
      with tf.name_scope('fwd/test'):
        with summary_writer.as_default():
            tf.summary.scalar('loss-total-test', gen_total_loss, step=epoch)
            tf.summary.scalar('corr',corr,step=epoch)
            tf.summary.image('stimuli_GT', tf.transpose(tf.reshape(stimuli[0:BATCH_size,0,:],[BATCH_size, -1,tf.math.sqrt(model.y_data_dim +0.),tf.math.sqrt(model.y_data_dim +0.) ]), perm=[0,2,3,1]),step=epoch)
            tf.summary.image('stimuli_pred', tf.transpose(tf.reshape(y_pred_n[0:BATCH_size],[BATCH_size, -1,tf.math.sqrt(model.y_data_dim +0.),tf.math.sqrt(model.y_data_dim +0.) ]), perm=[0,2,3,1]),step=epoch)

   return y_pred_n[0:BATCH_size], stimuli[0:BATCH_size,...], gen_total_loss, corr



# @tf.function
def train_forward(stimuli, spikes, epoch,train_what, model, model_optimizer):
    
    obs_rate = spikes
    windowed_data(spikes, w_size=30,shift=1 )
   
   
    n_avg=1
    if train_what=='train':
      
      
      with tf.GradientTape() as model_tape:
          
            gen_total_loss,est_rate,nll= model.compute_loss(stimuli,obs_rate,return_parts=True)


      
      model_gradients = model_tape.gradient(gen_total_loss,
                                              model.trainable_variables)
      model_optimizer.apply_gradients(zip(model_gradients,
                                              model.trainable_variables))
      est_rate_expected=tf.zeros_like(est_rate)
     
      for i in range(n_avg):
           
            _,est_rate,_= model.compute_loss(stimuli,obs_rate,return_parts=True)
            est_rate_expected+=est_rate
      est_rate_expected/=n_avg
      corr=K.mean(correlation_coefficient(tf.squeeze(obs_rate),tf.squeeze(est_rate_expected)))



      # if epoch%3==0:
      #     plt.plot(mean[:,0].numpy())
      with tf.name_scope('fwd/train'):
        with summary_writer.as_default():
            tf.summary.scalar('loss-total-train', gen_total_loss, step=epoch)
            tf.summary.scalar('loss-rec-train', nll, step=epoch)
            tf.summary.scalar('corr',corr,step=epoch)


        
    elif train_what=='val':

      gen_total_loss,est_rate,nll= model.compute_loss(stimuli,obs_rate,return_parts=True)

      

      est_rate_expected=tf.zeros_like(est_rate)
      for i in range(n_avg):
            _,est_rate,_= model.compute_loss(stimuli,obs_rate,return_parts=True)
            est_rate_expected+=est_rate
      est_rate_expected/=n_avg
      corr=K.mean(correlation_coefficient(tf.squeeze(obs_rate),tf.squeeze(est_rate_expected)))
      # if epoch%3==0:
      #     plt.plot(mean[:,0].numpy())
      
      with tf.name_scope('fwd/val'):
        with summary_writer.as_default():
            tf.summary.scalar('loss-total-val', gen_total_loss, step=epoch)
            tf.summary.scalar('loss-rec-val', nll, step=epoch)
            tf.summary.scalar('corr',corr,step=epoch)



    elif train_what=='test':

       
      gen_total_loss,est_rate,nll= model.compute_loss(stimuli,obs_rate,return_parts=True)
      
      est_rate_expected=tf.zeros_like(est_rate)
      for i in range(n_avg):
            _,est_rate,_= model.compute_loss(stimuli,obs_rate,return_parts=True)
            est_rate_expected+=est_rate
      est_rate_expected/=n_avg
      corr=K.mean(correlation_coefficient(tf.squeeze(obs_rate),tf.squeeze(est_rate_expected)))


      # if epoch%3==0:
      #     plt.plot(mean[:,0].numpy())
      with tf.name_scope('fwd/test'):
        with summary_writer.as_default():
            tf.summary.scalar('loss-total-test', gen_total_loss, step=epoch)
            tf.summary.scalar('loss-rec-test', nll, step=epoch)
            tf.summary.scalar('corr',corr,step=epoch)


    return est_rate,obs_rate,gen_total_loss,corr
