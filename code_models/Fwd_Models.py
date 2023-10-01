import tensorflow as tf
import numpy as np
from my_network import forward_model_base, forward_model_noisy, actor_cnn, perception_network
from metrics import correlation_coefficient
import math

# FWD_model
class Fwd(tf.keras.Model):
    def __init__(self, output_size, height_in, width_in, **kwargs):
        """ Decoder parent class with no specified output distribution
            :param output_size: output dimensionality
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
        """
        super(Fwd, self).__init__()
        self.net =  forward_model_noisy(output_size, HEIGHT_in=height_in ,WIDTH_in=width_in)
        self.output_size=output_size

    def __call__(self, x, training = None):
        mapped = self.net(x)
        
        return mapped
    
class ML_fwd(tf.keras.Model):
    def __init__(self, x_data_dim, y_data_dim, time_length,
                  model=Fwd,**kwargs):
        """ Basic Variational Autoencoder with Standard Normal prior
            :param latent_dim: latent space dimensionality
            :param data_dim: original data dimensionality
            :param time_length: time series duration
            
        """
        super(ML_fwd, self).__init__()


        self.x_data_dim = x_data_dim
        self.time_length = time_length
        self.model = model(y_data_dim, int(math.sqrt(x_data_dim)),int(math.sqrt(x_data_dim)))
        self.output_size=y_data_dim

            
    def my_poisson(self, y_true,y_pred):
        
        return y_pred-tf.math.multiply(y_true,tf.math.log(y_pred))
    
    def __call__(self, inputs):
        return self.model(inputs) + 1e-4

    def _compute_loss(self, x, y, return_parts=False):
        assert len(x.shape) == 3, "Input should have shape: [time_length, batch_size, data_dim]"
        x = tf.identity(x)  # in case x is not a Tensor already...

        x = tf.reshape(x, [self.time_length, -1, int(np.sqrt(0.0 + self.x_data_dim)), int(np.sqrt(0.0 + self.x_data_dim))])
        
        y_hat = self.model(x) #(1, TL, d)
        y_hat += 1e-4
        
        nll = self.my_poisson(y, y_hat) # shape=(1,T,d)
        nll = tf.reduce_mean(nll,axis=2) # shape=(1,T)
        nll = tf.reduce_mean(nll, 1) # scalar

        loss = tf.reduce_mean(nll)  # scalar

        if return_parts:
            nll = tf.reduce_mean(nll)  # scalar
            return loss, y_hat, nll
        else:
            return loss, y_hat

    def compute_loss(self, x, y, return_parts=False):
        return self._compute_loss(x,y, return_parts=return_parts)

    def forward(self, x, y, return_loss=None):
        
        assert len(x.shape) == 3, "Input should have shape: [time_length, batch_size, data_dim]"
        x = tf.identity(x)  # in case x is not a Tensor already...
        
        x = tf.reshape(x, [self.time_length, -1, int(tf.math.sqrt(0.0 + self.x_data_dim)), int(tf.math.sqrt(0.0 + self.x_data_dim))])
        
        y_hat = self.model(x) #(1, TL, d)
        y_hat += 1e-4
        
        if return_loss == True:
            
            nll = self.my_poisson(y, y_hat) # shape=(1,T,d)
            nll = tf.reduce_sum(nll,axis=2) # shape=(1,T)
            nll = tf.reduce_sum(nll, 1) # scalar
            loss = tf.reduce_mean(nll)  # scalar
            return  loss/(self.time_length*self.output_size), y_hat
        else:
            return y_hat

# Actor_model
class Actor(tf.keras.Model):
    def __init__(self, latent_dim, **kwargs):
        """ Decoder parent class with no specified output distribution
            :param output_size: output dimensionality
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
        """
        super(Actor, self).__init__()
        self.net =  actor_cnn(latent_dim)

    def __call__(self, x, training = None):
        mapped = self.net(x)
        
        return mapped
    
class ML_actor(tf.keras.Model):
    def __init__(self, x_data_dim, latent_dim, time_length,
                  model=Actor,**kwargs):
        """ Basic Variational Autoencoder with Standard Normal prior
            :param latent_dim: latent space dimensionality
            :param data_dim: original data dimensionality
            :param time_length: time series duration
            
        """
        super(ML_actor, self).__init__()


        self.x_data_dim = x_data_dim
        self.time_length = time_length
        self.model = model(latent_dim)
        self.output_size=x_data_dim
        self.latent_dim=latent_dim

            

    def forward(self, x):
        
        assert len(x.shape) == 3, "Input should have shape: [time_length, batch_size, data_dim]"
        x = tf.identity(x)  # in case x is not a Tensor already...
        
        x = tf.reshape(x, [self.time_length, -1, int(np.sqrt(0.0 + self.x_data_dim)), int(np.sqrt(0.0 + self.x_data_dim))])
        
        y_hat = self.model(x) #(1, TL, d)

        return y_hat
    

            
# Perception_model
class Perception(tf.keras.Model):
    def __init__(self, output_size, **kwargs):
        """ Decoder parent class with no specified output distribution
            :param output_size: output dimensionality
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
        """
        super(Perception, self).__init__()
        self.net =  perception_network(output_size)

    def __call__(self, x, training = None):
        mapped = self.net(x)
        
        return mapped
    
class ML_perception(tf.keras.Model):
    def __init__(self, x_data_dim, y_data_dim, time_length,window_length,
                  model=Perception,**kwargs):
        """ Basic Variational Autoencoder with Standard Normal prior
            :param latent_dim: latent space dimensionality
            :param data_dim: original data dimensionality
            :param time_length: time series duration
            
        """
        super(ML_perception, self).__init__()

        self.x_data_dim = x_data_dim
        self.y_data_dim = y_data_dim
        self.time_length = time_length
        self.window_length = window_length

        self.model = model(y_data_dim)
        self.output_size=y_data_dim
        
    def __call__(self, inputs):
        return self.model(inputs) + 1e-4
            
    def forward(self, x):
        
        assert len(x.shape) == 2, "Input should have shape: [batch_size, data_dim]"
        x = tf.identity(x)  # in case x is not a Tensor already...
        
        y_hat = self.model(x) #(1, TL, d)

        return y_hat

    def _compute_loss(self, x, y, return_parts=False):
        assert len(x.shape) == 2, "Input should have shape: [time_length, batch_size, data_dim]"
        x = tf.identity(x)  # in case x is not a Tensor already...
        x = tf.reshape(x, [1, self.window_length*self.x_data_dim])

        y_hat = (self.model(x)) #(1, TL, d)
        
        loss = tf.reduce_sum(tf.math.square(y_hat- y), axis = [0,1])

        if return_parts:
            return loss, y_hat
        else:
            return y_hat

    def compute_loss(self, x, y, return_parts=False):
        return self._compute_loss(x,y, return_parts=return_parts)
        
    def build_model(self):
        self._compute_loss(tf.random.normal(shape=( 1,  270), dtype=tf.float32),
                          tf.zeros(shape=(1, 2500), dtype=tf.float32))
        return 0

