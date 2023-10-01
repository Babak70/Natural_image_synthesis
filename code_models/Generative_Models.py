import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np
from vae_utils import reduce_logmeanexp
from tensorflow.keras.losses import Poisson as Poisson

from my_network import forward_model_base, encoder_disjoint_cnn, decoder_disjoint_cnn, encoder_joint_cnn_base, decoder_joint_cnn_base

from metrics import fraction_of_explained_variance,correlation_coefficient
from matplotlib import pyplot as plt
import time
import tikzplotlib 
from gp_kernel import *

"""
code inspired by  Fortuin, Vincent, et al., PMLR, 2020.
"""

class DiagonalEncoder(tf.keras.Model):
    def __init__(self, z_size,**kwargs):
        """ Encoder with factorized Normal posterior over temporal dimension
            :param z_size: latent space dimensionality

        """
        super(DiagonalEncoder, self).__init__()
        self.z_size = int(z_size)

        self.net = encoder_disjoint_cnn(z_size)

    def __call__(self, x, training = None):
        
        mapped = self.net(x)
        loc=mapped[..., :self.z_size]
        logvar=mapped[..., self.z_size:]
        scale_diag=tf.math.exp(logvar*0.5)
        
        return tfd.MultivariateNormalDiag(
          loc=loc,
          scale_diag=scale_diag)
        # return tf.split(mapped,2,axis=1)
        
class JointEncoder(tf.keras.Model):
    def __init__(self, z_size, transpose=False, **kwargs):
        """ Encoder with factorized Normal posterior over temporal dimension
            :param z_size: latent space dimensionality

        """
        super(JointEncoder, self).__init__()
        self.z_size = int(z_size)
        self.net = encoder_joint_cnn_base(z_size)
        self.transpose = transpose


    def __call__(self, x, training = None, return_parts = False):
        
        mapped = self.net(x)
        if self.transpose:
            
            num_dim = len(x.shape.as_list())
            perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
            mapped = tf.transpose(mapped, perm=perm)
            loc = mapped[..., :self.z_size, :]
            logvar = mapped[..., self.z_size:, :]
            scale_diag = tf.math.exp(logvar*0.5)
            if return_parts:
                return  tfd.MultivariateNormalDiag(loc=loc,scale_diag=scale_diag),loc, scale_diag
            else:
                return tfd.MultivariateNormalDiag(loc=loc,scale_diag=scale_diag)
        
        else:
            
            
            loc = mapped[..., :self.z_size]
            logvar = mapped[..., self.z_size:]
            scale_diag = tf.math.exp(logvar*0.5)
            if return_parts:
                return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag),loc, scale_diag
            else:
                return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)


class BandedJointEncoder(tf.keras.Model):
    def __init__(self, z_size, **kwargs):
        """ Encoder with 1d-convolutional network and multivariate Normal posterior
            :param z_size: latent space dimensionality

        """
        
        super(BandedJointEncoder, self).__init__()
        self.z_size = int(z_size)
        self.net = encoder_joint_cnn_base(z_size, multiplier=3)


    def __call__(self, x, return_parts =False):
        mapped = self.net(x) #(BS,time_length, data_dim) 

        batch_size = mapped.shape.as_list()[0]
        time_length = mapped.shape.as_list()[1]

        # Obtain mean and precision matrix components
        num_dim = len(mapped.shape.as_list())
        perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
        mapped_transposed = tf.transpose(mapped, perm=perm)
        mapped_mean = mapped_transposed[:, :self.z_size]
        mapped_covar = mapped_transposed[:, self.z_size:]


        mapped_reshaped = tf.reshape(mapped_covar, [batch_size, self.z_size, 2*time_length])

        dense_shape = [batch_size, self.z_size, time_length, time_length]
        idxs_1 = np.repeat(np.arange(batch_size), self.z_size*(2*time_length-1))
        idxs_2 = np.tile(np.repeat(np.arange(self.z_size), (2*time_length-1)), batch_size)
        idxs_3 = np.tile(np.concatenate([np.arange(time_length), np.arange(time_length-1)]), batch_size*self.z_size)
        idxs_4 = np.tile(np.concatenate([np.arange(time_length), np.arange(1,time_length)]), batch_size*self.z_size)
        idxs_all = np.stack([idxs_1, idxs_2, idxs_3, idxs_4], axis=1)

        # ~10x times faster on CPU then on GPU
        with tf.device('/cpu:0'):
            # Obtain covariance matrix from precision one
            mapped_values = tf.reshape(mapped_reshaped[:, :, :-1], [-1])
            prec_sparse = tf.sparse.SparseTensor(indices=idxs_all, values=mapped_values, dense_shape=dense_shape)
            prec_sparse = tf.sparse.reorder(prec_sparse)
            prec_tril = tf.sparse.add(tf.zeros(prec_sparse.dense_shape, dtype=tf.float32), prec_sparse)
            eye = tf.eye(num_rows=prec_tril.shape.as_list()[-1], batch_shape=prec_tril.shape.as_list()[:-2])
            prec_tril = prec_tril + eye
            cov_tril = tf.linalg.triangular_solve(matrix=prec_tril, rhs=eye, lower=False)
            cov_tril = tf.where(tf.math.is_finite(cov_tril), cov_tril, tf.zeros_like(cov_tril))

        num_dim = len(cov_tril.shape)
        perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
        cov_tril_lower = tf.transpose(cov_tril, perm=perm)
        z_dist = tfd.MultivariateNormalTriL(loc=mapped_mean, scale_tril=cov_tril_lower)
        
        if return_parts:
            
            return z_dist,mapped_mean, cov_tril_lower
        else:
            return z_dist

# Decoders

class Decoder(tf.keras.Model):
    def __init__(self, **kwargs):
        """ Decoder parent class with no specified output distribution
            :param output_size: output dimensionality
        """
        super(Decoder, self).__init__()

    def __call__(self, x, training = None):
        pass


class PoissonDecoder_together(Decoder):
    """ Decoder with Gaussian output distribution (used for SPRITES and Physionet) """
    
    def __call__(self, x):
        mapped = self.net(x)
        rate = self.net2(mapped)
        return rate, mapped    
    #     return tfd.Poisson(
    # rate=rate, log_rate=None,
    # interpolate_nondiscrete=True, validate_args=False, allow_nan_stats=False,
    # name='Poisson')
    
class PoissonDecoder_FwdOnly(Decoder):
    def __init__(self, output_size, **kwargs):

        super(PoissonDecoder_FwdOnly, self).__init__()
        self.net =  forward_model_base(output_size)
        self.output_size = output_size
    
    def __call__(self, x):
        
        return self.net(x)
        

class PoissonDecoder_DecodeOnly(Decoder):
    def __init__(self, **kwargs):
        
        super(PoissonDecoder_DecodeOnly, self).__init__()
        self.net = decoder_joint_cnn_base()
        # self.net = decoder_cnn()
        
    def __call__(self, x):
        
        return self.net(x)

# VAE models

class VAE(tf.keras.Model):
    def __init__(self, latent_dim, x_data_dim, y_data_dim, time_length,
                  encoder=JointEncoder,
                  decoder_decode=PoissonDecoder_DecodeOnly,
                  decoder_fwd=PoissonDecoder_FwdOnly,
                  beta=1.0, M=1, K=1, **kwargs):
        """ Basic IB-Disjoint model with Standard Normal prior
            :param latent_dim: latent space dimensionality
            :param data_dim: original data dimensionality
            :param time_length: time series duration
            :param encoder_sizes: layer sizes for the encoder network
            :param decoder: decoder model class {Bernoulli, Gaussian}Decoder
            :param beta: tradeoff coefficient between reconstruction and KL terms in ELBO
            :param M: number of Monte Carlo samples for ELBO estimation
            :param K: number of importance weights for IWAE model (see: https://arxiv.org/abs/1509.00519)
        """
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.x_data_dim = x_data_dim
        self.y_data_dim = y_data_dim
        self.time_length = time_length


        self.encoder = encoder(latent_dim)
        self.decoder_decode = decoder_decode()
        self.decoder_fwd = decoder_fwd(y_data_dim)

        self.beta = beta
        self.K = K
        self.M = M
        

    def encode(self, x, return_parts=False):
        
        x = tf.identity(x)  # in case x is not a Tensor already...
        if return_parts:
            qz_x, loc, scale_diag=self.encoder(x, return_parts=return_parts)
            return qz_x, loc, scale_diag
        else:
            return self.encoder(x, return_parts=return_parts)

    def decode(self, z, return_parts=None):
        z = tf.identity(z)  # in case z is not a Tensor already...
        mapped = self.decoder_decode(z)
        rate = self.decoder_fwd(mapped)
        if return_parts==True:
            return rate, mapped
        else:
            return rate
    def decode_fwd(self, x_hat):
        x_hat = tf.identity(x_hat)  # in case z is not a Tensor already...
        return self.decoder_fwd(x_hat)

        
    def decode_decode(self, z):
        z = tf.identity(z)  # in case z is not a Tensor already...
        return self.decoder_decode(z)

    def __call__(self, inputs):
        return self.decode(self.encode(inputs).sample()) + 1e-4

    def generate(self, noise=None, num_samples=1):
        if noise is None:
            noise = tf.random_normal(shape=(num_samples, self.latent_dim))
        return self.decode(noise)
    
    def _get_prior(self):

        self.prior = tfd.MultivariateNormalDiag(loc=tf.zeros(self.latent_dim, dtype=tf.float32),
                                                    scale_diag=tf.ones(self.latent_dim, dtype=tf.float32))
        return self.prior
    
    def reparameterize(self, mean, logvar,eps=None):
        if eps== None:
                eps = tf.random.normal(shape=mean.shape)
        # return tf.cast(eps *tf.math.log(1 + tf.exp(logvar * .5)),dtype=tf.float32) + mean # modified a bit for stability 
        return eps *tf.exp(logvar * .5) + mean # modified a bit for stability 

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)
    def my_poisson(self, y_true,y_pred):
        
        return y_pred-tf.math.multiply(y_true,tf.math.log(y_pred))
    
    
    def nb_loss(self,y_true, y_pred):#!!! just copy pasted. Have not checked it 
        eps = 1e-10
        y_true = tf.squeeze(y_true, axis=2)
        y_pred = y_pred[:, :]
    
        alpha = y_pred[:, :, 0]
        alpha = 1. / (alpha + eps)
    
        mu = y_pred[:, :, 1]
    
        t1 = -tf.lgamma(y_true + alpha + eps)
        t2 = tf.lgamma(alpha + eps)
        t3 = tf.lgamma(y_true + 1.0) 
    
        t4 = -(alpha * (tf.log(alpha + eps)))
        t5 = -(y_true * (tf.log(mu + eps)))
        t6 = (alpha + y_true) * tf.log(alpha + mu + eps)
    
        loss = t1 + t2 + t3 + t4 + t5 + t6
    
        return tf.reduce_mean(loss)



    def compute_mse(self, x, y=None, m_mask=None, binary=False):
        # Used only for evaluation
        assert len(x.shape) == 3, "Input should have shape: [time_length, batch_size, data_dim]"
        if y is None: y = x

        z_mean = self.encode(x).mean()
        x_hat_mean = self.decode(z_mean).mean()  # shape=(BS, TL, D)
        if binary:
            x_hat_mean = tf.round(x_hat_mean)
            
        mse = tf.math.squared_difference(x_hat_mean, y)
        if m_mask is not None:
            m_mask = tf.cast(m_mask, tf.bool)
            mse = tf.where(m_mask, mse, tf.zeros_like(mse))  # !!! inverse mask, set zeros for observed
        return tf.reduce_sum(mse)

    def _compute_loss(self, x, y, return_parts=False, mask = None):
        assert len(x.shape) == 3, "Input should have shape: [time_length, batch_size, data_dim]"
        x = tf.identity(x)  # in case x is not a Tensor already...
        x = tf.tile(x, [self.M * self.K, 1, 1])  # shape=(M*K*BS, TL, D) has to be shape=(1, TL, D)


        pz = self._get_prior()
        qz_x = self.encode(x) 
        z = qz_x.sample() #(1, TL or d, latent_dim)
        if mask == None:
            mask = tf.ones_like(z)
        z = tf.math.multiply(z, mask)
            
        px_z, stimuli_tnsf = self.decode(z, return_parts=True) #(1, TL, d)
        px_z += 1e-4
        TV_loss = tf.reduce_sum(tf.image.total_variation(tf.transpose(stimuli_tnsf,perm=[0,2,3,1])))

        nll = self.my_poisson(y, px_z) # shape=(1,T,d)
        nll = tf.reduce_sum(nll,axis=2) # shape=(1,T)
        nll = tf.reduce_sum(nll, axis=1) # scalar
        stimuli_tnsf = tf.reshape(stimuli_tnsf, [ self.time_length, -1,self.x_data_dim])
        
        mse = tf.reduce_sum(tf.transpose(tf.math.square(stimuli_tnsf- x),[1,0,2]), axis = [1, 2])
        mse_y = tf.reduce_sum(tf.math.square(px_z- y), axis = [1, 2])

        # nll = -px_z.log_prob(y)  # shape=(M*K*BS, TL, D)
        # nll = tf.where(tf.math.is_finite(nll), nll, tf.zeros_like(nll))
        # nll = tf.reduce_sum(nll, [1, 2])  # shape=(M*K*BS)
        
        
        # y_hat = px_z.sample()  # shape=(M*K*BS, TL, D)
        # y_hat = tf.where(tf.math.is_finite(y_hat), y_hat, tf.zeros_like(y_hat))
        
        # corr=(correlation_coefficient(tf.squeeze(y),tf.squeeze(y_hat)))
        # nll = -tf.math.log((1.+ corr)/2.)
        # nll = tf.reduce_sum(nll, [1])  # shape=(M*K*BS)
        
        if self.K > 1:
            kl = qz_x.log_prob(z) - pz.log_prob(z)  # shape=(M*K*BS, TL or d)
            kl = tf.where(tf.is_finite(kl), kl, tf.zeros_like(kl))
            kl = tf.reduce_sum(kl, 1)  # shape=(M*K*BS)

            weights = -nll - kl  # shape=(M*K*BS)
            weights = tf.reshape(weights, [self.M, self.K, -1])  # shape=(M, K, BS)

            elbo = reduce_logmeanexp(weights, axis=1)  # shape=(M, 1, BS)
            elbo = tf.reduce_mean(elbo)  # scalar
        else:
            ## if K==1, compute KL analytically, analytical gives lagged convergence as compared to sampled KL
            kl = self.kl_divergence(qz_x, pz)  # shape=(1, TL or d)
            # kl = qz_x.log_prob(z) - pz.log_prob(z)  # shape=(1, TL or d)
            kl = tf.where(tf.math.is_finite(kl), kl, tf.zeros_like(kl))
            kl = tf.reduce_sum(kl, 1)  # scalar


            elbo = -nll*self.beta -1.*kl -mse*0.0 # shape=(1) K=1
            elbo = tf.reduce_mean(elbo)  # scalar

        if return_parts:
            nll = tf.reduce_mean(nll)  # scalar
            kl = tf.reduce_mean(kl)  # scalar
            mse = tf.reduce_mean(mse)  # scalar
# Note: normalization of KL is erroneous because it needs to be divided by the number of posterior learnable covariance matrix. Here it is luckiliy true as 2*latent_time=time_length
            return -elbo, nll, kl, mse, nll/(self.time_length*self.decoder_fwd.output_size), kl/(self.time_length*self.latent_dim) ,  mse/ (self.time_length* self.x_data_dim), px_z
        else:
            return -elbo, px_z


    def _compute_loss_FwdOnly(self, x, y, return_parts=False):
        assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
        x = tf.identity(x)  # in case x is not a Tensor already...
        x = tf.reshape(x, [self.time_length, -1, int(tf.math.sqrt(0.0 + self.x_data_dim)), int(tf.math.sqrt(0.0 + self.x_data_dim))])
        px_z = self.decode_fwd(x) # shape=(BS, TL, D)
        px_z=px_z +1e-4
        nll=Poisson(tf.keras.losses.Reduction.NONE)(y,px_z)*9 # shape=(M*K*BS)
        nll = tf.reduce_sum(nll, 1)


        loss = tf.reduce_mean(nll)  # scalar

        if return_parts:
            nll = tf.reduce_mean(nll)  # scalar
            return loss, nll, nll/self.decoder_fwd.output_size, px_z
        else:
            return loss


    def compute_loss(self, x, y, return_parts=False, FwdOnly = False, mask =None):
        
        if FwdOnly == True:
            return self._compute_loss_FwdOnly(x,y, return_parts=return_parts)
        else:
            return self._compute_loss(x,y, return_parts=return_parts, mask=mask)

    def kl_divergence(self, a, b):
        return tfd.kl_divergence(a, b)
    
    def build_model(self):
        self._compute_loss(tf.random.normal(shape=(self.time_length, 1,  self.x_data_dim), dtype=tf.float32),
                          tf.zeros(shape=(1, self.time_length, self.y_data_dim), dtype=tf.float32))
        return 0

    def cumulative_latent(self,x, y, batch_num):

        corr_keep =np.ndarray((1,self.latent_dim))
        nll_keep =np.ndarray((1,self.latent_dim))
        kl_traverse = np.ndarray((1,self.latent_dim))

        assert len(x.shape) == 3, "Input should have shape: [time_length, batch_size, data_dim]"
        x = tf.identity(x)  # in case x is not a Tensor already...
        y=tf.squeeze(y)
        qz_x,loc,scale_diag = self.encode(x, return_parts=True) 
        z = qz_x.sample() #(1, TL or d, latent_dim)
        px_z_orig= self.decode(z,return_parts=False) #(1, TL, d)
        px_z_orig+=1e-4
        nll = self.my_poisson(y, px_z_orig) # shape=(1,T,d)
        nll = tf.reduce_sum(nll,axis=2) # shape=(1,T)
        nll = tf.reduce_sum(nll, axis=1) # scalar
        nll = tf.reduce_mean(nll)  # scalar
        nll_orig = nll
        px_z_orig=tf.squeeze(px_z_orig)
        
        corr_orig=tf.reduce_mean(correlation_coefficient(y,px_z_orig))
        z=tf.squeeze(z)
        # pz = self._get_prior()
        kl_orig_l=[]
        for i in range(self.latent_dim):
            qzl_x = tfd.MultivariateNormalDiag(loc=loc[...,i][...,tf.newaxis],scale_diag=scale_diag[...,i][...,tf.newaxis])
            pzl= tfd.MultivariateNormalDiag(loc=tf.zeros(1, dtype=tf.float32),
                                                    scale_diag=tf.ones(1, dtype=tf.float32))
            temp = self.kl_divergence(qzl_x, pzl)
            temp=tf.where(tf.math.is_finite(temp), temp, tf.zeros_like(temp))
            temp=tf.reduce_sum(temp,axis=1)
            kl_orig_l.append(temp)
        kl_orig_all = tf.concat( kl_orig_l, axis=0)
        kl_orig = tf.reduce_sum(kl_orig_all, 0)  # scalar

        
        kl_sorted_args = tf.squeeze(tf.argsort(kl_orig_all, axis=-1, direction='DESCENDING'))
        kl_sorted_vals = tf.squeeze(tf.sort(kl_orig_all, axis=-1, direction='DESCENDING'))
        
        
        for i in range(self.latent_dim):

                keep_indx = kl_sorted_args.numpy()[:(i+1)]
                temp = z.numpy()
                keep_mask = np.zeros_like(z.numpy(),dtype=np.float32)
                keep_mask[:,keep_indx]=1.0
                z_ = tf.math.multiply(z, keep_mask)
                px_z_keep = self.decode(z_[tf.newaxis,...],return_parts=False) #(1, TL, d)
                
                px_z_keep+=1e-4
                nll = self.my_poisson(y, px_z_keep) # shape=(1,T,d)
                nll = tf.reduce_sum(nll,axis=2) # shape=(1,T)
                nll = tf.reduce_sum(nll, axis=1) # scalar
                nll = tf.reduce_mean(nll)  # scalar
                nll_keep[0,i] = nll/(self.time_length*self.decoder_fwd.output_size)
                px_z_keep=tf.squeeze(px_z_keep)
                corr_keep[0,i]=tf.reduce_mean(correlation_coefficient(y,px_z_keep))
                
                
                
        return corr_orig, np.squeeze(corr_keep), nll_orig, np.squeeze(nll_keep)
    
    
    def traverse(self,x, y, batch_num, return_corr_keep=False):
        corr_traverse =np.ndarray((7,self.latent_dim))
        kl_traverse = np.ndarray((1,self.latent_dim))
        plot_time_len = 100
        x_plot = np.arange(0, self.time_length/100, 0.01)
        
        
        assert len(x.shape) == 3, "Input should have shape: [time_length, batch_size, data_dim]"
        x = tf.identity(x)  # in case x is not a Tensor already...
        y=tf.squeeze(y)
        qz_x,loc,scale_diag = self.encode(x, return_parts=True) 
        z = qz_x.sample() #(1, TL or d, latent_dim)
        px_z_orig= self.decode(z,return_parts=False) #(1, TL, d)
        px_z_orig+=1e-4
        px_z_orig=tf.squeeze(px_z_orig)
        corr_orig=tf.reduce_mean(correlation_coefficient(y,px_z_orig))
        z=tf.squeeze(z)
        # pz = self._get_prior()
        kl_orig_l=[]
        for i in range(self.latent_dim):
            qzl_x = tfd.MultivariateNormalDiag(loc=loc[...,i][...,tf.newaxis],scale_diag=scale_diag[...,i][...,tf.newaxis])
            pzl= tfd.MultivariateNormalDiag(loc=tf.zeros(1, dtype=tf.float32),
                                                    scale_diag=tf.ones(1, dtype=tf.float32))
            temp = self.kl_divergence(qzl_x, pzl)
            temp=tf.where(tf.math.is_finite(temp), temp, tf.zeros_like(temp))
            temp=tf.reduce_sum(temp,axis=1)
            kl_orig_l.append(temp)
        kl_orig_all = tf.concat( kl_orig_l, axis=0)
        kl_orig = tf.reduce_sum(kl_orig_all, 0)  # scalar
        kl_orig = kl_orig
        
        kl_sorted_args = tf.squeeze(tf.argsort(kl_orig_all, axis=-1, direction='DESCENDING'))
        kl_sorted_vals = tf.squeeze(tf.sort(kl_orig_all, axis=-1, direction='DESCENDING'))
        
        
        np.savetxt('./figures/traverse/batch-{}/KL-args.txt'.format(batch_num), kl_sorted_args.numpy(), delimiter=',')
        np.savetxt('./figures/traverse/batch-{}/KL-vals.txt'.format(batch_num), kl_sorted_vals, delimiter=',')
        
        plt.figure()
        plt.plot(kl_sorted_args.numpy(),kl_sorted_vals.numpy(),'*')
        plt.xlabel('latent dim')
        plt.ylabel('kl (non-normalized)')
        plt.savefig('./figures/traverse/batch-{}/KL'.format(batch_num))
        plt.close('all')
        z_sampled = tfd.MultivariateNormalDiag(loc=tf.zeros(self.time_length//2, dtype=tf.float32),
                                                    scale_diag=tf.ones(self.time_length//2, dtype=tf.float32)).sample()
        
        
        variation = 5
        num_latent = 2
        fig, axs = plt.subplots(num_latent, variation, figsize=(10,4), sharex=True, sharey=True, tight_layout = True)
        for i in range(num_latent):
            k = kl_sorted_args[tf.convert_to_tensor(i)]
            
            for j in range(variation):
                  
                  temp = z.numpy()
                  temp[:,k] = (j-variation//2)*z_sampled
                  z_= temp
                  px_z_ablated = self.decode(z_[tf.newaxis,...],return_parts=False) #(1, TL, d)
                  px_z_ablated += 1e-4
                  px_z_ablated = tf.squeeze(px_z_ablated)
                  corr = tf.reduce_mean(correlation_coefficient(y,px_z_ablated))
                  

                  slice_i = slice((0)*plot_time_len,(0+1)*plot_time_len)
                  slice_i = slice(0,1000)
                  axs[i,j].plot(x_plot[slice_i], y[:,0].numpy()[slice_i], lw=2, label="true rate", color = '#808080')
                  # axs[i,j].plot(px_z_orig[:,0].numpy(), label="orig. response")
                  axs[i,j].plot(x_plot[slice_i],px_z_ablated[:,0].numpy()[slice_i],lw=2, label="traverse response", color = '#008000', alpha=.8)
                  # axs[i,j].set_ylabel("rate [spk/s]")
                  # axs[i,j].text(0.8, 0.8, ' corr_orig: {:.3f}, corr {:.3f}, kl {:.3f}'.format(round(corr_orig.numpy(),2),round(corr.numpy(),2), round(kl_orig_all[0,i].numpy(),2)),
                  #                horizontalalignment='center', verticalalignment='center', transform=axs[i,j].transAxes)
                  axs[i,j].tick_params(axis='x', which='major', labelsize=10)
                  axs[i,j].tick_params(axis='y', which='major', labelsize=10)
        fig.text(0.5, -0.001, 'Time (s)', ha='center')
        fig.text(-0.001, 0.5, 'Rate (spikes/s)', va='center', rotation='vertical')
        fig.tight_layout()
        plt.show()
        fig.savefig('./figures/traverse/batch-{}/traverse-all.png'.format(batch_num))
        tikzplotlib.save('./figures/traverse/batch-{}/traverse-all.tex'.format(batch_num))

        for i in range(self.latent_dim):
            k = kl_sorted_args[tf.convert_to_tensor(i)]

            for j in range(-3,4):
                  
                  temp=z.numpy()
          
                  temp[:,k]=j*z_sampled
                  # z_=tf.convert_to_tensor(temp)
                  z_= temp
                  px_z_ablated = self.decode(z_[tf.newaxis,...],return_parts=False) #(1, TL, d)
                  px_z_ablated+=1e-4
                  px_z_ablated=tf.squeeze(px_z_ablated)
                  
                  corr=tf.reduce_mean(correlation_coefficient(y,px_z_ablated))
                  corr_traverse [j,k] = corr
                  # # plotting
                  plt.figure()
                  plt.plot(y[:,0].numpy(), '-k', lw=3, label="true rate")
                  plt.plot(px_z_orig[:,0].numpy(), 'b' , lw=2, label="orig. response")
                  plt.plot(px_z_ablated[:,0].numpy(), '--g',  lw=2, label="traverse response")
                  
                  
                  if    (j==-3):
                      if i==self.latent_dim-1:
                          1
                      
                      elif i==0:
                          plt.legend()
                      else:
                          plt.xticks([], [])
                      
                  elif (i==self.latent_dim-1) and (j!=-3):
                          plt.yticks([], [])
                  else:
                          plt.xticks([], [])
                          plt.yticks([], [])
                  
                  plt.savefig('./figures/traverse/batch-{}/image {}.png'.format(batch_num,(7*(i)+j+4)))
                  plt.show()
                  time.sleep(.1)


        return corr_traverse , corr_orig, tf.squeeze(kl_orig_all)
    
    def forward(self, x , y = None, return_loss=None, noise=None, return_param=False, mask=None):
        
        assert len(x.shape) == 3, "Input should have shape: [time_length, batch_size, data_dim]"
        x = tf.identity(x)  # in case x is not a Tensor already...
        
        if return_param:
            qz_x, loc,scale = self.encode(x, return_parts=True) 
        else:
            qz_x = self.encode(x) 
            
        z = qz_x.sample() #(1, TL or d, latent_dim)
        
        if mask == None:
            mask = tf.ones_like(z)
        z = tf.math.multiply(z, mask)
        
        px_z, tsnf_stimuli = self.decode(z,return_parts=True) #(1, TL, d)
        px_z += 1e-4
        
        if return_param:
            return px_z, z, loc, scale
        
        
        if return_loss == True:

            nll = self.my_poisson(y, px_z) # shape=(1,T,d)
            nll = tf.reduce_sum(nll,axis=2) # shape=(1,T)
            nll = tf.reduce_sum(nll, axis=1) # scalar
            nll = tf.reduce_mean(nll)  # scalar
            stimuli_tnsf = tf.transpose(tsnf_stimuli, perm=[1, 0, 2, 3])
            stimuli_tnsf = tf.reshape(stimuli_tnsf, [ self.time_length, -1,self.x_data_dim])
            mse = tf.reduce_sum(tf.transpose(tf.math.square(stimuli_tnsf- x),[1,0,2]), axis = [1, 2])
            mse = tf.reduce_mean(mse) # scalar
            
            return  nll/(self.time_length*self.decoder_fwd.output_size),  px_z, z, tsnf_stimuli
        else:

            return px_z, z, tsnf_stimuli
        
    def mutual_info(self, x, y):
            assert len(x.shape) == 3, "Input should have shape: [time_length, batch_size, data_dim]"
            x = tf.identity(x)  # in case x is not a Tensor already...
            x = tf.tile(x, [self.M * self.K, 1, 1])  # shape=(M*K*BS, TL, D) has to be shape=(1, TL, D)
    
    
            pz = self._get_prior()
            qz_x = self.encode(x) 
            z = qz_x.sample() #(1, TL or d, latent_dim)
            px_z, stimuli_tnsf = self.decode(z, return_parts=True) #(1, TL, d)
            px_z += 1e-4
    
            nll = self.my_poisson(y, px_z) # shape=(1,T,d)
            nll = tf.reduce_sum(nll,axis=2) # shape=(1,T)
            nll = tf.reduce_sum(nll, axis=1) # scalar
            stimuli_tnsf = tf.reshape(stimuli_tnsf, [ self.time_length, -1,self.x_data_dim])
            mse = tf.reduce_sum(tf.transpose(tf.math.square(stimuli_tnsf- x),[1,0,2]), axis = [1, 2])
            mse_y = tf.reduce_sum(tf.math.square(px_z- y), axis = [1, 2])

            if self.K > 1:
                kl = qz_x.log_prob(z) - pz.log_prob(z)  # shape=(M*K*BS, TL or d)
                kl = tf.where(tf.is_finite(kl), kl, tf.zeros_like(kl))
                kl = tf.reduce_sum(kl, 1)  # shape=(M*K*BS)
    
                weights = -nll - kl  # shape=(M*K*BS)
                weights = tf.reshape(weights, [self.M, self.K, -1])  # shape=(M, K, BS)
    
                elbo = reduce_logmeanexp(weights, axis=1)  # shape=(M, 1, BS)
                elbo = tf.reduce_mean(elbo)  # scalar
            else:
                ## if K==1, compute KL analytically, analytical gives lagged convergence as compared to sampled KL
                kl = self.kl_divergence(qz_x, pz)  # shape=(1, TL or d)
                # kl = qz_x.log_prob(z) - pz.log_prob(z)  # shape=(1, TL or d)
                kl = tf.where(tf.math.is_finite(kl), kl, tf.zeros_like(kl))
                kl = tf.reduce_sum(kl, 1)  # scalar
    
            nll = tf.reduce_mean(nll)  # scalar
            kl = tf.reduce_mean(kl)  # scalar
            mse = tf.reduce_mean(mse)  # scalar
 # Note: normalization of KL is erroneous because it needs to be divided by the number of posterior learnable covariance matrix. Here it is luckiliy true as 2*latent_time=time_length
            return  -nll, kl,  px_z


class GP_VAE(VAE):
    def __init__(self, *args, latent_time_length = 1000, kernel="cauchy", sigma=1., length_scale=1.0, kernel_scales=1, **kwargs):
        """ Proposed IB-GP model with Gaussian Process prior
            :param kernel: Gaussial Process kernel ["cauchy", "diffusion", "rbf", "matern"]
            :param sigma: scale parameter for a kernel function
            :param length_scale: length scale parameter for a kernel function
            :param kernel_scales: number of different length scales over latent space dimensions
        """
        super(GP_VAE, self).__init__(*args, **kwargs)
        self.kernel = kernel
        self.sigma = sigma
        self.length_scale = length_scale
        self.kernel_scales = kernel_scales
        self.latent_time_length = latent_time_length

        if isinstance(self.encoder, JointEncoder):
            self.encoder.transpose = True

        # Precomputed KL components for efficiency
        self.pz_scale_inv = None
        self.pz_scale_log_abs_determinant = None
        self.prior = None

    def decode(self, z, return_parts=None):
        z = tf.identity(z)  # in case z is not a Tensor already...
        num_dim = len(z.shape)
        assert num_dim > 2
        perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
        mapped = self.decoder_decode(tf.transpose(z, perm=perm))
        rate = self.decoder_fwd(mapped)
        if return_parts==True:
            return rate, mapped
        else:
            return rate
    
    def decode_decode(self, z):
        z = tf.identity(z)  # in case z is not a Tensor already...
        num_dim = len(z.shape)
        assert num_dim > 2
        perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
        return self.decoder_decode(tf.transpose(z, perm=perm))
    
    
    def _get_prior(self):
        
        if self.prior is None:
            
            # Compute kernel matrices for each latent dimension
            kernel_matrices=[]
            
            for i in range(self.kernel_scales):
                if self.kernel == "rbf":
                    kernel_matrices.append(rbf_kernel(self.latent_time_length, self.length_scale / 2**i))
                elif self.kernel == "diffusion":
                    kernel_matrices.append(diffusion_kernel(self.latent_time_length, self.length_scale / 2**i))
                elif self.kernel == "matern":
                    kernel_matrices.append(matern_kernel(self.latent_time_length, self.length_scale / 2**i))
                elif self.kernel == "cauchy":
                    kernel_matrices.append(cauchy_kernel(self.latent_time_length, self.sigma, self.length_scale / 2**i))

            # Combine kernel matrices for each latent dimension
            tiled_matrices = []
            total = 0
            for i in range(self.kernel_scales):
                if i == self.kernel_scales-1:
                    multiplier = self.latent_dim - total
                else:
                    multiplier = int(tf.math.ceil(self.latent_dim / self.kernel_scales))
                    total += multiplier
                tiled_matrices.append(tf.tile(tf.expand_dims(kernel_matrices[i], 0), [multiplier, 1, 1]))
            kernel_matrix_tiled = tf.concat(tiled_matrices,axis=0)
            assert len(kernel_matrix_tiled) == self.latent_dim
            
            self.prior = tfd.MultivariateNormalFullCovariance(
                loc=tf.zeros([self.latent_dim, self.latent_time_length], dtype=tf.float32),
                covariance_matrix=kernel_matrix_tiled)
        return self.prior
    
    
    def ablation(self, x, y, mask_indx):
        
        assert len(x.shape) == 3, "Input should have shape: [time_length, batch_size, data_dim]"
        x = tf.identity(x)  # in case x is not a Tensor already...

        qz_x = self.encode(x) 
        z = qz_x.sample() #(1, TL or d, latent_dim)
        px_z_orig, tsnf_stimuli_orig = self.decode(z,return_parts=True) #(1, TL, d)
        px_z_orig+=1e-4
        
        z=tf.squeeze(z)
        temp=z.numpy()
        temp[mask_indx,:]=0.
        z_=tf.convert_to_tensor(temp)
        px_z_ablated, tsnf_stimuli_ablated = self.decode(z_[tf.newaxis,...],return_parts=True) #(1, TL, d)
        px_z_ablated+=1e-4
        
        return px_z_orig, px_z_ablated, tsnf_stimuli_orig, tsnf_stimuli_ablated
    
    def cumulative_latent(self,x, y, batch_num):

        corr_keep = np.ndarray((1,self.latent_dim))
        nll_keep =np.ndarray((1,self.latent_dim))
        kl_traverse = np.ndarray((1,self.latent_dim))

        assert len(x.shape) == 3, "Input should have shape: [time_length, batch_size, data_dim]"
        x = tf.identity(x)  # in case x is not a Tensor already...
        y=tf.squeeze(y)
        qz_x ,mapped_mean, cov_tril_lower= self.encode(x,return_parts=True)
        
        z = qz_x.sample() #(1, TL or d, latent_dim)
        px_z_orig= self.decode(z,return_parts=False) #(1, TL, d)
        px_z_orig+=1e-4
        nll = self.my_poisson(y, px_z_orig) # shape=(1,T,d)
        nll = tf.reduce_sum(nll,axis=2) # shape=(1,T)
        nll = tf.reduce_sum(nll, axis=1) # scalar
        nll = tf.reduce_mean(nll)  # scalar
        nll_orig = nll
        px_z_orig=tf.squeeze(px_z_orig)
        corr_orig=tf.reduce_mean(correlation_coefficient(y, px_z_orig))
        z=tf.squeeze(z)
        pz = self._get_prior()
        kl_orig_all =  self.kl_divergence(qz_x, pz)
        kl_orig_all = tf.where(tf.math.is_finite(kl_orig_all), kl_orig_all, tf.zeros_like(kl_orig_all))
        kl_orig = tf.reduce_sum(kl_orig_all, 1)  # scalar
        kl_orig = kl_orig
        
        kl_sorted_args = tf.squeeze(tf.argsort(kl_orig_all, axis=-1, direction='DESCENDING'))
        kl_sorted_vals = tf.squeeze(tf.sort(kl_orig_all, axis=-1, direction='DESCENDING'))

        for i in range(self.latent_dim):
            
         
                
                keep_indx = kl_sorted_args.numpy()[:(i+1)]
                keep_mask = np.zeros_like(z.numpy(),dtype=np.float32)
                keep_mask[keep_indx,:]=1.0
                z_ = tf.math.multiply(z, keep_mask)
                px_z_keep = self.decode(z_[tf.newaxis,...],return_parts=False) #(1, TL, d)
                px_z_keep+=1e-4
                nll = self.my_poisson(y, px_z_keep) # shape=(1,T,d)
                nll = tf.reduce_sum(nll,axis=2) # shape=(1,T)
                nll = tf.reduce_sum(nll, axis=1) # scalar
                nll = tf.reduce_mean(nll)  # scalar
                nll_keep[0,i] = nll/(self.time_length*self.decoder_fwd.output_size)
                px_z_keep=tf.squeeze(px_z_keep)
                corr_keep[0,i]=tf.reduce_mean(correlation_coefficient(y,px_z_keep))
                
        return corr_orig, np.squeeze(corr_keep),nll_orig, np.squeeze(nll_keep)
                
    def traverse(self,x, y,batch_num=0):
        plot_time_len = 100
        x_plot = np.arange(0, self.time_length/100, 0.01)
        corr_traverse = np.ndarray((7,self.latent_dim))
        kl_traverse = np.ndarray((1,self.latent_dim))

        assert len(x.shape) == 3, "Input should have shape: [time_length, batch_size, data_dim]"
        x = tf.identity(x)  # in case x is not a Tensor already...
        y=tf.squeeze(y)
        qz_x ,mapped_mean, cov_tril_lower= self.encode(x,return_parts=True)
        
        z = qz_x.sample() #(1, TL or d, latent_dim)
        px_z_orig= self.decode(z,return_parts=False) #(1, TL, d)
        px_z_orig+=1e-4
        px_z_orig=tf.squeeze(px_z_orig)
        corr_orig=tf.reduce_mean(correlation_coefficient(y, px_z_orig))
        z=tf.squeeze(z)
        pz = self._get_prior()
        kl_orig_all =  self.kl_divergence(qz_x, pz)
        kl_orig_all = tf.where(tf.math.is_finite(kl_orig_all), kl_orig_all, tf.zeros_like(kl_orig_all))
        kl_orig = tf.reduce_sum(kl_orig_all, 1)  # scalar
        kl_orig = kl_orig
        
        kl_sorted_args = tf.squeeze(tf.argsort(kl_orig_all, axis=-1, direction='DESCENDING'))
        kl_sorted_vals = tf.squeeze(tf.sort(kl_orig_all, axis=-1, direction='DESCENDING'))
        
        np.savetxt('./figures/traverse/batch-{}/KL-args.txt'.format(batch_num), kl_sorted_args.numpy(), delimiter=',')
        np.savetxt('./figures/traverse/batch-{}/KL-vals.txt'.format(batch_num), kl_sorted_vals, delimiter=',')
        
        plt.figure()
        plt.plot(kl_sorted_args.numpy(),kl_sorted_vals.numpy(),'*')
        plt.xlabel('latent dim')
        plt.ylabel('kl (non-normalized)')
        plt.savefig('./figures/traverse/batch-{}/KL'.format(batch_num))
        plt.close('all')
        
        neuron_num = 0
        variation = 7
        num_latent = 2
        SCALE = 5
        fig, axs = plt.subplots(num_latent, variation,figsize=(10,2.5), sharex=True, sharey=True, tight_layout = True)
        axs = np.reshape(axs, [ num_latent,variation])
        for i in range(num_latent):
            k = kl_sorted_args[tf.convert_to_tensor(i)]
            z_sampled = pz.sample()[k,:]
            for j in range(variation):
                  
                  temp = z.numpy()
                  temp[k,:] = (j-variation//2)*z_sampled
                  z_= temp
                  px_z_ablated = self.decode(z_[tf.newaxis,...],return_parts=False) #(1, TL, d)
                  px_z_ablated += 1e-4
                  px_z_ablated = tf.squeeze(px_z_ablated)
                  corr = tf.reduce_mean(correlation_coefficient(y,px_z_ablated))
                  

                  slice_i = slice((0)*plot_time_len,(0+1)*plot_time_len)
                  slice_i = slice(0,1000)
                  axs[i,j].plot(x_plot[slice_i], y[:,neuron_num].numpy()[slice_i], lw=2, label="true rate", color = '#808080')
                  # axs[i,j].plot(px_z_orig[:,neuron_num].numpy(), label="orig. response")
                  axs[i,j].plot(x_plot[slice_i],px_z_ablated[:,neuron_num].numpy()[slice_i],lw=2, label="traverse response", color = '#008000', alpha=.8)
                  # axs[i,j].set_ylabel("rate [spk/s]")
                  # axs[i,j].text(0.8, 0.8, ' corr_orig: {:.3f}, corr {:.3f}, kl {:.3f}'.format(round(corr_orig.numpy(),2),round(corr.numpy(),2), round(kl_orig_all[0,i].numpy(),2)),
                  #                horizontalalignment='center', verticalalignment='center', transform=axs[i,j].transAxes)
                  axs[i,j].tick_params(axis='x', which='major', labelsize=10)
                  axs[i,j].tick_params(axis='y', which='major', labelsize=10)
        fig.text(0.5, -0.001, 'Time (s)', ha='center')
        fig.text(-0.001, 0.5, 'Rate (spikes/s)', va='center', rotation='vertical')
        fig.tight_layout()
        # plt.show()
        fig.savefig('./figures/traverse/batch-{}/traverse-all.png'.format(batch_num))
        fig.savefig('./figures/traverse/batch-{}/traverse-all.eps'.format(batch_num),format='eps')
        tikzplotlib.save('./figures/traverse/batch-{}/traverse-all.tex'.format(batch_num))


        for i in range(self.latent_dim):
            k = kl_sorted_args[tf.convert_to_tensor(i)]
            z_sampled = pz.sample()[k,:]
            for j in range(-3,4):
                  
                  temp = z.numpy()
                  temp[k,:] = j*z_sampled
                  z_= temp
                  px_z_ablated = self.decode(z_[tf.newaxis,...],return_parts=False) #(1, TL, d)
                  px_z_ablated += 1e-4
                  px_z_ablated = tf.squeeze(px_z_ablated)
                  corr = tf.reduce_mean(correlation_coefficient(y,px_z_ablated))
                  corr_traverse[j+3,k] = corr
                  
                    # # plotting
                  plt.figure()
                  plt.plot(y[:,0].numpy(), '-k', lw=3, label="true rate")
                  plt.plot(px_z_orig[:,0].numpy(), 'b' , lw=2, label="orig. response")
                  plt.plot(px_z_ablated[:,0].numpy(), '--g',  lw=2, label="traverse response")
                  if    (j==-3):
                        if i==self.latent_dim-1:
                            1
                      
                        elif i==0:
                            plt.legend()
                        else:
                            plt.xticks([], [])
                      
                  elif (i==self.latent_dim-1) and (j!=-3):
                            plt.yticks([], [])
                  else:
                            plt.xticks([], [])
                            plt.yticks([], [])
                  
                  plt.savefig('./figures/traverse/batch-{}/image {}.png'.format(batch_num,(7*(i)+j+4)))
                  plt.show()
                  time.sleep(.1)
                  
        return corr_traverse, corr_orig, tf.squeeze(kl_orig_all)
    
    def kl_divergence(self, a, b):
        """ Batched KL divergence `KL(a || b)` for multivariate Normals.
            See https://github.com/tensorflow/probability/blob/master/tensorflow_probability
                        /python/distributions/mvn_linear_operator.py
            It's used instead of default KL class in order to exploit precomputed components for efficiency
        """

        def squared_frobenius_norm(x):
            """Helper to make KL calculation slightly more readable."""
            return tf.reduce_sum(tf.square(x), axis=[-2, -1])

        def is_diagonal(x):
            """Helper to identify if `LinearOperator` has only a diagonal component."""
            return (isinstance(x, tf.linalg.LinearOperatorIdentity) or
                    isinstance(x, tf.linalg.LinearOperatorScaledIdentity) or
                    isinstance(x, tf.linalg.LinearOperatorDiag))

        if is_diagonal(a.scale) and is_diagonal(b.scale):
            # Using `stddev` because it handles expansion of Identity cases.
            b_inv_a = (a.stddev() / b.stddev())[..., tf.newaxis]
        else:
            if self.pz_scale_inv is None:
                self.pz_scale_inv = tf.linalg.inv(b.scale.to_dense())
                self.pz_scale_inv = tf.where(tf.math.is_finite(self.pz_scale_inv),
                                              self.pz_scale_inv, tf.zeros_like(self.pz_scale_inv))

            if self.pz_scale_log_abs_determinant is None:
                self.pz_scale_log_abs_determinant = b.scale.log_abs_determinant()

            a_shape = a.scale.shape
            if len(b.scale.shape) == 3:
                _b_scale_inv = tf.tile(self.pz_scale_inv[tf.newaxis], [a_shape[0]] + [1] * (len(a_shape) - 1))
            else:
                _b_scale_inv = tf.tile(self.pz_scale_inv, [a_shape[0]] + [1] * (len(a_shape) - 1))

            b_inv_a = _b_scale_inv @ a.scale.to_dense()

        # ~10x times faster on CPU then on GPU
        with tf.device('/cpu:0'):
            kl_div = (self.pz_scale_log_abs_determinant - a.scale.log_abs_determinant() +
                      0.5 * (-tf.cast(a.scale.domain_dimension_tensor(), a.dtype) +
                      squared_frobenius_norm(b_inv_a) + squared_frobenius_norm(
                      b.scale.solve((b.mean() - a.mean())[..., tf.newaxis]))))
        return kl_div
        