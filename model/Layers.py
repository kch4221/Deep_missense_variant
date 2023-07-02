import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

class Conv1(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units
                
    def build(self, input_shape):
        self.layers_ = []
        for i in range(input_shape[-2]):
            self.layers_.append(tf.keras.layers.Dense(self.units))

    def call(self, inputs):
        x = []
        for i in range(inputs.shape[-2]):
            p = tf.reshape(self.layers_[i](inputs[:,i,:]), [inputs.shape[0], 1, self.units])
            x.append(p)
        x = tf.concat(x, axis=1)
        return x


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, n_head = 2):
        super().__init__()
        self.head = n_head
    
    def build(self, input_shape):
        dim = input_shape[-1]
        assert dim%self.head == 0
        
        self.Q = tf.keras.layers.Conv1D(dim, 1, use_bias=False)
        self.K = tf.keras.layers.Conv1D(dim, 1, use_bias=False)
        self.V = tf.keras.layers.Conv1D(dim, 1, use_bias=False)
        self.O = tf.keras.layers.Conv1D(dim, 1, use_bias=False)
        
        
    def split_head(self, x):
        x = tf.reshape(x, [x.shape[0], -1, self.head, x.shape[-1]//self.head])
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return x
    
    
    def call(self, x):
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        
        q = self.split_head(q)
        k = self.split_head(k)
        v = self.split_head(v)
        
        qk = tf.matmul(q, k, transpose_b=True)
        qk = tf.nn.softmax(qk, axis=-1)
        qkv = tf.matmul(qk, v)
        concat = tf.reshape(qkv, x.shape)
        o = self.O(concat)
        return o
 

def PositionCoding(length, dim):
    df = pd.DataFrame(np.zeros([dim, length]))
    for pos in df.columns:
        df[pos] = pos
    for i in df.index:
        if i % 2 == 0:
            df.iloc[i, :] = df.iloc[i, :].apply(lambda pos: np.sin(pos/np.power(10000, i/dim)))
        else:
            df.iloc[i, :] = df.iloc[i, :].apply(lambda pos: np.cos(pos/np.power(10000, (i-1)/dim)))
    return np.transpose(df.values)


class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, n_head):
        super().__init__()
        self.attention = AttentionLayer(n_head)
        
        
    def build(self, input_shapes):
        inp_dim = input_shapes[-1]
        length = input_shapes[-2]
        self.PE = PositionCoding(length, inp_dim)
        self.out = tf.keras.layers.Conv1D(inp_dim, 1)
        
        
    def call(self, inputs, PE=False):
        if PE:
            x = inputs + self.PE
        else:
            x = inputs
        a = self.attention(x)
        x = tf.keras.layers.Add()([a, x])
        x = tf.keras.layers.LayerNormalization(axis = 2, epsilon=1e-6, center=True, scale=True)(x)
        f = self.out(x)
        x = tf.keras.layers.Add()([x, f])
        x = tf.keras.layers.LayerNormalization(axis = 2, epsilon=1e-6, center=True, scale=True)(x)
        return x

    
def KLD_diag_gaussians(mu, log_sigma, prior_mu, prior_log_sigma):
    return prior_log_sigma - log_sigma + 0.5 * (tf.exp(2. * log_sigma) + tf.square(mu - prior_mu)) * tf.exp(-2.0 * prior_log_sigma) - 0.5


class VarianceLayer(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units
        
    def Sampler(self, mu, sigma):
        eps = tf.random.normal(sigma.shape)
        return mu + eps * (tf.exp(sigma)**0.5)    
        
    def build(self, input_shape):
        self.wmu = self.add_variable(name='wmu',shape=[input_shape[-1], self.units],
                                    initializer=tf.random_normal_initializer(0, tf.sqrt(2.0/(input_shape[-1] + self.units))))
        self.wsigma = self.add_variable(name='wsigma',shape=[input_shape[-1], self.units], initializer=tf.constant_initializer(sigma_init))
        self.bmu = self.add_variable(name='bmu',shape=[self.units], initializer=tf.constant_initializer(0.1))
        self.bsigma = self.add_variable(name='bsigma',shape=[self.units], initializer=tf.constant_initializer(sigma_init))
    
    def KLD_params(self):
        res = - tf.reduce_sum(KLD_diag_gaussians(self.wmu, self.wsigma, 0, 0.))
        res -= tf.reduce_sum(KLD_diag_gaussians(self.bmu, self.bsigma, 0, 0.)) 
        return res/neff
    
    
    def call(self, inputs, var=True):     
        w = self.Sampler(self.wmu, self.wsigma)
        b = self.Sampler(self.bmu, self.bsigma)
        if not var:
            w = self.wmu
            b = self.bmu
        output = tf.matmul(inputs, w) + b
        
        return output
    
    
class OutputLayer(layers.Layer):
    def __init__(self, units, seq_len, n_pat = 4):
        super().__init__()
        self.units = units
        self.seq_len = seq_len
        self.n_pat = n_pat
        
        
    def Sampler(self, mu, sigma):
        eps = tf.random.normal(sigma.shape)
        return mu + eps * (tf.exp(sigma)**0.5)    
        
    def build(self, input_shape):
        self.inputshape = input_shape
        self.w_out_mu = self.add_variable(name='w_out_mu',shape=[input_shape[-1] * self.seq_len, self.units], 
                                          initializer=tf.random_normal_initializer(0, tf.sqrt(2.0/(input_shape[-1] * self.seq_len+self.units))))
        self.w_out_sigma = self.add_variable(name='w_out_sigma',shape=[input_shape[-1] * self.seq_len, self.units], 
                                             initializer=tf.constant_initializer(sigma_init))
        
        self.w_conv_mu = self.add_variable(name='w_conv_mu',shape=[self.units, alp_len], 
                                           initializer=tf.random_normal_initializer(0, tf.sqrt(2.0/(self.units*alp_len))))
        self.w_conv_sigma = self.add_variable(name='w_conv_sigma',shape=[self.units, alp_len], 
                                              initializer=tf.constant_initializer(sigma_init))
        
        self.w_scale_mu = self.add_variable(name='w_scale_mu',shape=[int(input_shape[-1] / self.n_pat), self.seq_len], 
                                            initializer=tf.constant_initializer(0.0))
        self.w_scale_sigma = self.add_variable(name='w_scale_sigma',shape=[int(input_shape[-1] / self.n_pat), self.seq_len], 
                                               initializer=tf.constant_initializer(sigma_init))
        
        self.bmu = self.add_variable(name='bmu',shape=[self.seq_len * alp_len], 
                                     initializer=tf.random_normal_initializer(0.1))
        self.bsigma = self.add_variable(name='bsigma',shape=[self.seq_len * alp_len], 
                                        initializer=tf.constant_initializer(sigma_init))
        self.psw_mu = self.add_variable(name='psw_mu', shape=[1], initializer=tf.constant_initializer(1))
        self.psw_sigma = self.add_variable(name='psw_sigma', shape=[1], initializer=tf.constant_initializer(sigma_init))
    
    
    def FadeOut(self):
        return tf.reduce_sum(KLD_diag_gaussians(self.w_scale_mu, self.w_scale_sigma, logit_mu, tf.math.log(logit_sigma)))
    
    
    def KLD_params(self):
        res = - tf.reduce_sum(KLD_diag_gaussians(self.w_conv_mu, self.w_conv_sigma, 0, 0))
        res -= tf.reduce_sum(KLD_diag_gaussians(self.w_out_mu, self.w_out_sigma, 0, 0))
        res -= tf.reduce_sum(KLD_diag_gaussians(self.bmu, self.bsigma, 0, 0))
        res -= tf.reduce_sum(KLD_diag_gaussians(self.psw_mu, self.psw_sigma, 0, 0))
        return (res - self.FadeOut())/neff
     
    
    
    
    def call(self, inputs, var = True):
        if not var:
            w_out = self.w_out_mu
            w_conv = self.w_conv_mu
            w_scale = self.w_scale_mu
            b = self.bmu
            psw = self.Sampler(self.psw_mu, self.psw_sigma)
        else:
            w_out = self.Sampler(self.w_out_mu, self.w_out_sigma)
            w_conv = self.Sampler(self.w_conv_mu, self.w_conv_sigma)
            w_scale = self.Sampler(self.w_scale_mu, self.w_scale_sigma)
            b = self.Sampler(self.bmu, self.bsigma)
            psw = self.psw_mu
        
        
        w_out = tf.matmul(w_out, w_conv)
        w_scale = tf.tile(w_scale, (self.n_pat, 1))
        w_scale = tf.expand_dims(w_scale, -1)
        w_out = tf.reshape(w_out, (self.inputshape[-1], self.seq_len, alp_len))
        w_out = tf.multiply(w_out, w_scale)
        w_out = tf.reshape(w_out, (self.inputshape[-1], self.seq_len * alp_len))
        
        output = tf.matmul(inputs, w_out) + b
        
        return tf.multiply(output, tf.math.log(1.0 + tf.exp(psw)))