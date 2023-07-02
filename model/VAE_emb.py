import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import warnings
warnings.filterwarnings("ignore")
import collections
import random
from tensorflow.keras import layers
from model.Layers import *
from model.Params import *


class Deep_Seq(tf.keras.Model):
    def __init__(self):
        super(Deep_Seq, self).__init__()
        self.att1 = AttentionBlock(8)
        self.att2 = AttentionBlock(8)
        #self.att2_ = AttentionBlock(8)
        #self.att2__ = AttentionBlock(8)
        self.flat1 = layers.Flatten()
        self.encoder1 = layers.Dense(1500)
        self.encoder2 = layers.Dense(1500)
        
        self.hmu = layers.Dense(latent_dim)
        self.hsigma = layers.Dense(latent_dim)
        
        self.decoder1 = layers.Dense(200)
        self.decoder2 = layers.Dense(500)

        self.out = layers.Dense(seq_len*32)
        
        self.reshape = layers.Reshape((seq_len, 32))
        self.att3 = AttentionBlock(2)
        self.att4 = AttentionBlock(2)
        #self.att4_ = AttentionBlock(2)
        #self.att4__ = AttentionBlock(2)
        self.cout = Conv1(alp_len)
        
            
    def Encoder(self, x):
        #x = tf.one_hot(x, alp_len)
        x = tf.nn.embedding_lookup(embedding_recon, x)
        x = self.att1(x, PE=True)
        x = tf.nn.relu(x)
        x = self.att2(x)
        x = tf.nn.relu(x)
        #x = self.att2_(x)
        #x = tf.nn.relu(x)
        #x = self.att2__(x)
        #x = tf.nn.relu(x)
        x = self.flat1(x)
        x = self.encoder1(x)
        x = tf.nn.relu(x)
        x = self.encoder2(x)
        x = tf.nn.relu(x)
        mu = self.hmu(x)
        sigma = self.hsigma(x)
        
        return mu, sigma
    
    def Sampler(self, mu, sigma):
        eps = tf.random.normal(sigma.shape)
        return mu + eps * (tf.exp(sigma)**0.5)
    
    def Decoder(self, z, var=True):
        z = self.decoder1(z)#
        z = tf.nn.relu(z)
        
        z = self.decoder2(z)
        z = tf.nn.relu(z)

        z = self.out(z)
        z = self.reshape(z)
        z = tf.nn.relu(z)
        z = self.att3(z, PE=True)
        z = tf.nn.relu(z)
        z = self.att4(z)
        z = tf.nn.relu(z)
        #z = self.att4_(z)
        #z = tf.nn.relu(z)
        #z = self.att4__(z)
        #z = tf.nn.relu(z)
        #z = z + PositionCoding(z.shape[-2], z.shape[-1])
        x_hat = self.cout(z)
        return x_hat
    
    def KLD_params(self):
        return  self.out.KLD_params()+ self.decoder1.KLD_params() + self.decoder2.KLD_params()
        #
    
    def call(self, x):
        mu, sigma = self.Encoder(x)
        z = self.Sampler(mu, sigma)
        x_hat = self.Decoder(z)
        
        return x_hat, mu, sigma
    
    def recon(self, x):
        mu, sigma = self.Encoder(x)
        x_hat = self.Decoder(mu, var=False)
        
        return x_hat, mu, sigma