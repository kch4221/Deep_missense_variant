import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import warnings
warnings.filterwarnings("ignore")
import collections
import random
from tensorflow.keras import layers
import model.Layers
from model.Layers import AttentionBlock
import model.VAE_emb as VAE
from model.Params import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_numpy', type=str, default='data/train.npy', help='train data path with numpy format')
    parser.add_argument('--var_idseq', type=str, default='data/var_idseq.npy', help='variant id sequence')
    parser.add_argument('--variants_csv', type=str, default='data/variants.csv', help='variant csv file')

    args = parser.parse_args()

    train = np.load(args.train_numpy)
    var_idseq = np.load(args.var_idseq)
    variants = pd.read_csv(args.variants_csv)
    vae = VAE.Deep_Seq()#model
    batch_size = 128
    optimizer = tf.optimizers.Adam(0.0001)
    np.random.shuffle(train)
    train_data = tf.data.Dataset.from_tensor_slices(train[:-512])
    train_data = train_data.repeat().shuffle(1000).batch(batch_size).prefetch(1)
    val_data = train[-512:]

    ##########################################################################################


    train_step = 1000000
    display_step = 200
    log_step = 5
    losses = []
    corrs = []
    val_losses = []
    max_c = 0.5

    for step, batch_x in enumerate(train_data.take(train_step + 1)):
        with tf.GradientTape() as tape:
            x_, mu, sigma = vae(batch_x)
            kl_div = tf.reduce_sum(-0.5 * (2 * sigma + 1 - tf.square(mu) - tf.exp(2 * sigma)), axis=1)
            kl_div = tf.reduce_mean(kl_div)
            ce = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(tf.one_hot(batch_x, alp_len), x_), axis=1)
            loss = tf.reduce_mean(ce) + kl_div# - vae.KLD_params()
        gradient = tape.gradient(loss, vae.trainable_variables)
        optimizer.apply_gradients(zip(gradient, vae.trainable_variables))
        
        if step % log_step == 0:
            # wide type

            pre, mu, sigma= vae.recon(var_idseq)
            var_kl = tf.reduce_mean(-0.5 * (2 * sigma + 1 - tf.square(mu) - tf.exp(2 * sigma)), axis=1).numpy()
            elbo = - tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.one_hot(var_idseq, alp_len), pre), axis=1).numpy() - var_kl     
            variants['pre'] = elbo 
            corr = variants.corr('spearman').values[1][-1]
            corrs.append(corr)
            losses.append(loss.numpy())

            val_x, _, _b = vae(val_data)
            val_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.one_hot(val_data, alp_len), val_x))
            val_losses.append(val_loss.numpy())
            if corr > max_c:
                max_c = corr
                vae.save_weights('p_at_lstm.h5')
                pd.DataFrame({'loss':losses, 'val_loss':val_losses, 'corr':corrs}).to_csv(str(step)+'analysis.csv', index=False)

            if step % display_step == 0 and step != 0:
                print("step: %i, loss: %f, val_loss: %f, spearman_corr: %f, ce: %f, max: %f"  
                    % (step, loss, val_loss*seq_len,  corr, tf.reduce_mean(ce),  max_c))