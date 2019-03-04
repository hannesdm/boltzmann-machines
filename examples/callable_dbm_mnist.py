#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train 2-layer Bernoulli DBM on MNIST dataset with pre-training.
Hyper-parameters are similar to those in MATLAB code [1].
Some of them were changed for more efficient computation on GPUs,
another ones to obtain more stable learning (lesser number of "died" units etc.)
RBM #2 trained with increasing k in CD-k and decreasing learning rate
over time.

Per sample validation mean reconstruction error for DBM (mostly) monotonically
decreases during training and is about 5.27e-3 at the end.

The training took approx. 9 + 55 + 185 min = 4h 9m on GTX 1060.

After the model is trained, it is discriminatively fine-tuned.
The code uses early stopping so max number of MLP epochs is often not reached.
It achieves 1.32% misclassification rate on the test set.

Note that DBM is trained without centering.

Links
-----
[1] http://www.cs.toronto.edu/~rsalakhu/DBM.html
"""
# print(__doc__)

import os
import numpy as np
from keras import regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.initializers import glorot_uniform
from keras.models import Sequential
from keras.layers import Dense, Activation

from boltzmann_machines import DBM
from boltzmann_machines.rbm import BernoulliRBM
from boltzmann_machines.utils.optimizers import MultiAdam


def create_rbm1(X, rbm1_dirpath, n_hidden=512, initial_n_gibbs_steps=1, lr=0.05, epochs=64,
                batch_size=48, l2=1e-3, random_seed=1337):
    if os.path.isdir(rbm1_dirpath):
        print("\nLoading RBM #1 ...\n\n")
        rbm1 = BernoulliRBM.load_model(rbm1_dirpath)
    else:
        print("\nTraining RBM #1 ...\n\n")
        rbm1 = BernoulliRBM(n_visible=784,
                            n_hidden=n_hidden,
                            W_init=0.001,
                            vb_init=0.,
                            hb_init=0.,
                            n_gibbs_steps=initial_n_gibbs_steps,
                            learning_rate=lr,
                            momentum=[0.5] * 5 + [0.9],
                            max_epoch=epochs,
                            batch_size=batch_size,
                            l2=l2,
                            sample_h_states=True,
                            sample_v_states=True,
                            sparsity_cost=0.,
                            dbm_first=True,  # !!!
                            metrics_config=dict(
                                msre=True,
                                pll=True,
                                train_metrics_every_iter=500,
                            ),
                            verbose=True,
                            display_filters=30,
                            display_hidden_activations=24,
                            v_shape=(28, 28),
                            random_seed=random_seed,
                            dtype='float32',
                            tf_saver_params=dict(max_to_keep=1),
                            model_path=rbm1_dirpath)
        rbm1.fit(X)
    return rbm1


def create_rbm2(Q, rbm2_dirpath, n_visible, n_hidden=1024, increase_n_gibbs_steps_every=20,
                initial_n_gibbs_steps=1, epochs=120, batch_size=48, l2=2e-4, lr=0.01,
                random_seed=1111):
    if os.path.isdir(rbm2_dirpath):
        print("\nLoading RBM #2 ...\n\n")
        rbm2 = BernoulliRBM.load_model(rbm2_dirpath)
    else:
        print("\nTraining RBM #2 ...\n\n")

        n_every = increase_n_gibbs_steps_every

        n_gibbs_steps = np.arange(initial_n_gibbs_steps,
                                  initial_n_gibbs_steps + epochs / n_every)
        learning_rate = lr / np.arange(1, 1 + epochs / n_every)
        n_gibbs_steps = np.repeat(n_gibbs_steps, n_every)
        learning_rate = np.repeat(learning_rate, n_every)

        rbm2 = BernoulliRBM(n_visible=n_visible,
                            n_hidden=n_hidden,
                            W_init=0.005,
                            vb_init=0.,
                            hb_init=0.,
                            n_gibbs_steps=n_gibbs_steps,
                            learning_rate=learning_rate,
                            momentum=[0.5] * 5 + [0.9],
                            max_epoch=max(epochs, n_every),
                            batch_size=batch_size,
                            l2=l2,
                            sample_h_states=True,
                            sample_v_states=True,
                            sparsity_cost=0.,
                            dbm_last=True,  # !!!
                            metrics_config=dict(
                                msre=True,
                                pll=True,
                                train_metrics_every_iter=500,
                            ),
                            verbose=True,
                            display_filters=0,
                            display_hidden_activations=24,
                            random_seed=random_seed,
                            dtype='float32',
                            tf_saver_params=dict(max_to_keep=1),
                            model_path=rbm2_dirpath)
        rbm2.fit(Q)
    return rbm2


def extract_weights(dbm):
    weights = dbm.get_tf_params(scope='weights')
    W = weights['W']
    hb = weights['hb']
    W2 = weights['W_1']
    hb2 = weights['hb_1']
    return W, hb, W2, hb2


def create_mlp(X_train, y_train, X_val, y_val, dbm,
               n_hidden1=512, n_hidden2=1024, l2=1e-5,
               lrm1=0.01, lrm2=0.1, lrm3=1, mlp_val_metric='val_acc',
               early_stopping_patience=12,
               reduce_lr_patience=6,
               epochs=100, batch_size=128, seeds=(None, None, None)):
    W, hb, W2, hb2 = extract_weights(dbm)
    print("Layer 1 weights shape:", W.shape)
    print("Layer 2 weights shape:", W2.shape)
    import keras.backend as K  # multiple calls can give errors because of tensorflow - keras interactions
    K.clear_session()
    dense_params = {}
    if W is not None and hb is not None:
        dense_params['weights'] = (W, hb)

    dense2_params = {}
    if W2 is not None and hb2 is not None:
        dense2_params['weights'] = (W2, hb2)

    # define and initialize MLP model, default seeds = (3333, 4444, 5555)
    mlp = Sequential([
        Dense(n_hidden1, input_shape=(784,),
              kernel_regularizer=regularizers.l2(l2),
              kernel_initializer=glorot_uniform(seed=seeds[0]),
              **dense_params),
        Activation('sigmoid'),
        Dense(n_hidden2,
              kernel_regularizer=regularizers.l2(l2),
              kernel_initializer=glorot_uniform(seed=seeds[1]),
              **dense2_params),
        Activation('sigmoid'),
        Dense(10, kernel_initializer=glorot_uniform(seed=seeds[2])),
        Activation('softmax'),
    ])
    mlp.compile(optimizer=MultiAdam(lr=0.001,
                                    lr_multipliers={'dense_1': lrm1,
                                                    'dense_2': lrm2,
                                                    'dense_3': lrm3, }),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # train and evaluate classifier
    early_stopping = EarlyStopping(monitor=mlp_val_metric, patience=early_stopping_patience, verbose=2)
    reduce_lr = ReduceLROnPlateau(monitor=mlp_val_metric, factor=0.2, verbose=2,
                                  patience=reduce_lr_patience, min_lr=1e-5)
    try:
        mlp.fit(X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=False,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping, reduce_lr],
                verbose=1)
    except KeyboardInterrupt:
        pass  # return as usual
    # keras.backend.clear_session()
    return mlp


def create_dbm(X_train, X_val, rbms, Q, G, dbm_dirpath, n_particles=100, initial_n_gibbs_steps=1,
               max_mf_updates=50, mf_tol=1e-7, epochs=500, batch_size=100, lr=2e-3, l2=1e-7, max_norm=6.,
               sparsity_target=(0.2, 0.1), sparsity_cost=(1e-4, 5e-5), sparsity_damping=0.9,
               random_seed=2222):
    if os.path.isdir(dbm_dirpath):
        print("\nLoading DBM ...\n\n")
        dbm = DBM.load_model(dbm_dirpath)
        dbm.load_rbms(rbms)  # !!!
    else:
        print("\nTraining DBM ...\n\n")
        dbm = DBM(rbms=rbms,
                  n_particles=n_particles,
                  v_particle_init=X_train[:n_particles].copy(),
                  h_particles_init=(Q[:n_particles].copy(),
                                    G[:n_particles].copy()),
                  n_gibbs_steps=initial_n_gibbs_steps,
                  max_mf_updates=max_mf_updates,
                  mf_tol=mf_tol,
                  learning_rate=np.geomspace(lr, 5e-6, 400),
                  momentum=np.geomspace(0.5, 0.9, 10),
                  max_epoch=epochs,
                  batch_size=batch_size,
                  l2=l2,
                  max_norm=max_norm,
                  sample_v_states=True,
                  sample_h_states=(True, True),
                  sparsity_target=sparsity_target,
                  sparsity_cost=sparsity_cost,
                  sparsity_damping=sparsity_damping,
                  train_metrics_every_iter=400,
                  val_metrics_every_epoch=2,
                  random_seed=random_seed,
                  verbose=True,
                  display_filters=10,
                  display_particles=20,
                  v_shape=(28, 28),
                  dtype='float32',
                  tf_saver_params=dict(max_to_keep=1),
                  model_path=dbm_dirpath)
        dbm.fit(X_train, X_val)
    return dbm


def main():
    pass


if __name__ == '__main__':
    main()
