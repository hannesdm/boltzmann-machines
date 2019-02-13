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
import argparse
import numpy as np
from keras import regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.initializers import glorot_uniform
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.metrics import accuracy_score

from . import env
from boltzmann_machines import DBM
from boltzmann_machines.rbm import BernoulliRBM
from boltzmann_machines.utils import (RNG, Stopwatch,
                                      one_hot, one_hot_decision_function, unhot)
from boltzmann_machines.utils.dataset import load_mnist
from boltzmann_machines.utils.optimizers import MultiAdam


def make_rbm1(X, rbm1_dirpath, n_hidden=512, initial_n_gibbs_steps=1, lr=0.05, epochs=64,
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


def make_rbm2(Q, rbm2_dirpath, n_visible, n_hidden=1024, increase_n_gibbs_steps_every=20,
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


def make_dbm(X_train, X_val, rbms, Q, G, dbm_dirpath, n_particles=100, initial_n_gibbs_steps=1,
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
    # training settings
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general/data
    parser.add_argument('--gpu', type=str, default='0', metavar='ID',
                        help="ID of the GPU to train on (or '' to train on CPU)")
    parser.add_argument('--n-train', type=int, default=59000, metavar='N',
                        help='number of training examples')
    parser.add_argument('--n-val', type=int, default=1000, metavar='N',
                        help='number of validation examples')

    # RBM #2 related
    parser.add_argument('--increase-n-gibbs-steps-every', type=int, default=20, metavar='I',
                        help='increase number of Gibbs steps every specified number of epochs for RBM #2')

    # common for RBMs and DBM
    parser.add_argument('--n-hiddens', type=int, default=(512, 1024), metavar='N', nargs='+',
                        help='numbers of hidden units')
    parser.add_argument('--n-gibbs-steps', type=int, default=(1, 1, 1), metavar='N', nargs='+',
                        help='(initial) number of Gibbs steps for CD/PCD')
    parser.add_argument('--lr', type=float, default=(0.05, 0.01, 2e-3), metavar='LR', nargs='+',
                        help='(initial) learning rates')
    parser.add_argument('--epochs', type=int, default=(64, 120, 500), metavar='N', nargs='+',
                        help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=(48, 48, 100), metavar='B', nargs='+',
                        help='input batch size for training, `--n-train` and `--n-val`' + \
                             'must be divisible by this number (for DBM)')
    parser.add_argument('--l2', type=float, default=(1e-3, 2e-4, 1e-7), metavar='L2', nargs='+',
                        help='L2 weight decay coefficients')
    parser.add_argument('--random-seed', type=int, default=(1337, 1111, 2222), metavar='N', nargs='+',
                        help='random seeds for models training')

    # save dirpaths
    parser.add_argument('--rbm1-dirpath', type=str, default='../models/dbm_mnist_rbm1/', metavar='DIRPATH',
                        help='directory path to save RBM #1')
    parser.add_argument('--rbm2-dirpath', type=str, default='../models/dbm_mnist_rbm2/', metavar='DIRPATH',
                        help='directory path to save RBM #2')
    parser.add_argument('--dbm-dirpath', type=str, default='../models/dbm_mnist/', metavar='DIRPATH',
                        help='directory path to save DBM')

    # DBM related
    parser.add_argument('--n-particles', type=int, default=100, metavar='M',
                        help='number of persistent Markov chains')
    parser.add_argument('--max-mf-updates', type=int, default=50, metavar='N',
                        help='maximum number of mean-field updates per weight update')
    parser.add_argument('--mf-tol', type=float, default=1e-7, metavar='TOL',
                        help='mean-field tolerance')
    parser.add_argument('--max-norm', type=float, default=6., metavar='C',
                        help='maximum norm constraint')
    parser.add_argument('--sparsity-target', type=float, default=(0.2, 0.1), metavar='T', nargs='+',
                        help='desired probability of hidden activation')
    parser.add_argument('--sparsity-cost', type=float, default=(1e-4, 5e-5), metavar='C', nargs='+',
                        help='controls the amount of sparsity penalty')
    parser.add_argument('--sparsity-damping', type=float, default=0.9, metavar='D',
                        help='decay rate for hidden activations probs')

    # parse and check params
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    for x, m in (
            (args.n_gibbs_steps, 3),
            (args.lr, 3),
            (args.epochs, 3),
            (args.batch_size, 3),
            (args.l2, 3),
            (args.random_seed, 3),
            (args.sparsity_target, 2),
            (args.sparsity_cost, 2),
            (args.mlp_lrm, 3),
    ):
        if len(x) == 1:
            x *= m

    # prepare data (load + scale + split)
    print("\nPreparing data ...\n\n")
    X, y = load_mnist(mode='train', path='../data/')
    X /= 255.


    # pre-train RBM #1
    rbm1 = make_rbm1(X, args)

    # freeze RBM #1 and extract features Q = p_{RBM_1}(h|v=X)
    Q = None
    if not os.path.isdir(args.rbm2_dirpath) or not os.path.isdir(args.dbm_dirpath):
        print("\nExtracting features from RBM #1 ...")
        Q = rbm1.transform(X)
        print("\n")

    # pre-train RBM #2
    rbm2 = make_rbm2(Q, args)

    # freeze RBM #2 and extract features G = p_{RBM_2}(h|v=Q)
    G = None
    if not os.path.isdir(args.dbm_dirpath):
        print("\nExtracting features from RBM #2 ...")
        G = rbm2.transform(Q)
        print("\n")

    # jointly train DBM
    dbm = make_dbm((X_train, X_val), (rbm1, rbm2), (Q, G), args)

    # load test data
    X_test, y_test = load_mnist(mode='test', path='../data/')
    X_test /= 255.

    # discriminative fine-tuning: initialize MLP with
    # learned weights, add FC layer and train using backprop
    print("\nDiscriminative fine-tuning ...\n\n")

    W, hb = None, None
    W2, hb2 = None, None
    if not args.mlp_no_init:
        weights = dbm.get_tf_params(scope='weights')
        W = weights['W']
        hb = weights['hb']
        W2 = weights['W_1']
        hb2 = weights['hb_1']

    make_mlp((X_train, y_train), (X_val, y_val), (X_test, y_test),
             (W, hb), (W2, hb2), args)


if __name__ == '__main__':
    main()
