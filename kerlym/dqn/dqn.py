#!/usr/bin/env python
import networks
from gym import envs
import tensorflow as tf
import keras.backend as K
import numpy as np
from worker import *
from kerlym import preproc

class DQN:
    def __init__(self):
        self.experiment = "Breakout-v0"
        self.nthreads = 16
        env=lambda: envs.make(self.experiment)
        self.env = map(lambda x: env(), range(0, self.nthreads))
        self.model_factory = networks.simple_cnn
        self.nframes = 1
        #self.nframes = 4
        self.learning_rate = 1e-4
        self.epsilon = 0.5
        self.gamma = 0.99
        self.network_update_frequency = 32
        self.target_network_update_frequency = 10000
        self.T = 0
        self.TMAX = 80000000
        self.checkpoint_interval = 600
        self.checkpoint_dir = "/tmp/"
        self.preprocessor = preproc.karpathy_preproc
        self.difference_obs = True

        # set up output shape to be either pre-processed or not
        if not self.preprocessor == None:
            o = self.preprocessor(np.zeros( self.env[0].observation_space.shape ) )
            self.input_dim_orig = [self.nframes]+list(o.shape)
        else:
            self.input_dim_orig = [self.nframes]+list(self.env[0].observation_space.shape)
        self.input_dim = np.product( self.input_dim_orig )
        print self.input_dim, self.input_dim_orig


        # set up the TF session
        self.session = tf.Session()
        K.set_session(self.session)
        self.setup_graphs()
        self.saver = tf.train.Saver()

    def setup_graphs(self):
        # Create shared deep q network
        s, q_network = self.model_factory(self, self.env[0])
        network_params = q_network.trainable_weights
        q_values = q_network(s)

        # Create shared target network
        st, target_q_network = self.model_factory(self, self.env[0])
        target_network_params = target_q_network.trainable_weights
        target_q_values = target_q_network(st)

        # Op for periodically updating target network with online network weights
        reset_target_network_params = [target_network_params[i].assign(network_params[i]) for i in range(len(target_network_params))]

        # Define cost and gradient update op
        a = tf.placeholder("float", [None, self.env[0].action_space.n])
        y = tf.placeholder("float", [None])
        action_q_values = tf.reduce_sum(tf.mul(q_values, a), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(y - action_q_values))
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grad_update = optimizer.minimize(cost, var_list=network_params)
    
        self.graph_ops = {"s" : s,
                 "q_values" : q_values,
                 "st" : st,
                 "target_q_values" : target_q_values,
                 "reset_target_network_params" : reset_target_network_params,
                 "a" : a,
                 "y" : y,
                 "grad_update" : grad_update}


    def train(self):
        # Initialize target network weights
        self.session.run(self.graph_ops["reset_target_network_params"])
        self.session.run(tf.initialize_all_variables())
        threads = map(lambda tid: dqn_learner(self, tid), range(0,self.nthreads))
        for t in threads:
            t.start()

        print "Waiting for threads to finish..."
        for t in threads:
            t.join()


    def prepare_obs(self, obs):
        if not self.preprocessor == None:
            obs = self.preprocessor(obs)
        return obs

    def diff_obs(self, obs, last_obs=None):
        if self.difference_obs and not type(last_obs) == type(None):
            obs = obs - last_obs
        return obs

