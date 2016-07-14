#!/usr/bin/env python
import networks
from gym import envs
import tensorflow as tf
import keras.backend as K
import numpy as np
from worker import *
from kerlym import preproc
from kerlym.statbin import statbin
import matplotlib.pyplot as plt
import Queue

class A3C:
    def __init__(self, experiment="Breakout-v0", env=None, nthreads=16, nframes=1, epsilon=0.5, 
            enable_plots=False, render=False, learning_rate=1e-4, 
            modelfactory= networks.simple_cnn, difference_obs=True, 
            preprocessor = preproc.karpathy_preproc, discount=0.99,
            batch_size = 32, epsilon_min=0.05, epsilon_schedule=None,
            stats_rate = 10, 
            **kwargs ):
        self.kwargs = kwargs
        self.experiment = experiment
        if env==None:
            env=lambda: envs.make(self.experiment)
        self.nthreads = nthreads
        self.env = map(lambda x: env(), range(0, self.nthreads))
        self.model_factory = modelfactory
        self.nframes = nframes
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_schedule = epsilon_schedule
        self.gamma = discount
        self.preprocessor = preprocessor
        self.difference_obs = difference_obs
        self.network_update_frequency = batch_size
        self.target_network_update_frequency = 10000
        self.T = 0
        self.TMAX = 80000000
        self.checkpoint_interval = 600
        self.checkpoint_dir = "/tmp/"
        self.enable_plots = enable_plots
        self.stats_rate = stats_rate
        self.ipy_clear = False
        self.next_plot = 0
        self.e = 0
        self.render = render

        self.render_rate_hz = 5.0
        self.render_ngames = 2
        self.plot_q = Queue.Queue()

        # set up output shape to be either pre-processed or not
        if not self.preprocessor == None:
            print self.env[0].observation_space.shape
            o = self.preprocessor(np.zeros( self.env[0].observation_space.shape ) )
            self.input_dim_orig = [self.nframes]+list(o.shape)
        else:
            self.input_dim_orig = [self.nframes]+list(self.env[0].observation_space.shape)
        self.input_dim = np.product( self.input_dim_orig )
        print self.input_dim, self.input_dim_orig

        # set up plotting storage
        self.stats = None
        if self.enable_plots:
            self.stats = {
                "tr":statbin(self.stats_rate),     # Total Reward
                "ft":statbin(self.stats_rate),     # Finishing Time
                "minvf":statbin(self.stats_rate),  # Min Value Fn
                "maxvf":statbin(self.stats_rate),  # Min Value Fn
                "cost":statbin(self.stats_rate),   # Loss
            }

        # set up the TF session
        self.session = tf.Session()
        K.set_session(self.session)
        self.setup_graphs()
        self.saver = tf.train.Saver()


    def setup_graphs(self):

        # Create shared network
        s, policy_network, value_network = self.model_factory(self, self.env[0], **self.kwargs)
        policy_network_params = policy_network.trainable_weights
        value_network_params = value_network.trainable_weights
        pi_values = policy_network(s)
        V_values = value_network(s)

        # Create shared target network
        st, target_policy_network, target_value_network = self.model_factory(self, self.env[0])
        target_policy_network_params = target_policy_network.trainable_weights
        target_value_network_params = target_value_network.trainable_weights
        target_pi_values = target_policy_network(st)
        target_V_values = target_value_network(st)

        # Op for periodically updating target network with online network weights
        reset_target_policy_network_params = [target_policy_network_params[i].assign(policy_network_params[i]) for i in range(len(target_policy_network_params))]
        reset_target_value_network_params = [target_value_network_params[i].assign(value_network_params[i]) for i in range(len(target_value_network_params))]

        # Define A3C cost and gradient update equations
        #a = tf.placeholder("float", [None, self.env[0].action_space.n])
        #y = tf.placeholder("float", [None])
        R = tf.placeholder("float", [None, 1])
        #action_pi_values = tf.reduce_sum(tf.mul(pi_values, a), reduction_indices=1)

        # policy network update
        cost_pi = tf.reduce_mean( K.log( pi_values * (R-V_values) ) )
        #cost_pi = tf.reduce_mean( K.log( action_pi_values * (R-V_values) ) )
        optimizer_pi = tf.train.AdamOptimizer(self.learning_rate)
        grad_update_pi = optimizer_pi.minimize(cost_pi, var_list=policy_network_params)

        # value network update
        cost_V = tf.reduce_mean( tf.square( R - V_values ) )
        optimizer_V = tf.train.AdamOptimizer(self.learning_rate)
        grad_update_V = optimizer_V.minimize(cost_V, var_list=value_network_params)
 
        # store variables and update functions for access   
        self.graph_ops = {
                 "R" : R,
                 "s" : s,
                 "pi_values" : pi_values,
                 "V_values" : V_values,
                 "st" : st,
                 "reset_target_policy_network_params" : reset_target_policy_network_params,
                 "reset_target_value_network_params" : reset_target_value_network_params,
#                 "a" : a,
#                 "y" : y,
                 "grad_update_pi" : grad_update_pi,
                 "cost_pi" : cost_pi,
                 "grad_update_V" : grad_update_V,
                 "cost_V" : cost_V
                }


    def train(self):
        # Initialize target network weights
        self.session.run(self.graph_ops["reset_target_policy_network_params"])
        self.session.run(self.graph_ops["reset_target_value_network_params"])
        self.session.run(tf.initialize_all_variables())
        threads = map(lambda tid: a3c_learner(self, tid), range(0,self.nthreads))
        # start actor-learners
        for t in threads:
            t.start()

        # Start rendering
        if self.render:
            self.rt = render_thread(self.render_rate_hz, self.env[0:self.render_ngames] )
            self.rt.start()

        # Start plotting
        if self.enable_plots:
            self.pt = plotter_thread(self)
            self.pt.start()

        print "Waiting for threads to finish..."
        for t in threads:
            t.join()

        # Shut down rendering
        if self.render:
            self.rt.done = True
            self.rt.join()
        
        # Shut down plotting
        if self.enable_plots:
            self.pt.done = True
            self.pt.join()


    def prepare_obs(self, obs):
        if not self.preprocessor == None:
            obs = self.preprocessor(obs)
        return obs

    def diff_obs(self, obs, last_obs=None):
        if self.difference_obs and not type(last_obs) == type(None):
            obs = obs - last_obs
        return obs

    def update_epsilon(self):
        if not self.epsilon_schedule == None:
            self.epsilon = max(self.epsilon_min, 
                               self.epsilon_schedule(self.T, self.epsilon))

    def update_stats_threadsafe(self, stats, tid=0):
        if self.enable_plots:
            self.plot_q.put(stats)

    def update_stats(self, stats, tid=0):
        self.e += 1
        # update stats store
        for k in stats.keys():
            self.stats[k].add( stats[k] )

        # only plot from thread 0
        if self.stats == None or tid > 0:
            return
        
        # plot if its time
        if(self.e >= self.next_plot):
            self.next_plot = self.e + self.stats_rate
            if self.ipy_clear:
                from IPython import display
                display.clear_output(wait=True)
            fig = plt.figure(1)
            fig.canvas.set_window_title("A3C Training Stats for %s"%(self.experiment))
            plt.clf()
            plt.subplot(2,2,1)
            self.stats["tr"].plot()
            plt.title("Total Reward per Episode")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.legend(loc=2)
            plt.subplot(2,2,2)
            self.stats["ft"].plot()
            plt.title("Finishing Time per Episode")
            plt.xlabel("Episode")
            plt.ylabel("Finishing Time")
            plt.legend(loc=2)
            plt.subplot(2,2,3)
            self.stats["maxvf"].plot2(fill_col='lightblue', label='Avg Max VF')
            self.stats["minvf"].plot2(fill_col='slategrey', label='Avg Min VF')
            plt.title("Value Function Outputs")
            plt.xlabel("Episode")
            plt.ylabel("Value Fn")
            plt.legend(loc=2)
            ax = plt.subplot(2,2,4)
            self.stats["cost"].plot2()
            plt.title("Training Loss")
            plt.xlabel("Training Epoch")
            plt.ylabel("Loss")
            try:
#                ax.set_yscale("log", nonposy='clip')
                plt.tight_layout()
            except:
                pass
            plt.show(block=False)
            plt.draw()
            plt.pause(0.001)

              

