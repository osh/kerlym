import logging,os,cPickle,time
from statbin import statbin
from random import choice, random, sample
import numpy as np

import networks

class D2QN:
    def __init__(self, env, nframes=1, epsilon=0.1, discount=0.99, train=1, update_nsamp=1000, timesteps_per_batch=1000, dropout=0, batch_size=32, nfit_epoch=1, epsilon_schedule=None, modelfactory=networks.simple_dnn, enable_plots=False, max_memory=100000, stats_rate=10, fit_verbose=0, difference_obs=False, preprocessor=None, render=False, **kwargs):
        self.double = True
        self.fit_verbose = 0
        self.env = env
        self.nframes = nframes
        self.actions = range(env.action_space.n)
        self.epsilon = epsilon
        self.gamma = discount
        self.train = train
        self.update_nsamp = update_nsamp
        self.timesteps_per_batch = timesteps_per_batch
        self.observations = []
        self.nfit_epoch = nfit_epoch
        self.epsilon_schedule = epsilon_schedule
        self.enable_plots = enable_plots
        self.max_memory = max_memory
        self.stats_rate = stats_rate
        self.train_costs = []
        self.nterminal = 0
        self.difference_obs = difference_obs
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.dropout = dropout
        self.render = render

        # set up output shape to be either pre-processed or not
        if not preprocessor == None:
            o = preprocessor(np.zeros( env.observation_space.shape ) )
            self.input_dim_orig = [nframes] + list(o.shape)
        else:
            self.input_dim_orig = [nframes]+list(env.observation_space.shape)
        self.input_dim = np.product( self.input_dim_orig )

        print "Input Dim: ", self.input_dim, self.input_dim_orig
        print "Output Actions: ", self.actions

        self.old_state_m1 = None
        self.action_m1 = None
        self.reward_m1 = None

        self.updates = 0
        self.model_updates = 0

        self.models = map(lambda x: modelfactory(self, env=env, dropout=dropout, **kwargs), [0,1])
        self.stats = None

        print self.models[0].summary()


    def act( self, state=None, pstate=None, paction=None, preward=None):
        state = np.asarray(state).reshape(1, self.input_dim)
        qval = self.get_model(greedy=True).predict(state, batch_size=1)

        if self.train:
            if random() < self.epsilon:
            #if random() < self.epsilon  or pstate is None:
                action = np.random.randint(0, len(self.actions))
            else:
                action = np.argmax(qval)

        else:
            qval = self.get_model(greedy=True).predict(state, batch_size=1)
            if self.updates == 0 or pstate is None:
                action = np.random.randint(0, len(self.actions))
                self.updates += 1
            else:
                action = np.argmax(qval)

        return action, qval

    def update_train(self, p_state, action, p_reward, new_state, terminal, update_model=False):
        assert self.max_memory >= self.timesteps_per_batch
        if(len(self.observations) >= self.max_memory):
            delidx = np.random.randint(0,len(self.observations)-1-self.timesteps_per_batch)
            del self.observations[delidx]


        self.nterminal += 1 if terminal else 0
        self.observations.append((p_state, action, p_reward, new_state, terminal))
        self.updates += 1

        # Train Model once enough history
        if(update_model):

            # number of NN updates we have performed (used to index between double Q models)
            self.model_updates += 1

            X_train, y_train = self.process_minibatch(terminal)
            hist = self.get_model(greedy=False).fit(X_train,
                           y_train,
                           batch_size=self.batch_size,
                           nb_epoch=self.nfit_epoch,
                           #nb_epoch=1,
                           verbose=self.fit_verbose,
                           shuffle=True)

            self.train_costs.extend(hist.history["loss"])

    def get_model( self, greedy=False ):
        if greedy or not self.double:
            return self.models[(self.model_updates+1)%2]
        else:
            return self.models[self.model_updates%2]

    def process_minibatch(self, terminal_rewards):
        X_train = []
        y_train = []
        val = 0

        if self.update_nsamp == None:
            samples = self.observations
        else:
            nsamp_new = self.timesteps_per_batch                    # new samples to learn from
            nsamp = min(len(self.observations), self.update_nsamp)  # total number of samples we will train on
            nsamp_replay = max(0,nsamp-self.timesteps_per_batch)         # number of replay samples

            # concat our new and random replay samples
            samples = self.observations[-nsamp_new:] + sample(self.observations[0:-nsamp_new], nsamp_replay)

        for memory in samples:
            if val == 0:
                val += 1
                old_state_m1, action_m1, reward_m1, new_state_m1, terminal = memory
            else:
                # Get stored values.
                old_state_m, action_m, reward_m, new_state_m, terminal = memory

                input = old_state_m
                old_state_m = input.reshape(1, self.input_dim)
                old_qval =self.get_model(greedy=True).predict(old_state_m,
                                              batch_size=1,
                                              verbose=0)

                input2 = new_state_m
                new_state_m = input2.reshape(1, self.input_dim)
                newQ = self.get_model(greedy=True).predict(new_state_m,
                                          batch_size=1,
                                          verbose=0)

                #print newQ
                maxQ = np.max(newQ)
                y = np.zeros((1, len(self.actions)))
                y[:] = old_qval[:]

                # Check for terminal state.
                if terminal:
                    update = reward_m
                else:
                    update = (reward_m + (self.gamma * maxQ))

                y[0][action_m] = update
                X_train.append(old_state_m.reshape(self.input_dim,))
                y_train.append(y.reshape(len(self.actions),))
                self.old_state_m1, self.action_m1, self.reward_m1, new_state_m1, terminal = memory

        # Generate Numpy Arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        return X_train, y_train


    def learn(self, ipy_clear=False, max_episodes=100000000, max_pathlength=200):

        start_time = time.time()
        numeptotal = 0
        i = 0

        if self.enable_plots and self.stats is None:
            import matplotlib.pyplot as plt
            self.stats = {
                "tr":statbin(self.stats_rate),     # Total Reward
                "ft":statbin(self.stats_rate),     # Finishing Time
                "minvf":statbin(self.stats_rate),     # Min Value Fn
                "maxvf":statbin(self.stats_rate),     # Min Value Fn
            }

        for e in xrange(max_episodes):

            observation = self.env.reset()
            if not self.preprocessor == None:
                observation = self.preprocessor(observation)

            done = False
            total_reward = 0.0
            t = 0
            maxv = []
            minv = []

            obs = np.zeros( self.input_dim_orig )
            new_obs = np.zeros( self.input_dim_orig )
            obs[0,:] = observation

            while (not done) and (t<max_pathlength):
                t += 1
                if self.render:
                    self.env.render()
                action, values = self.act(obs)
                maxv.append(max(values.flatten()))
                minv.append(min(values.flatten()))

                new_observation, reward, done, info = self.env.step(action)

                # compute preprocessed observation if enabled ...
                if not self.preprocessor == None:
                    new_observation = self.preprocessor(new_observation)

                # compute difference observation if enabled ...
                if self.difference_obs:
                    # compute difference image
                    o = (new_observation-observation)
                    observation = new_observation
                else:
                    # use observation directly
                    o = new_observation

                new_obs[1:,:] = obs[-1:,:]
                new_obs[0,:] = o
                if not done and t == max_pathlength-1:
                    done = True

                do_update = (i%self.timesteps_per_batch==self.timesteps_per_batch-1)
                self.update_train( obs, action, reward, new_obs, done, do_update )

                obs[:,:] = new_obs[:,:]
                total_reward += reward
                i += 1

            print " * Episode %08d\tFrame %08d\tSamples: %08d\tTerminal: %08d\tReward: %d\tEpsilon: %f"%(e, t, len(self.observations), self.nterminal, total_reward, self.epsilon)
            if not self.epsilon_schedule == None:
                self.epsilon = self.epsilon_schedule(e, self.epsilon)

            if self.enable_plots:
                self.stats["tr"].add(total_reward)
                self.stats["ft"].add(t)
                self.stats["maxvf"].add(np.mean(maxv))
                self.stats["minvf"].add(np.mean(minv))

                if(e%self.stats_rate == self.stats_rate-1):
                    if ipy_clear:
                        from IPython import display
                        display.clear_output(wait=True)
                    fig = plt.figure(1)
                    fig.canvas.set_window_title("DDQN Training Stats for %s"%(self.env.__class__.__name__))
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
                    plt.plot(self.train_costs)
                    plt.title("Training Loss")
                    plt.xlabel("Training Epoch")
                    plt.ylabel("Loss")
                    try:
                        ax.set_yscale("log", nonposy='clip')
                        plt.tight_layout()
                    except:
                        pass
                    plt.show(block=False)
                    plt.draw()
                    plt.pause(0.001)

class DQN(D2QN):
    def __init__(self, *args, **kwargs):
        D2QN.__init__(self, *args, **kwargs)
        self.double = False
