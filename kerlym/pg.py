""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import cPickle as pickle
import gym,keras
import preproc, networks

# This policy gradient implementation is an adaptation of Karpathy's GIST
# https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5

class PG:
    def __init__(self, env, nframes=1, preprocessor = preproc.karpathy_preproc,
        modelfactory=networks.karpathy_simple_pgnet,
        dropout=0,
        render=False,
        discount=0.99,
        file_model='pg_model.json',
        file_weights='pg_model_wts.h5',
        resume=False,
        *args, **kwargs):

        self.env = env
        self.render = render
        self.preprocessor = preprocessor
        self.nframes = nframes
        self.discount = discount
        self.file_model = file_model
        self.file_weights = file_weights
        self.resume = resume
        print "init"

        # set up output shape to be either pre-processed or not
        if not preprocessor == None:
            o = preprocessor(np.zeros( env.observation_space.shape ) )
            self.input_dim_orig = [nframes] + list(o.shape)
        else:
            self.input_dim_orig = [nframes]+list(env.observation_space.shape)
        self.input_dim = np.product( self.input_dim_orig )

        # Make NN model
        self.model = modelfactory(self, env=env, dropout=dropout, **kwargs)
        print self.model.summary()

        # testing
        if self.resume:
            self.load()

    def save(self):
        open(self.file_model,'w').write(self.model.to_json())
        self.model.save_weights(self.file_weights, overwrite=True)

    def load(self):
        self.model = keras.models.model_from_json(open(self.file_model).read())
        self.model.load_weights(self.file_weights)
        self.model.compile(optimizer='rmsprop', loss='mse')

    def discount_rewards(self, r):
      """ take 1D float array of rewards and compute discounted reward """
      discounted_r = np.zeros_like(r)
      running_add = 0
      for t in reversed(xrange(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * self.discount + r[t]
        discounted_r[t] = running_add
      return discounted_r

    def learn(self, ipy_clear=False, max_episodes=100000000, max_pathlength=200):

        observation = self.env.reset()
        prev_x = None # used in computing the difference frame
        xs,hs,dlogps,drs = [],[],[],[]
        running_reward = None
        reward_sum = 0
        episode_number = 0
        while True:
          if self.render: self.env.render()

          # preprocess the observation, set input to network to be difference image
          if not self.preprocessor ==None:
              cur_x = self.preprocessor(observation)
          else:
              cur_x = observation

          x = cur_x - prev_x if prev_x is not None else np.zeros(self.input_dim, dtype='float32')
          x = x.flatten()
          prev_x = cur_x

          # forward the policy network and sample an action from the returned probability
          aprob = self.model.predict(x.reshape([1,self.input_dim]), batch_size=1).flatten()
          action = np.random.choice( self.env.action_space.n, 1, p=aprob/np.sum(aprob) )[0]

          # record various intermediates (needed later for backprop)
          xs.append(x) # observation
          y = np.zeros([self.env.action_space.n])
          y[action] = 1

          dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

          # step the environment and get new measurements
          observation, reward, done, info = self.env.step(action)
          reward_sum += reward

          drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

          if done: # an episode finished
            print "EP DONE"
            episode_number += 1

            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)
            xs,hs,dlogps,drs = [],[],[],[] # reset array memory

            # compute the discounted reward backwards through time
            discounted_epr = self.discount_rewards(epr)
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)

            self.model.fit(epx, epdlogp,
                    nb_epoch=3, verbose=2, shuffle=True)

            # boring book-keeping
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
            if episode_number % 100 == 0:
                self.save()
            reward_sum = 0
            observation = self.env.reset() # reset env
            prev_x = None

          if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
            print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')
