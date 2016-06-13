import threading
import numpy as np

class EnvWorker(threading.Thread):
    def __init__(self, env, input_dim_orig, preprocessor, difference_obs, act, iq, oq):
        threading.Thread.__init__(self)
        self.env = env
        self.input_dim_orig = input_dim_orig
        self.preprocessor = preprocessor
        self.render = False
        self.difference_obs = difference_obs
        self.episodes = 2
        self.iq = iq
        self.oq = oq
        self.act = act
    
    def run(self):

        while True:
            observation = self.env.reset()
    
            if not self.preprocessor == None:
                observation = self.preprocessor(observation)
    
            max_pathlength = 200
            done = False
            total_reward = 0.0
    
            obs = np.zeros( self.input_dim_orig )
            new_obs = np.zeros( self.input_dim_orig )
            obs[0,:] = observation
            t = 0
            while (not done) and (t<max_pathlength):
                t += 1
                if self.render:
                    self.env.render()
                action, values = self.act(obs)
    
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

   #             do_update = (i%self.timesteps_per_batch==self.timesteps_per_batch-1)
   #             self.update_train( obs, action, reward, new_obs, done, do_update )
#
                obs[:,:] = new_obs[:,:]
                total_reward += reward
                i += 1


     
