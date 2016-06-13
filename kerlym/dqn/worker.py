import threading
import numpy as np
import random

class dqn_learner(threading.Thread):

    def __init__(self, parent, tid):
        threading.Thread.__init__(self)
        self.parent = parent
        self.tid = tid
        self.env = self.parent.env[tid]

    def run(self):
        t = 0
        s_batch = []
        a_batch = []
        y_batch = []
        ops = self.parent.graph_ops

        while True:
            
            s_t_single = self.parent.prepare_obs(self.env.reset())
            s_t = np.tile(s_t_single,self.parent.nframes).flatten()
            terminal = False

            # Set up per-episode counters
            ep_reward = 0
            episode_ave_max_q = 0
            episode_ave_min_q = 0
            episode_ave_loss = 0
            ep_t = 0

            while True:
                # Forward the deep q network, get Q(s,a) values
                readout_t = ops["q_values"].eval(session = self.parent.session, feed_dict = {ops["s"] : [s_t]})

                # Choose next action based on e-greedy policy
                a_t = np.zeros([self.env.action_space.n])
                action_index = 0
                if random.random() <= self.parent.epsilon:
                    action_index = random.randrange(self.env.action_space.n)
                else:
                    action_index = np.argmax(readout_t)
                a_t[action_index] = 1

                # Gym excecutes action in game environment on behalf of actor-learner
                s_t1_single, r_t, terminal, info = self.env.step(action_index)
                s_t1_single = self.parent.prepare_obs(s_t1_single)
                s_t1 = self.parent.diff_obs(s_t1_single, s_t_single)
                s_t1 = np.concatenate( (s_t[0:(self.parent.nframes-1)*np.product(self.parent.input_dim_orig[1:])], s_t1.flatten() ) )

                # Accumulate gradients
                readout_j1 = ops["target_q_values"].eval(session = self.parent.session, feed_dict = {ops["st"] : [s_t1]})
                clipped_r_t = np.clip(r_t, -1, 1)
                if terminal:
                    y_batch.append(clipped_r_t)
                else:
                    y_batch.append(clipped_r_t + self.parent.gamma * np.max(readout_j1))
                a_batch.append(a_t)
                s_batch.append(s_t)
    
                # Update the state and counters
                s_t = s_t1
                s_t_single = s_t1_single
                self.parent.T += 1
                t += 1

                ep_t += 1
                ep_reward += r_t
                episode_ave_max_q += np.max(readout_t)
                episode_ave_min_q += np.min(readout_t)

                # Optionally update target network
                if self.parent.T % self.parent.target_network_update_frequency == 0:
                    self.parent.session.run(ops["reset_target_network_params"])

                # Optionally update online network
                if t % self.parent.network_update_frequency == 0 or terminal:
                    if s_batch:
                        self.parent.session.run(ops["grad_update"], 
                                                    feed_dict = {
                                                          ops["y"] : y_batch,
                                                          ops["a"] : a_batch,
                                                          ops["s"] : s_batch})
                    # Clear gradients
                    s_batch = []
                    a_batch = []
                    y_batch = []

                self.parent.update_epsilon()

                # Save model progress
                if t % self.parent.checkpoint_interval == 0 and self.tid == 0:
                    fp = self.parent.checkpoint_dir+"/checkpoint_"+self.parent.experiment+".ckpt"
                    print "Writing checkpoint: ", fp
                    self.parent.saver.save(self.parent.session, fp, global_step = t)

                # Print end of episode stats
                if terminal:
                    stats = {
                        'tr': ep_reward,
                        'ft':ep_t,
                        'maxvf':episode_ave_max_q/float(ep_t),
                        'minvf':episode_ave_min_q/float(ep_t)
                        }
                    self.parent.update_stats(stats, self.tid)
                    print "THREAD:", self.tid, "/ TIME", self.parent.T, "/ TIMESTEP", t, "/ EPSILON", self.parent.epsilon, "/ REWARD", ep_reward, "/ Q_MAX %.4f" % (episode_ave_max_q/float(ep_t))
                    break






