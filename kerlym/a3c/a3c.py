from logging import getLogger
import os,copy,time
import networks
import numpy as np
import chainer
from chainer import serializers
from chainer import functions as F
from kerlym import preproc
import copy_param
import multiprocessing as mp
from prepare_output_dir import prepare_output_dir
import dqn_head,policy,v_function,rmsprop_async,async
from init_like_torch import init_like_torch
logger = getLogger(__name__)

class A3CModel(chainer.Link):

    def pi_and_v(self, state, keep_same_state=False):
        raise NotImplementedError()

    def reset_state(self):
        pass

    def unchain_backward(self):
        pass



def phi(obs):
    resized = cv2.resize(obs.image_buffer, (84, 84))
    return resized.transpose(2, 0, 1).astype(np.float32) / 255


class A3CFF(chainer.ChainList, A3CModel):

    def __init__(self, n_actions):
        self.head = dqn_head.NIPSDQNHead(n_input_channels=3)
        self.pi = policy.FCSoftmaxPolicy(
            self.head.n_output_channels, n_actions)
        self.v = v_function.FCVFunction(self.head.n_output_channels)
        chainer.ChainList.__init__(self, self.head, self.pi, self.v)
        #super().__init__(self.head, self.pi, self.v)
        init_like_torch(self)

    def pi_and_v(self, state, keep_same_state=False):
        out = self.head(state)
        return self.pi(out), self.v(out)


class A3CLSTM(chainer.ChainList, A3CModel):

    def __init__(self, n_actions):
        self.head = dqn_head.NIPSDQNHead(n_input_channels=3)
        self.pi = policy.FCSoftmaxPolicy(
            self.head.n_output_channels, n_actions)
        self.v = v_function.FCVFunction(self.head.n_output_channels)
        self.lstm = L.LSTM(self.head.n_output_channels,
                           self.head.n_output_channels)
        super().__init__(self.head, self.lstm, self.pi, self.v)
        init_like_torch(self)

    def pi_and_v(self, state, keep_same_state=False):
        out = self.head(state)
        if keep_same_state:
            prev_h, prev_c = self.lstm.h, self.lstm.c
            out = self.lstm(out)
            self.lstm.h, self.lstm.c = prev_h, prev_c
        else:
            out = self.lstm(out)
        return self.pi(out), self.v(out)

    def reset_state(self):
        self.lstm.reset_state()

    def unchain_backward(self):
        self.lstm.h.unchain_backward()
        self.lstm.c.unchain_backward()



class A3C:
    def __init__(self, experiment="Breakout-v0", env=None, nthreads=16, nframes=1, epsilon=0.5,
            enable_plots=False, render=False, learning_rate=1e-4,
            modelfactory= networks.simple_cnn, difference_obs=True,
            preprocessor = preproc.karpathy_preproc, discount=0.99,
            batch_size = 32, epsilon_min=0.05, epsilon_schedule=None,
            stats_rate = 10,
            **kwargs ):

        print "A3C ..."
        if env == None:
            env = lambda: envs.make(self.experiment)
        self.env = env

    def train(self):
        seed = None
        lr = 7e-4
        window_visible = False
        scenario = 'basic'
        use_lstm = False
        t_max = 5
        processes = 8
        beta = 1e-2
        profile = False
        steps = 8*10**7
        eval_frequency = 10**5
        eval_n_runs = 10

        if seed is not None:
            random_seed.set_random_seed(seed)
    
        # Simultaneously launching multiple vizdoom processes makes program stuck,
        # so use the global lock
        env_lock = mp.Lock()
    
        def make_env(process_idx, test):
            with env_lock:
                return self.env()

        n_actions = 3

        def model_opt():
            if use_lstm:
                model = A3CLSTM(n_actions)
            else:
                model = A3CFF(n_actions)
            opt = rmsprop_async.RMSpropAsync(lr=lr, eps=1e-1, alpha=0.99)
            opt.setup(model)
            opt.add_hook(chainer.optimizer.GradientClipping(40))
            return model, opt
    
        self.run_a3c(processes, make_env, model_opt, phi, t_max=t_max,
                    beta=beta, profile=profile, steps=steps, lr=lr,
                    eval_frequency=eval_frequency,
                    eval_n_runs=eval_n_runs, args={})

        print "Train"

    def train_loop(self, process_idx, counter, make_env, max_score, args, agent, env,
               start_time, outdir, lr, steps):
        try:
    
            total_r = 0
            episode_r = 0
            global_t = 0
            local_t = 0
            obs = env.reset()
            r = 0
            done = False
    
            while True:
    
                # Get and increment the global counter
                with counter.get_lock():
                    counter.value += 1
                    global_t = counter.value
                local_t += 1

                if global_t > steps:
                    break

                agent.optimizer.lr = (
                    steps - global_t - 1) / steps * lr

                total_r += r
                episode_r += r

                a = agent.act(obs, r, done)

                if done:
                    if process_idx == 0:
                        print('{} global_t:{} local_t:{} lr:{} r:{}'.format(
                            outdir, global_t, local_t, agent.optimizer.lr,
                            episode_r))
                    episode_r = 0
                    obs = env.reset()
                    r = 0
                    done = False
                else:
                    obs, r, done, info = env.step(a)
    
                if global_t % args.eval_frequency == 0:
                    # Evaluation
    
                    # We must use a copy of the model because test runs can change
                    # the hidden states of the model
                    test_model = copy.deepcopy(agent.model)
                    test_model.reset_state()
    
                    mean, median, stdev = eval_performance(
                        process_idx, make_env, test_model, agent.phi,
                        args.eval_n_runs)
                    #with open(os.path.join(outdir, 'scores.txt'), 'a+') as f:
                    #    elapsed = time.time() - start_time
                    #    record = (global_t, elapsed, mean, median, stdev)
                    #    print('\t'.join(str(x) for x in record), file=f)
                    #with max_score.get_lock():
                    #    if mean > max_score.value:
                    #        # Save the best model so far
                    #        print('The best score is updated {} -> {}'.format(
                    #            max_score.value, mean))
                    #        filename = os.path.join(
                    #            outdir, '{}.h5'.format(global_t))
                    #        agent.save_model(filename)
                    #        print('Saved the current best model to {}'.format(
                    #            filename))
                    #        max_score.value = mean

        except KeyboardInterrupt:
            if process_idx == 0:
                # Save the current model before being killed
                agent.save_model(os.path.join(
                    outdir, '{}_keyboardinterrupt.h5'.format(global_t)))
                #print('Saved the current model to {}'.format(
                #    outdir), file=sys.stderr)
            raise

        if global_t == args.steps + 1:
            # Save the final model
            agent.save_model(
                os.path.join(args.outdir, '{}_finish.h5'.format(args.steps)))
            print('Saved the final model to {}'.format(args.outdir))


    def train_loop_with_profile(self, process_idx, counter, make_env, max_score, args,
                            agent, env, start_time, outdir):
        import cProfile
        cmd = 'train_loop(process_idx, counter, make_env, max_score, args, ' \
            'agent, env, start_time)'
        cProfile.runctx(cmd, globals(), locals(),
                        'profile-{}.out'.format(os.getpid()))


    def run_a3c(self, processes, make_env, model_opt, phi, t_max=1, beta=1e-2,
                profile=False, steps=8 * 10 ** 7, eval_frequency=10 ** 6,
                eval_n_runs=10, args={}, lr=7e-4):
    
        # Prevent numpy from using multiple threads
        os.environ['OMP_NUM_THREADS'] = '1'

        outdir = prepare_output_dir(args, None)
    
        print('Output files are saved in {}'.format(outdir))
    
        n_actions = 20 * 20
    
        model, opt = model_opt()
    
        shared_params = async.share_params_as_shared_arrays(model)
        shared_states = async.share_states_as_shared_arrays(opt)

        max_score = mp.Value('f', np.finfo(np.float32).min)
        counter = mp.Value('l', 0)
        start_time = time.time()
    
        # Write a header line first
        with open(os.path.join(outdir, 'scores.txt'), 'a+') as f:
            column_names = ('steps', 'elapsed', 'mean', 'median', 'stdev')
#            print('\t'.join(column_names), file=f)

        def run_func(process_idx):
            env = make_env(process_idx, test=False)
            model, opt = model_opt()
            async.set_shared_params(model, shared_params)
            async.set_shared_states(opt, shared_states)
    
            agent = A3C_context(model, opt, t_max, 0.99, beta=beta,
                        process_idx=process_idx, phi=phi)

            if profile:
                self.train_loop_with_profile(process_idx, counter, make_env, max_score,
                                    args, agent, env, start_time,
                                    outdir=outdir)
            else:
                self.train_loop(process_idx, counter, make_env, max_score,
                           args, agent, env, start_time, outdir, lr, steps)
    
        async.run_async(processes, run_func)


class A3C_context(object):
    """A3C: Asynchronous Advantage Actor-Critic.

    See http://arxiv.org/abs/1602.01783
    """

    def __init__(self, model, optimizer, t_max, gamma, beta=1e-2,
                 process_idx=0, clip_reward=True, phi=lambda x: x,
                 pi_loss_coef=1.0, v_loss_coef=0.5,
                 keep_loss_scale_same=False):

        # Globally shared model
        self.shared_model = model

        # Thread specific model
        self.model = copy.deepcopy(self.shared_model)

        self.optimizer = optimizer
        self.t_max = t_max
        self.gamma = gamma
        self.beta = beta
        self.process_idx = process_idx
        self.clip_reward = clip_reward
        self.phi = phi
        self.pi_loss_coef = pi_loss_coef
        self.v_loss_coef = v_loss_coef
        self.keep_loss_scale_same = keep_loss_scale_same

        self.t = 0
        self.t_start = 0
        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_states = {}
        self.past_rewards = {}
        self.past_values = {}

    def sync_parameters(self):
        copy_param.copy_param(target_link=self.model,
                              source_link=self.shared_model)

    def act(self, state, reward, is_state_terminal):

        if self.clip_reward:
            reward = np.clip(reward, -1, 1)

        if not is_state_terminal:
            statevar = chainer.Variable(np.expand_dims(self.phi(state), 0))

        self.past_rewards[self.t - 1] = reward

        if (is_state_terminal and self.t_start < self.t) \
                or self.t - self.t_start == self.t_max:

            assert self.t_start < self.t

            if is_state_terminal:
                R = 0
            else:
                _, vout = self.model.pi_and_v(statevar, keep_same_state=True)
                R = float(vout.data)

            pi_loss = 0
            v_loss = 0
            for i in reversed(range(self.t_start, self.t)):
                R *= self.gamma
                R += self.past_rewards[i]
                v = self.past_values[i]
                if self.process_idx == 0:
                    logger.debug('s:%s v:%s R:%s',
                                 self.past_states[i].data.sum(), v.data, R)
                advantage = R - v
                # Accumulate gradients of policy
                log_prob = self.past_action_log_prob[i]
                entropy = self.past_action_entropy[i]

                # Log probability is increased proportionally to advantage
                pi_loss -= log_prob * float(advantage.data)
                # Entropy is maximized
                pi_loss -= self.beta * entropy
                # Accumulate gradients of value function

                v_loss += (v - R) ** 2 / 2

            if self.pi_loss_coef != 1.0:
                pi_loss *= self.pi_loss_coef

            if self.v_loss_coef != 1.0:
                v_loss *= self.v_loss_coef

            # Normalize the loss of sequences truncated by terminal states
            if self.keep_loss_scale_same and \
                    self.t - self.t_start < self.t_max:
                factor = self.t_max / (self.t - self.t_start)
                pi_loss *= factor
                v_loss *= factor

            if self.process_idx == 0:
                logger.debug('pi_loss:%s v_loss:%s', pi_loss.data, v_loss.data)

            total_loss = pi_loss + F.reshape(v_loss, pi_loss.data.shape)

            # Compute gradients using thread-specific model
            self.model.zerograds()
            total_loss.backward()
            # Copy the gradients to the globally shared model
            self.shared_model.zerograds()
            copy_param.copy_grad(
                target_link=self.shared_model, source_link=self.model)
            # Update the globally shared model
            if self.process_idx == 0:
                norm = self.optimizer.compute_grads_norm()
                logger.debug('grad norm:%s', norm)
            self.optimizer.update()
            if self.process_idx == 0:
                logger.debug('update')

            self.sync_parameters()
            self.model.unchain_backward()

            self.past_action_log_prob = {}
            self.past_action_entropy = {}
            self.past_states = {}
            self.past_rewards = {}
            self.past_values = {}

            self.t_start = self.t

        if not is_state_terminal:
            self.past_states[self.t] = statevar
            pout, vout = self.model.pi_and_v(statevar)
            self.past_action_log_prob[self.t] = pout.sampled_actions_log_probs
            self.past_action_entropy[self.t] = pout.entropy
            self.past_values[self.t] = vout
            self.t += 1
            if self.process_idx == 0:
                logger.debug('t:%s entropy:%s, probs:%s',
                             self.t, pout.entropy.data, pout.probs.data)
            return pout.action_indices[0]
        else:
            self.model.reset_state()
            return None

    def load_model(self, model_filename):
        """Load a network model form a file
        """
        serializers.load_hdf5(model_filename, self.model)
        copy_param.copy_param(target_link=self.model,
                              source_link=self.shared_model)
        opt_filename = model_filename + '.opt'
        if os.path.exists(opt_filename):
            print('WARNING: {0} was not found, so loaded only a model'.format(
                opt_filename))
            serializers.load_hdf5(model_filename + '.opt', self.optimizer)

    def save_model(self, model_filename):
        """Save a network model to a file
        """
        serializers.save_hdf5(model_filename, self.model)
        serializers.save_hdf5(model_filename + '.opt', self.optimizer)
