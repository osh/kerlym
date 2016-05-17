#!/usr/bin/env python
import os,random
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu%d,floatX=float32"%(random.randint(0,3))
import tempfile,logging,sys

from optparse import OptionParser
parser = OptionParser()
parser.add_option("-e", "--env", dest="env", default="MountainCar-v0",                  help="Which GYM Environment to run")
parser.add_option("-n", "--net", dest="net", default="simple_dnn",                      help="Which NN Architecture to use for Q-Function approximation")
parser.add_option("-f", "--update_freq", dest="update_freq", default=1000, type='int',  help='Frequency of NN updates specified in time steps')
parser.add_option("-u", "--update_size", dest="update_size", default=1100, type='int',  help='Number of samples to train on each update')
parser.add_option("-b", "--batch_size", dest="bs", default=32, type='int',              help="Batch size durring NN training")
parser.add_option("-o", "--dropout", dest="dropout", default=0.5, type='float',         help="Dropout rate in Q-Fn NN")
parser.add_option("-p", "--epsilon", dest="epsilon", default=0.1, type='float',         help="Exploration(1.0) vs Exploitation(0.0) action probability")
parser.add_option("-D", "--epsilon_decay", dest="epsilon_decay", default=1e-4, type='float',    help="Rate of epsilon decay: epsilon*=(1-decay)")
parser.add_option("-s", "--epsilon_min", dest="epsilon_min", default=0.05, type='float',help="Min epsilon value after decay")
parser.add_option("-d", "--discount", dest="discount", default=0.99, type='float',      help="Discount rate for future reards")
parser.add_option("-t", "--num_frames", dest="nframes", default=2, type='int',          help="Number of Sequential observations/timesteps to store in a single example")
parser.add_option("-m", "--max_mem", dest="maxmem", default=100000, type='int',         help="Max number of samples to remember")
parser.add_option("-P", "--plots", dest="plots", action="store_true", default=False,    help="Plot learning statistics while running")
parser.add_option("-F", "--plot_rate", dest="plot_rate", default=10, type='int',        help="Plot update rate in episodes")
parser.add_option("-S", "--submit", dest="submit", action="store_true", default=False,  help="Submit Results to OpenAI")
(options, args) = parser.parse_args()

training_dir = tempfile.mkdtemp()
logging.getLogger().setLevel(logging.DEBUG)

from gym import envs
env = envs.make(options.env)
if options.submit:
    env.monitor.start(training_dir)

import ddqn
agent = ddqn.D2QN(env, nframes=options.nframes, epsilon=options.epsilon, discount=options.discount, modelfactory=eval("ddqn.%s"%(options.net)),
                    epsilon_schedule=lambda episode,epsilon: max(0.05, epsilon*(1-options.epsilon_decay)),
                    update_nsamp=options.update_freq, batch_size=options.bs, dropout=options.dropout,
                    timesteps_per_batch=options.update_size, stats_rate=options.plot_rate,
                    enable_plots = options.plots, max_memory = options.maxmem )
agent.learn()
if options.submit:
    env.monitor.close()
    gym.upload(training_dir, algorithm_id='kerlym_ddqn_osh')
