#!/usr/bin/env python
import os
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu%d,floatX=float32"%(random.randint(0,3))
import tempfile,logging,sys

from optparse import OptionParser
parser = OptionParser()
parser.add_option("-e", "--env", dest="env", default="MountainCar-v0")
parser.add_option("-n", "--net", dest="net", default="simple_dnn")
parser.add_option("-f", "--update_freq", dest="freq", default=1000, type='int')
parser.add_option("-b", "--batch_size", dest="bs", default=32, type='int')
parser.add_option("-o", "--dropout", dest="dropout", default=0.5, type='float')
parser.add_option("-p", "--epsilon", dest="epsilon", default=0.1, type='float')
parser.add_option("-d", "--discount", dest="discount", default=0.99, type='float')
parser.add_option("-t", "--num_frames", dest="nframes", default=2, type='int')
parser.add_option("-m", "--max_mem", dest="maxmem", default=100000, type='int')
parser.add_option("-P", "--plots", dest="plots", action="store_true", default=False)
(options, args) = parser.parse_args()

training_dir = tempfile.mkdtemp()
logging.getLogger().setLevel(logging.DEBUG)

print options.plots

from gym import envs
env = envs.make(options.env)
#env.monitor.start(training_dir)

import ddqn
agent = ddqn.D2QN(env, nframes=options.nframes, epsilon=options.epsilon, discount=options.discount, modelfactory=eval("ddqn.%s"%(options.net)),
                    epsilon_schedule=lambda episode,epsilon: epsilon*(1-1e-4),
                    update_nsamp=options.freq, batch_size=options.bs, dropout=options.dropout, 
                    enable_plots = options.plots, max_memory = options.maxmem )
agent.learn()
#env.monitor.close()
#gym.upload(training_dir,
#           algorithm_id='ddqn_osh')



