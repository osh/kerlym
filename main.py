#!/usr/bin/env python
import os
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu%d,floatX=float32"%(random.randint(0,3))
import tempfile,logging,sys
from gym import envs
import ddqn

training_dir = tempfile.mkdtemp()
logging.getLogger().setLevel(logging.DEBUG)

if len(sys.argv) > 1:
    task = sys.argv[1]
else:
    task = "MountainCar-v0"

env = envs.make(task)
#env.monitor.start(training_dir)

agent = ddqn.D2QN(env, nframes=2, epsilon=0.25, discount=0.9, modelfactory=ddqn.embedding_rnn,
                    epsilon_schedule=lambda episode,epsilon: epsilon*(1-1e-2) )
agent.learn()
#env.monitor.close()
#gym.upload(training_dir,
#           algorithm_id='ddqn_tjo')
