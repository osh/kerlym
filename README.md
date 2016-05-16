# D2QN // Double Q-Learning Agent with Keras DeepNN Q-Function Approxmation

This repo implements a Deep Double Q-Learning Agent for OpenAI's Gym using Keras NN Primitives on top of Theano/TensorFlow
It is intended to be easy to make experiemnting with network configuration, different tasks, and RL tuning and testing over relatively large runs easy and straightforward.

# Usage

```
./run_pong.sh
```

or

```
Usage: main.py [options]

Options:
  -h, --help            show this help message and exit
  -e ENV, --env=ENV
  -n NET, --net=NET
  -f FREQ, --update_freq=FREQ
  -b BS, --batch_size=BS
  -o DROPOUT, --dropout=DROPOUT
  -p EPSILON, --epsilon=EPSILON
  -d DISCOUNT, --discount=DISCOUNT
  -t NFRAMES, --num_frames=NFRAMES
  -m MAXMEM, --max_mem=MAXMEM
  -P, --plots  
```

or

```python
from gym import envs
env = envs.make(options.env)
#env.monitor.start(training_dir)

import ddqn
agent = ddqn.D2QN(env, nframes=2, epsilon=0.1, discount=0.99, 
                    modelfactory=ddqn.simple_cnn,
                    update_nsamp=1000, batch_size=32, dropout=0.5, 
                    enable_plots = True, max_memory = 1000000, 
                    epsilon_schedule=lambda episode,epsilon: epsilon*(1-1e-4)
                    )
agent.learn()
```

# Acknowledgements

Much thanks to all of the following projects for their inspiration and contributions
 - https://github.com/dandxy89/rf_helicopter
 - https://github.com/sherjilozair/dqn
 - OpenAI Gym & Keras

Cheers
Tim
