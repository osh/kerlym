# KEras Reinforcement Learning gYM agents, KeRLym

This repo is intended to host a handful of reinforcement learning agents implemented using the Keras (http://keras.io/) deep learning library for Theano and Tensorflow.
It is intended to make it easy to run, measure, and experiment with different learning configuration and underlying value function approximation networks while running a variery of OpenAI Gym environments (https://gym.openai.com/).


# Agents

 - ddqn, double q-learning agent with various Keras NN's for Q approximation

# Usage

```
./run_pong.sh
```

or

```
Usage: kerlym.py [options]
Exmaple: python kerlym.py -e Go9x9-v0 -n simple_dnn -P

Options:
  -h, --help            show this help message and exit
  -e ENV, --env=ENV     Which GYM Environment to run
  -n NET, --net=NET     Which NN Architecture to use for Q-Function
                        approximation
  -f UPDATE_FREQ, --update_freq=UPDATE_FREQ
                        Frequency of NN updates specified in time steps
  -u UPDATE_SIZE, --update_size=UPDATE_SIZE
                        Number of samples to train on each update
  -b BS, --batch_size=BS
                        Batch size durring NN training
  -o DROPOUT, --dropout=DROPOUT
                        Dropout rate in Q-Fn NN
  -p EPSILON, --epsilon=EPSILON
                        Exploration(1.0) vs Exploitation(0.0) action
                        probability
  -D EPSILON_DECAY, --epsilon_decay=EPSILON_DECAY
                        Rate of epsilon decay: epsilon*=(1-decay)
  -d DISCOUNT, --discount=DISCOUNT
                        Discount rate for future reards
  -t NFRAMES, --num_frames=NFRAMES
                        Number of Sequential observations/timesteps to store
                        in a single example
  -m MAXMEM, --max_mem=MAXMEM
                        Max number of samples to remember
  -P, --plots           Plot learning statistics while running
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
 - Keras and Gym

Cheers
Tim
