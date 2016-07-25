kerlym --env Breakout-v0 \
       --net simple_cnn \
       --agent a3c \
       --num_frames 1 \
       --preprocessor atari \
       --learning_rate 1e-4 \
       --difference \
       --render \
       --plots \
       --plot_rate 10 \
       --concurrency 16

