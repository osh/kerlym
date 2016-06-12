#kerlym -R -e CartPole-v0 -E none -n simple_dnn -t 1  -P -a ddqn -t 2 -p 0.25 -o 0 -R 0.001 -f 25 -u 32 
kerlym -R -e CartPole-v0 -E none -n karpathy_simple_pgnet -P -a pg -t 1 -r 0.001
#kerlym -R -e MountainCar-v0 -E none -n karpathy_simple_pgnet -t 1  -P -a pg -t 1

