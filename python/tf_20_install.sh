conda create -n rl_tf_2 python=3.6

source activate rl_tf_2

pip install tensorflow-gpu==2.0.0-alpha0
# we need cuda 10
conda install cudatoolkit
conda install cudnn

pip install gym
pip install matplotlib
