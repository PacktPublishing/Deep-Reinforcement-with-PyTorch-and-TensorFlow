conda create -n rl_tf_2 python=3.6

source activate rl_tf_2

pip install tensorflow-gpu==2.0.0-alpha0
# we need cuda 10
conda install cudatoolkit
conda install cudnn

pip install gym
pip install matplotlib

conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

# set path to our python env
#!/home/user/anaconda3/envs/rl_tf_2/bin/python
# at
#sudo vim $(which ipython)
#sudo vim $(which ipython3)
#sudo vim $(which jupyter)

# check that kernel.json is referencing to the correct pyth
#jupyter kernelspec list

# run noteboo   k
jupyter notebook ~/PycharmProjects/Deep-Reinforcement-with-PyTorch-and-TensorFlow/



conda install swig # needed to build Box2D in the pip install
pip install box2d-py # a repackaged version of pybox2d



$ git clone https://github.com/openai/gym
$ cd gym
$ pip install -e .
$ pip install -e .[Box2D]