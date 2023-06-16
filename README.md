# Optimally Sparse Automatic Noise Filtering 

Bachelor thesis study based on the work of:
`Bram Grooten, Ghada Sokar, Shibhansh Dohare, Elena Mocanu, Matthew E
Taylor, Mykola Pechenizkiy, and Decebal Constantin Mocanu. 2023. Automatic
noise filtering with dynamic sparse training in deep reinforcement learning. https://arxiv.org/abs/2302.06548`

Compared to the original code, changes have been made to the files `utils.py` and `sparse_utiles.py`.

# Abstract
Efficiently learning how to select relevant input features remains a daunting task in the development of artificial intelligence models, 
especially given high noise levels (80\% or more). However, when autonomous systems tauntingly spring into our daily, noisy lives, the importance 
of this task cannot be ignored. Automatically detecting relevant features has the potential to improve the way that systems all around us, 
from autonomous vehicles to security software, perform in supporting us. Considerable progress in this direction has been made in the last few years, 
particularly in Deep Reinforcement Learning, where models such as Automatic Noise Filtering (Grooten et al., 2023) have been shown to outperform 
previously existing artificial neural networks, even in the noisiest environments (up to 98% Gaussian noise). The results have been achieved 
using a partially sparse model (i.e. a network that only keeps the input layer sparse), unlike other state-of-the-art models, which employ a fully sparse network. 
However, this discovery begs the question of what an optimal sparsity distribution for noisy environments would look like. The present research builds 
on the outlined work exactly in this direction and aims to analyse how various sparsity distributions - inspired from previous, related work, as well as from fields 
such as cognitive neuroscience, probability theory and graph theory - affect the performance of neural networks in highly noisy (80% or more Gaussian noise) environments.



# Install
Credit for the _Requirements_, _Instructions_ and _Test_ to `Grooten et al. (2023)`, from code available online at https://github.com/bramgrooten/automatic-noise-filtering.

### Requirements
* Python 3.8
* PyTorch 1.9
* [MuJoCo-py](https://github.com/openai/mujoco-py) 
* [OpenAI gym](https://github.com/openai/gym)
* Linux (using [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) may work in Windows)

### Instructions 
First make a virtual environment:
```shell
sudo apt install python3.8 python3.8-venv python3.8-dev
python3.8 -m venv venv
source venv/bin/activate
```

If you don't have MuJoCo 2.10 yet:
```shell
cd ~
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz
mkdir .mujoco
mv mujoco210 .mujoco/
rm mujoco210-linux-x86_64.tar.gz
```

Now you have MuJoCo. Proceed with:
```shell
pip install mujoco_py==2.1.2.14 gym==0.21.0 torch==1.9.0
pip install wandb --upgrade
```


Now try to import mujoco_py in a python console, 
and do what the error messages tell you. 
(Like adding lines to your `.bashrc` file.)
```python
$ python
>>> import mujoco_py
```

You may need to install the following packages to solve some errors:
```shell
sudo apt install libosmesa6-dev libglew-dev patchelf
```


# Usage

### Train
To train an ANF-SAC agent on the ENE with 90% noise features, in the Walker2d-v3 environment, with global sparsity 0.9 and sparsity distribution 'new' run:
```
python main.py \
    --policy ANF-SAC \
    --env Walker2d-v3 \
    --fake_features 0.9 \
    --global_sparsity 0.9 \
    --sparsity_distribution_method new\
    --wandb_mode disabled
```


Possible policies: `ANF-SAC`, `ANF-TD3`, `SAC`, `TD3`.

Possible environments: `HalfCheetah-v3`, `Hopper-v3`, `Walker2d-v3`, `Humanoid-v3`.

Possible sparsity distributions: `new`, `ER`, `uniform`, `sparse_input`, `exp_input`, `sparse_output`, `exp_output`, `inverse_ER`, `normal`, `random`, `dynamic`. Default: `uniform`.

Possible output layer settings: `sparse` (add `--output layer sparse` to the parameter list), `dense` (default mode).

Possible visualisation modes with _wandb_ (see https://wandb.ai/site): `online`, `offline`, `disabled`.

Show all available arguments: `python main.py --help`

### Test

See the file `view_mujoco.py` to test a trained agent on a single episode and view its behavior.


