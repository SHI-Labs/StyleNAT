## StyleNAT: Giving Each Head a New Perspective
<a href="https://arxiv.org/abs/2211.05770"><img src="https://img.shields.io/badge/arxiv-https%3A%2F%2Farxiv.org%2Fabs%2F2211.05770-red"/></a>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/stylenat-giving-each-head-a-new-perspective/image-generation-on-ffhq-256-x-256)](https://paperswithcode.com/sota/image-generation-on-ffhq-256-x-256?p=stylenat-giving-each-head-a-new-perspective)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/stylenat-giving-each-head-a-new-perspective/image-generation-on-ffhq-1024-x-1024)](https://paperswithcode.com/sota/image-generation-on-ffhq-1024-x-1024?p=stylenat-giving-each-head-a-new-perspective)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/stylenat-giving-each-head-a-new-perspective/image-generation-on-lsun-churches-256-x-256)](https://paperswithcode.com/sota/image-generation-on-lsun-churches-256-x-256?p=stylenat-giving-each-head-a-new-perspective)

##### Authors: [Steven Walton](https://github.com/stevenwalton), [Ali Hassani](https://github.com/alihassanijr), [Xingqian Xu](https://github.com/xingqian2018), Zhangyang Wang, [Humphrey Shi](https://github.com/honghuis)

![header](images/header.png)
StyleNAT is a Style-based GAN that exploits [Neighborhood
Attention](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer) to
extend the power of localized attention heads to capture long range features and
maximize information gain within the generative process.
The flexibility of the the system allows it to be adapted to various
environments and datasets.

## Abstract:
Image generation has been a long sought-after but challenging task, and performing the generation task in an efficient manner is similarly difficult.
Often researchers attempt to create a "one size fits all" generator, where there are few differences in the parameter space for drastically different datasets.
Herein, we present a new transformer-based framework, dubbed StyleNAT, targeting high-quality image generation with superior efficiency and flexibility. 
At the core of our model, is a carefully designed framework that partitions attention heads to capture local and global information, which is achieved through using Neighborhood Attention (NA).
With different heads able to pay attention to varying receptive fields, the model is able to better combine this information, and adapt, in a highly flexible manner, to the data at hand.
StyleNAT attains a new SOTA  FID score on FFHQ-256 with 2.046, beating prior arts with convolutional models such as StyleGAN-XL and transformers such as HIT and StyleSwin, and a new transformer SOTA on FFHQ-1024 with an FID score of 4.174.
These results show a 6.4% improvement on FFHQ-256 scores when compared to StyleGAN-XL with a 28% reduction in the number of parameters and 56% improvement in sampling throughput. 

## Architecture
![architecture](images/architecture.png)

## Performance
![compute](images/fidparams.png)

Dataset | FID | Throughput (imgs/s) | Number of Parameters (M) |
|:---:|:---:|:---:|:---:|
FFHQ 256 | [2.046](https://shi-labs.com/projects/stylenat/checkpoints/FFHQ256_940k_flip.pt) | 32.56 | 48.92 |
FFHQ 1024 | [4.174](https://shi-labs.com/projects/stylenat/checkpoints/FFHQ1024_700k.pt)  | - | 49.45 |
Church 256 | 3.400  | - | - |

## Building and Using StyleNAT
We recommend building an environment with conda to get the best performance. We
recommend the following build instructions but your millage may vary.
```bash
conda create --name stylenat python=3.10
conda activate stylenat
conda install pytorch torchvision cudatoolkit=11.6 -c pytorch -c nvidia
# Use xargs to install lines one at a time since natten requires torch to be installed first
cat requirements.txt | xargs -L1 pip install 
```
Note: some version issues can create poor FIDs. Always check your build
environment first with the `evaluate` method. With the best FFHQ score you
should always get under an FID < 2.10 (hopefully closer to 2.05). 

Note: You may need to install torch and torchvision first due to dependence. Pip
does not build sequentially and NATTEN may fail to build. 

Notes: 
- [NATTEN can be sped up by using pre-built wheels directly.](https://shi-labs.com/natten/)

- Some arguments and configurations have changed slightly. Everything should be
backwards compatible but if they aren't please open an issue.

- This is research code, not production. There are plenty of optimizations that
can be implemented easily. We also are obsessive about logging information and
storing into checkpoints. Official checkpoints may not have all information as
current code tracks due to research and development. Most important things
should exist but if you're missing something important open an issue. Sorry,
seeds and rng states are only available if they exist in the checkpoints.

## Inference
Using META's hydra-core we can easily run. We simply have to run
```bash
python main.py type=inference
```
Note that the first time you run this it will take some time, upfirdn2d is compiling. 

By default this will create 10 random inference images with a checkpoint and the
names will be saved as the name of the random seed.

You can specify seeds by using
```bash
python main.py type=inference inference.seeds=[1,2,3,4]
```
If you would like to specify a set of seeds in a range use the following command
`python main 'inference.seeds="range(start, stop, step)"'`


## Evaluation
If you would like to check the performance of a model we provide the evaluation
mode type. Simply run
```bash
python main.py type=evaluation
```
See the config file to set the proper dataset, checkpoint, etc.

# Training
If you would like to train a model from scratch we provide the following mode
```bash
python main.py type=train restart.ckpt=null
```
We suggest explicitly setting the checkpoint to null so that you don't 
accidentally load a checkpoint.
It is also advised to create a new run file and call
```bash
python main.py type=train restart.ckpt=null runs=my_new_run
```
We also support distributed training. Simply use torchrun
```bash
torchrun --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS --node_rank=$NODE_RANK
main.py type=train
```

## Modifying Hydra-Configs
The `confs` directory holds yaml configs for different types of runs. If you would
like to adjust parameters (such as changing checkpoints, inference, number of
images, specifying seeds, and so on) you should edit this file. The `confs/runs` folder holds
parameters for the model and training options. It is not advised to modify these
files. It is better to copy them to a new file and use those if you wish to
train a new model.

## "Secret" Hydra args
There's a few unspecified hydra configs around wandb. We're just providing a
simple version. But we also support `tags` and `description` under this
argument.



## Citation:
```bibtex
@article{walton2022stylenat,
    title         = {StyleNAT: Giving Each Head a New Perspective},
    author        = {Steven Walton and Ali Hassani and Xingqian Xu and Zhangyang Wang and Humphrey Shi},
    year          = 2022,
    url           = {https://arxiv.org/abs/2211.05770},
    eprint        = {2211.05770},
    archiveprefix = {arXiv},
    primaryclass  = {cs.CV}
}
```

## Acknowledgements
This code heavily relies upon
[StyleSwin](https://github.com/microsoft/StyleSwin) which also relies upon
[rosinality's StyleGAN2-pytorch](https://github.com/rosinality/stylegan2-pytorch) library.
We also utilize [mseitzer's pytorch-fid](https://github.com/mseitzer/pytorch-fid).
Finally, we utilize SHI-Lab's [NATTEN](https://github.com/SHI-Labs/NATTEN/).

We'd also like to thank Intelligence Advanced Research Projects Activity
(IARPA), University of Oregon, University of Illinois at Urbana-Champaign, and
Picsart AI Research (PAIR) for their generous support. 
