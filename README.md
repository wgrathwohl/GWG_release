# GWG_release
Official release of code for "Oops I Took A Gradient: Scalable Sampling for Discrete Distributions"

# Code for sampling experiments can be found in: 
rbm_sample.py, ising_sample.py, fhmm_sample.py, potts_sample.py, svgd_sample.py

# To generate training data for ising inference experiments run:
The following command trains a basic cifar10 model.
```
python train.py --exp=cifar10_model --step_lr=100.0 --num_steps=40 --cuda --ensembles=1 --kl_coeff=1.0 --kl=True --multiscale --self_attn --cond
```

