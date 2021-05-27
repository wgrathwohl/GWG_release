# GWG_release
Official release of code for ["Oops I Took A Gradient: Scalable Sampling for Discrete Distributions"](https://arxiv.org/abs/2102.04509) which has been accepted for a long presentation to ICML 2021. 

The paper is by [me](http://www.cs.toronto.edu/~wgrathwohl/), Kevin Swersky, Milad Hashemi, [David Duvenaud](http://www.cs.toronto.edu/~duvenaud/), and [Chris Maddison](https://www.cs.toronto.edu/~cmaddis/)

See Gibbs-With-Gradients sampling from an Ising model: ![](gwg.gif)

# Code for sampling experiments can be found in: 
rbm_sample.py, ising_sample.py, fhmm_sample.py, potts_sample.py, svgd_sample.py


# To generate training data for ising inference experiments run:

```
./generate_data.sh
```

# Datasets for EBM training can be found at:
https://github.com/jmtomczak/vae_vampprior/tree/master/datasets

Download them and unzip as:

    GWG_release/

        datasets/
            Caltech...
            FreyFaces...
            Histo...
            MNIST_static/
            Omniglot/
        

If you would like access to the protein data please contact me at wgrathwohl@gmail.com, they are quite large and don't fit here :(

# To train a binary EBM run:

```
python pcd_ebm_ema.py --save_dir $DIR} \
    --sampler gwg --sampling_steps $NUM_STEPS --viz_every 100 \
    --model resnet-64 --print_every 10 --lr .0001 --warmup_iters 10000 --buffer_size 10000 --n_iters 50000 \
    --buffer_init mean --base_dist --reinit_freq 0.0 \
    --eval_every 5000 --eval_sampling_steps 10000 &
```

# To train a categorical EBM run:

```
python pcd_ebm_ema_cat.py --save_dir $DIR \
          --sampler gwg --sampling_steps $NUM_STEPS --viz_every 100 \
          --model resnet-64 --proj_dim $PROJ_DIM --print_every 10 --lr .0001 --warmup_iters 10000 --buffer_size 1000 \
          --n_iters 50000 \
          --buffer_init mean --base_dist --p_control 0.0 --reinit_freq 0.0 \
          --eval_every 5000 --eval_sampling_steps 10000 --dataset_name ${DATA}
```

# To evaluate with AIS run:
This takes a while...
```
python eval_ais.py \
    --ckpt_path $CKPT_path \
    --save_dir $DIR \
    --sampler gwg --model resnet-64 --buffer_size 10000 \
    --n_iters 300000 --base_dist --n_samples 500 \
    --eval_sampling_steps 300000 --ema --viz_every 1000
```


