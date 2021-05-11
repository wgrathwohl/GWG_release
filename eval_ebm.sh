# THIS TAKES A WHILE BROOOOOOO


OUTBASE=/scratch/gobi2/gwohl/GWG_EXPERIMENTS/deep_ebm_resnet_gibbs_EVAL
MODELBASE=/scratch/gobi2/gwohl/GWG_EXPERIMENTS/deep_ebm_resnet_gibbs

MODEL=lr_.00001_steps_800
for ITERS in 1000 3000 10000 30000 100000 300000
do
  for DATA in caltech static_mnist dynamic_mnist omniglot
  do
    #srun --gres=gpu:1 -c 8 --mem=10G -p gpu  \
    python eval_ais.py \
    --ckpt_path ${MODELBASE}/${DATA}/${MODEL}/best_ckpt.pt \
    --save_dir ${OUTBASE}/${DATA}/${MODEL}/iters_${ITERS} \
    --sampler gwg --model resnet-64 --buffer_size 10000 \
    --n_iters ${ITERS} --base_dist --n_samples 500 \
    --eval_sampling_steps ${ITERS} --ema --viz_every 1000 #&
  done
done