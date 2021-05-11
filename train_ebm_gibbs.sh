#
#
#BASE=/scratch/gobi2/gwohl/GWG_EXPERIMENTS/deep_ebm2
#for PC in 0.01
#do
#  for MODEL in resnet-64 mlp-512
#  do
#    for DATA in static_mnist omniglot caltech
#    do
#      for WU in 1000
#      do
#        srun --gres=gpu:1 -c 8 --mem=10G -p gpu  \
#        python pcd_ebm.py --save_dir ${BASE}/${DATA}/${MODEL}_pc_${PC}_wu_${WU}_base_dist \
#        --sampler gwg --sampling_steps 20 --viz_every 250 --model  ${MODEL} \
#        --print_every 10 --lr .0001 --warmup_iters ${WU} --n_iters 25000 \
#        --buffer_size 10000 --buffer_init mean --dataset_name ${DATA} --base_dist \
#        --p_control ${PC} --eval_every 5000 --eval_sampling_steps 5000 &
#
#        srun --gres=gpu:1 -c 8 --mem=10G -p gpu  \
#        python pcd_ebm.py --save_dir ${BASE}/${DATA}/${MODEL}_pc_${PC}_wu_${WU} \
#        --sampler gwg --sampling_steps 20 --viz_every 250 --model  ${MODEL} \
#        --print_every 10 --lr .0001 --warmup_iters ${WU} --n_iters 25000 \
#        --buffer_size 10000 --buffer_init uniform --dataset_name ${DATA} \
#        --p_control ${PC} --eval_every 5000 --eval_sampling_steps 5000 &
#      done
#    done
#  done
#done


#
#BASE=/scratch/gobi2/gwohl/GWG_EXPERIMENTS/deep_ebm3
#for PC in 0.0
#do
#  for MODEL in resnet-64 mlp-512
#  do
#    for DATA in static_mnist omniglot caltech
#    do
#      for WU in 1000
#      do
#        srun --gres=gpu:1 -c 8 --mem=10G -p gpu  \
#        python pcd_ebm.py --save_dir ${BASE}/${DATA}/${MODEL}_pc_${PC}_wu_${WU}_base_dist \
#        --sampler gwg --sampling_steps 20 --viz_every 250 --model  ${MODEL} \
#        --print_every 10 --lr .00001 --warmup_iters ${WU} --n_iters 25000 \
#        --buffer_size 10000 --buffer_init mean --dataset_name ${DATA} --base_dist \
#        --p_control ${PC} --eval_every 5000 --eval_sampling_steps 5000 &
#
#        srun --gres=gpu:1 -c 8 --mem=10G -p gpu  \
#        python pcd_ebm.py --save_dir ${BASE}/${DATA}/${MODEL}_pc_${PC}_wu_${WU} \
#        --sampler gwg --sampling_steps 20 --viz_every 250 --model  ${MODEL} \
#        --print_every 10 --lr .00001 --warmup_iters ${WU} --n_iters 25000 \
#        --buffer_size 10000 --buffer_init uniform --dataset_name ${DATA} \
#        --p_control ${PC} --eval_every 5000 --eval_sampling_steps 5000 &
#      done
#    done
#  done
#done
#
#
#
#BASE=/scratch/gobi2/gwohl/GWG_EXPERIMENTS/deep_ebm4
#for PC in 0.0
#do
#  for MODEL in resnet-64 mlp-512
#  do
#    for DATA in static_mnist omniglot caltech
#    do
#
#        srun --gres=gpu:1 -c 8 --mem=10G -p gpu  \
#        python pcd_ebm.py --save_dir ${BASE}/${DATA}/${MODEL}_pc_${PC}_wu_${WU}_base_dist_low_lr \
#        --sampler gwg --sampling_steps 20 --viz_every 250 --model  ${MODEL} \
#        --print_every 10 --lr .00001 --warmup_iters 1000 --n_iters 25000 \
#        --buffer_size 10000 --buffer_init mean --dataset_name ${DATA} --base_dist \
#        --p_control ${PC} --eval_every 5000 --eval_sampling_steps 5000 --reinit_freq .01 &
#
#        srun --gres=gpu:1 -c 8 --mem=10G -p gpu  \
#        python pcd_ebm.py --save_dir ${BASE}/${DATA}/${MODEL}_pc_${PC}_wu_${WU}_base_dist_high_lr \
#        --sampler gwg --sampling_steps 20 --viz_every 250 --model  ${MODEL} \
#        --print_every 10 --lr .0001 --warmup_iters 10000 --n_iters 25000 \
#        --buffer_size 10000 --buffer_init mean --dataset_name ${DATA} --base_dist \
#        --p_control ${PC} --eval_every 5000 --eval_sampling_steps 5000 --reinit_freq .01 &
#
#    done
#  done
#done

#
#BASE=/scratch/gobi2/gwohl/GWG_EXPERIMENTS/deep_ebm_mlp_AIS
#
#for REINIT in 0.01 0.0
#do
#  for BUFFSIZE in 100 1000 10000
#  do
#    for LR in .001 #.0001 .00001
#    do
#      for SS in 40 80
#      do
#        srun --gres=gpu:1 -c 8 --mem=10G -p gpu  \
#        python pcd_ebm.py --save_dir ${BASE}/buff_${BUFFSIZE}_ri_${REINIT}_lr_${LR}_steps_${SS}_base_dist \
#        --sampler gwg --sampling_steps ${SS} --viz_every 100 \
#        --model mlp-512 --print_every 10 --lr ${LR} --warmup_iters 1000 --buffer_size ${BUFFSIZE} --n_iters 25000 \
#        --buffer_init mean --base_dist --p_control 0.0 --l2 0.0 --reinit_freq ${REINIT} \
#        --eval_every 1000 --eval_sampling_steps 10000 &
#
#        srun --gres=gpu:1 -c 8 --mem=10G -p gpu  \
#        python pcd_ebm.py --save_dir ${BASE}/buff_${BUFFSIZE}_ri_${REINIT}_lr_${LR}_steps_${SS} \
#        --sampler gwg --sampling_steps ${SS} --viz_every 100 \
#        --model mlp-512 --print_every 10 --lr ${LR} --warmup_iters 1000 --buffer_size ${BUFFSIZE} --n_iters 25000 \
#        --buffer_init mean --p_control 0.0 --l2 0.0 --reinit_freq ${REINIT} \
#        --eval_every 1000 --eval_sampling_steps 10000 &
#      done
#    done
#  done
#done
#
#
#BASE=/scratch/gobi2/gwohl/GWG_EXPERIMENTS/deep_ebm_mlp/static_mnist
#
#BUFFSIZE=10000
#REINIT=0.0
#for LR in .001 .0001
#do
#  for SS in 40 80
#  do
#    srun --gres=gpu:1 -c 8 --mem=10G -p gpu  \
#    python pcd_ebm_ema.py --save_dir ${BASE}/lr_${LR}_steps_${SS} \
#    --sampler gwg --sampling_steps ${SS} --viz_every 100 \
#    --model mlp-512 --print_every 10 --lr ${LR} --warmup_iters 1000 --buffer_size ${BUFFSIZE} --n_iters 50000 \
#    --buffer_init mean --base_dist --p_control 0.0 --l2 0.0 --reinit_freq ${REINIT} \
#    --eval_every 2500 --eval_sampling_steps 10000 &
#  done
#done
#
#
#BASE=/scratch/gobi2/gwohl/GWG_EXPERIMENTS/deep_ebm_resnet/static_mnist
#
#BUFFSIZE=10000
#REINIT=0.0
#for LR in .0001
#do
#  for SS in 40 80
#  do
#    srun --gres=gpu:1 -c 8 --mem=10G -p gpu  \
#    python pcd_ebm_ema.py --save_dir ${BASE}/lr_${LR}_steps_${SS} \
#    --sampler gwg --sampling_steps ${SS} --viz_every 100 \
#    --model resnet-64 --print_every 10 --lr ${LR} --warmup_iters 10000 --buffer_size ${BUFFSIZE} --n_iters 50000 \
#    --buffer_init mean --base_dist --p_control 0.0 --l2 0.0 --reinit_freq ${REINIT} \
#    --eval_every 5000 --eval_sampling_steps 10000 &
#  done
#done


#
#BASE=/scratch/gobi2/gwohl/GWG_EXPERIMENTS/deep_ebm_mlp
#
#BUFFSIZE=10000
#REINIT=0.0
#for DATA in dynamic_mnist omniglot caltech
#do
#  for LR in .001 .0001
#  do
#    for SS in 40 80
#    do
#      srun --gres=gpu:1 -c 8 --mem=10G -p gpu  \
#      python pcd_ebm_ema.py --save_dir ${BASE}/${DATA}/lr_${LR}_steps_${SS} \
#      --sampler gwg --sampling_steps ${SS} --viz_every 100 \
#      --model mlp-512 --print_every 10 --lr ${LR} --warmup_iters 1000 --buffer_size ${BUFFSIZE} --n_iters 50000 \
#      --buffer_init mean --base_dist --p_control 0.0 --l2 0.0 --reinit_freq ${REINIT} \
#      --eval_every 2500 --eval_sampling_steps 10000 --dataset_name ${DATA} &
#    done
#  done
#done
#
#BASE=/scratch/gobi2/gwohl/GWG_EXPERIMENTS/deep_ebm_resnet
#
#BUFFSIZE=10000
#REINIT=0.0
#for DATA in caltech
#do
#  for LR in .00001
#  do
#    for SS in 40 80
#    do
#      srun --gres=gpu:1 -c 8 --mem=10G -p gpu  \
#      python pcd_ebm_ema.py --save_dir ${BASE}/${DATA}/lr_${LR}_steps_${SS} \
#      --sampler gwg --sampling_steps ${SS} --viz_every 100 \
#      --model resnet-64 --print_every 10 --lr ${LR} --warmup_iters 10000 --buffer_size ${BUFFSIZE} --n_iters 50000 \
#      --buffer_init mean --base_dist --p_control 0.0 --l2 0.0 --reinit_freq ${REINIT} \
#      --eval_every 5000 --eval_sampling_steps 10000 --dataset_name ${DATA} &
#    done
#  done
#done

#
#
#BASE=/scratch/gobi2/gwohl/GWG_EXPERIMENTS/deep_ebm_cat2
#
#BUFFSIZE=1000
#REINIT=0.0
#
#for L2 in 0.0 #.0001
#do
#  for DATA in freyfaces histopathology
#  do
#    for LR in .00001
#    do
#      for SS in 40
#      do
#        for PROJ in 4 8
#        do
##          srun --gres=gpu:1 -c 8 --mem=10G -p gpu  \
##          python pcd_ebm_ema_cat.py --save_dir ${BASE}/resnet/${DATA}/lr_${LR}_steps_${SS}_proj_${PROJ}_l2_${L2}_no_base \
##          --sampler gwg --sampling_steps ${SS} --viz_every 100 \
##          --model resnet-64 --proj_dim ${PROJ} --print_every 10 --lr ${LR} --warmup_iters 10000 --buffer_size ${BUFFSIZE} \
##          --n_iters 50000 \
##          --buffer_init mean --p_control 0.0 --l2 ${L2} --reinit_freq ${REINIT} \
##          --eval_every 5000 --eval_sampling_steps 10000 --dataset_name ${DATA} &
#
#          srun --gres=gpu:1 -c 8 --mem=10G -p gpu  \
#          python pcd_ebm_ema_cat.py --save_dir ${BASE}/resnet/${DATA}/lr_${LR}_steps_${SS}_proj_${PROJ}_l2_${L2} \
#          --sampler gwg --sampling_steps ${SS} --viz_every 100 \
#          --model resnet-64 --proj_dim ${PROJ} --print_every 10 --lr ${LR} --warmup_iters 10000 --buffer_size ${BUFFSIZE} \
#          --n_iters 50000 \
#          --buffer_init mean --base_dist --p_control 0.0 --l2 ${L2} --reinit_freq ${REINIT} \
#          --eval_every 5000 --eval_sampling_steps 10000 --dataset_name ${DATA} &
#
#  #      srun --gres=gpu:1 -c 8 --mem=10G -p gpu  \
#  #      python pcd_ebm_ema_cat.py --save_dir ${BASE}/mlp/${DATA}/lr_${LR}_steps_${SS}_proj_${PROJ} \
#  #      --sampler gwg --sampling_steps ${SS} --viz_every 100 \
#  #      --model mlp-512 --proj_dim ${PROJ} --print_every 10 --lr ${LR} --warmup_iters 1000 --buffer_size ${BUFFSIZE} --n_iters 50000 \
#  #      --buffer_init mean --p_control 0.0 --l2 0.0 --reinit_freq ${REINIT} \
#  #      --eval_every 2500 --eval_sampling_steps 10000 --dataset_name ${DATA} &
#        done
#      done
#    done
#  done
#done


#
# python pcd_ebm_ema_cat.py --save_dir /scratch/gobi2/gwohl/test_cat_frey3 \
# --model mlp-512 --proj_dim 8 --sampling_steps 40 --lr .0001 --buffer_init mean \
# --reinit_freq 0.0 --sampler gwg --viz_every 100 --dataset_name freyfaces \
# --buffer_size 1000 --print_every 10 --eval_every 5000 --eval_sampling_steps 10000



BASE=/scratch/gobi2/gwohl/GWG_EXPERIMENTS/deep_ebm_resnet_gibbs

for SS in 40
do
  for DATA in static_mnist dynamic_mnist omniglot
  do
    srun --gres=gpu:1 -c 8 --mem=10G -p gpu  \
    python pcd_ebm_ema.py --save_dir ${BASE}/${DATA}/lr_.0001_steps_${SS} \
    --sampler rand_gibbs --sampling_steps ${SS} --viz_every 100 \
    --model resnet-64 --print_every 10 --lr .0001 --warmup_iters 10000 --buffer_size 10000 --n_iters 50000 \
    --buffer_init mean --base_dist --p_control 0.0 --l2 0.0 --reinit_freq 0.0 \
    --eval_every 5000 --eval_sampling_steps 10000 --dataset_name ${DATA} &
  done

  srun --gres=gpu:1 -c 8 --mem=10G -p gpu  \
    python pcd_ebm_ema.py --save_dir ${BASE}/caltech/lr_.00001_steps_${SS} \
    --sampler rand_gibbs --sampling_steps ${SS} --viz_every 100 \
    --model resnet-64 --print_every 10 --lr .00001 --warmup_iters 1000 --buffer_size 1000 --n_iters 50000 \
    --buffer_init mean --base_dist --p_control 0.0 --l2 0.0 --reinit_freq 0.0 \
    --eval_every 5000 --eval_sampling_steps 10000 --dataset_name caltech &
done
