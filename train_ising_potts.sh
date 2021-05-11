##
##for DIM in 10 25 50
##do
##  for SIGMA in -.1 0.0 .1 .25 .5 1.
##    do
##      srun --gres=gpu:1 -c 8 --mem=10G -p gpu python pcd.py \
##            --save_dir /scratch/gobi2/gwohl/GWG_DATASETS/ising_dim_${DIM}_sigma_${SIGMA} \
##            --model lattice_ising --data_model lattice_ising --dim ${DIM} --sigma ${SIGMA} &
##  done
##done
##
##
##for DIM in 10 25 50
##do
##  for SIGMA in -.1 0.0 .1 .25 .5 1.
##    do
##      srun --gres=gpu:1 -c 8 --mem=10G -p gpu python pcd.py \
##            --save_dir /scratch/gobi2/gwohl/GWG_DATASETS/potts_dim_${DIM}_sigma_${SIGMA} \
##            --model lattice_potts --data_model lattice_potts --dim ${DIM} --sigma ${SIGMA} &
##  done
##done
#
SEED=1111
#
#for DIM in 10 25 50
#do
#  for SIGMA in .15
#    do
#      srun --gres=gpu:1 -c 8 --mem=10G -p gpu python pcd.py \
#            --save_dir /scratch/gobi2/gwohl/GWG_DATASETS/ising_dim_${DIM}_sigma_${SIGMA} \
#            --model lattice_ising --data_model lattice_ising --dim ${DIM} --sigma ${SIGMA} &
#  done
#done

#
#BASE=/scratch/gobi2/gwohl/GWG_EXPERIMENTS/potts
#for SAMPLER in gwg rand_gibbs
#do
#  for STEPS in 5 10 25 50 100
#  do
#    for DIM in 10 25
#    do
#      for SIGMA in -.1 0.0 .1 .15 .25 .5 1.
#        do
#          srun --gres=gpu:1 -c 8 --mem=10G -p gpu python pcd.py \
#               --data_file /scratch/gobi2/gwohl/GWG_DATASETS/potts_dim_${DIM}_sigma_${SIGMA}/data.pkl \
#               --save_dir ${BASE}/pcd_potts_dim_${DIM}_sigma_${SIGMA}_steps_${STEPS}_${SAMPLER} \
#               --sigma ${SIGMA} --batch_size 256 --buffer_size 256 --lr .001 \
#               --sampling_steps ${STEPS} \
#               --viz_every 1000 --dim ${DIM} --n_iters 10000 --sampler ${SAMPLER} --model lattice_potts --seed ${SEED} &
#      done
#    done
#  done
#done
#
#
#BASE=/scratch/gobi2/gwohl/GWG_EXPERIMENTS/ising
#for SAMPLER in gwg rand_gibbs
#do
#  for STEPS in 5 10 25 50 100
#  do
#    for DIM in 10 25 50
#    do
#      for SIGMA in -.1 0.0 .1 .15 .25 .5 1.
#        do
#          srun --gres=gpu:1 -c 8 --mem=10G -p gpu python pcd.py \
#               --data_file /scratch/gobi2/gwohl/GWG_DATASETS/ising_dim_${DIM}_sigma_${SIGMA}/data.pkl \
#               --save_dir ${BASE}/pcd_ising_dim_${DIM}_sigma_${SIGMA}_steps_${STEPS}_${SAMPLER} \
#               --sigma ${SIGMA} --batch_size 256 --buffer_size 256 --lr .001 \
#               --sampling_steps ${STEPS} \
#               --viz_every 1000 --dim ${DIM} --n_iters 10000 --sampler ${SAMPLER} --model lattice_ising --seed ${SEED} &
#      done
#    done
#  done
#done


#BASE=/scratch/gobi2/gwohl/GWG_EXPERIMENTS/ising3d
#for SAMPLER in gwg rand_gibbs
#do
#  for STEPS in 5 10 25 50 100
#  do
#    for L1 in 0.01 0.02154435 0.04641589 0.1 0.21544347 0.46415888 1. 2.15443469 4.64158883 10.
#    do
#      srun --gres=gpu:1 -c 8 --mem=10G -p gpu python pcd.py \
#       --data_file /scratch/gobi2/gwohl/GWG_DATASETS/ising3d_dim_4_sigma_.2/data.pkl \
#       --save_dir ${BASE}/pcd_ising3d_steps_${STEPS}_l1_${L1}_${SAMPLER} \
#       --model lattice_ising_3d --dim 4 --sigma .2  --batch_size 256 --buffer_size 256 --lr .001 \
#       --sampler ${SAMPLER} --sampling_steps ${STEPS} --l1 ${L1} \
#       --viz_every 1000 --n_iters 50000 --seed ${SEED} &
#    done
#  done
#done
#
#
#BASE=/scratch/gobi2/gwohl/GWG_EXPERIMENTS/ising_2d
#for SAMPLER in gwg rand_gibbs
#do
#  for STEPS in 5 10 25 50 100
#  do
#    for DIM in 50
#    do
#      for SIGMA in -.1 0.0 .25 .5
#        do
#          for L1 in .01 .1 1.0
#          do
#            srun --gres=gpu:1 -c 8 --mem=10G -p gpu python pcd.py \
#                 --data_file /scratch/gobi2/gwohl/GWG_DATASETS/ising_dim_${DIM}_sigma_${SIGMA}/data.pkl \
#                 --save_dir ${BASE}/pcd_ising_dim_${DIM}_sigma_${SIGMA}_steps_${STEPS}_l1_${L1}_${SAMPLER} \
#                 --sigma ${SIGMA} --batch_size 256 --buffer_size 256 --lr .001 \
#                 --sampling_steps ${STEPS} \
#                 --viz_every 1000 --dim ${DIM} --n_iters 10000 --sampler ${SAMPLER} --model lattice_ising_2d \
#                 --l1 ${L1} --seed ${SEED} &
#          done
#      done
#    done
#  done
#done
#
#SEED=1111
#BASE=/scratch/gobi2/gwohl/GWG_EXPERIMENTS/ising_er
#
#for SAMPLER in gwg rand_gibbs
#do
#  for STEPS in 5 10 25 50 100
#  do
#    for L1 in .01 .1 1.0
#    do
#
#      srun --gres=gpu:1 -c 8 --mem=10G -p gpu python pcd.py \
#      --data_file /scratch/gobi2/gwohl/GWG_DATASETS/ising_er_nodes_200_conn_4_seed_${SEED}/data.pkl \
#      --graph_file /scratch/gobi2/gwohl/GWG_DATASETS/ising_er_nodes_200_conn_4_seed_${SEED}/J.pkl \
#      --save_dir ${BASE}/pcd_ising_nodes_200_conn_4_steps_${STEPS}_l1_${L1}_${SAMPLER} \
#      --batch_size 256 --buffer_size 256 \
#      --lr .001 --sampling_steps ${STEPS} --viz_every 1000 --dim 200 --n_iters 10000 \
#      --sampler ${SAMPLER} --model er_ising --l1 ${L1} --seed ${SEED} &
#
#      srun --gres=gpu:1 -c 8 --mem=10G -p gpu python pcd.py \
#      --data_file /scratch/gobi2/gwohl/GWG_DATASETS/ising_er_nodes_100_conn_2_seed_${SEED}/data.pkl \
#      --graph_file /scratch/gobi2/gwohl/GWG_DATASETS/ising_er_nodes_100_conn_2_seed_${SEED}/J.pkl \
#      --save_dir ${BASE}/pcd_ising_nodes_100_conn_2_steps_${STEPS}_l1_${L1}_${SAMPLER} \
#      --batch_size 256 --buffer_size 256 \
#      --lr .001 --sampling_steps ${STEPS} --viz_every 1000 --dim 100 --n_iters 10000 \
#      --sampler ${SAMPLER} --model er_ising --l1 ${L1} --seed ${SEED} &
#    done
#  done
#done
#

#BASE=/scratch/gobi2/gwohl/GWG_EXPERIMENTS/real_protein_BOVIN
#for L1 in .001
#do
#  for L2 in .01 .001 .0001 0.0
#  do
#    srun --gres=gpu:1 -c 8 --mem=10G -p gpu python pcd_potts.py \
#    --save_dir ${BASE}/l2_${L2}_l1_${L1} --model dense_potts --data BPT1_BOVIN \
#    --sampler gwg --sampling_steps 250 --print_every 10 --lr .001 --viz_every 100 --l1 ${L1} --weight_decay ${L2} \
#    --n_iters 10000 --batch_size 256 --buffer_size 256 &
#  done
#done


#BASE=/scratch/gobi2/gwohl/GWG_EXPERIMENTS/sampling/RBM
#for SEED in 1 2 3 4 5
#do
#  srun --gres=gpu:1 -c 8 --mem=10G -p gpu python rbm_block.py --save_dir ${BASE}/random/seed_${SEED} \
#  --viz_every 1000 --gt_steps 5000 --n_steps 100000 \
#  --n_visible 784 --n_hidden 500 --subsample 1 --burn_in 0.1 --n_samples 10 --n_test_samples 32 \
#  --ess_statistic hamming --seed ${SEED} &
#  srun --gres=gpu:1 -c 8 --mem=10G -p gpu python rbm_block.py --save_dir ${BASE}/mnist/seed_${SEED} \
#  --viz_every 1000 --gt_steps 5000 --n_steps 100000 \
#  --n_visible 784 --n_hidden 500 --subsample 1 --burn_in 0.1 --n_samples 10 --n_test_samples 32 \
#  --ess_statistic hamming --seed ${SEED} --data mnist &
#done
#
#BASE=/scratch/gobi2/gwohl/GWG_EXPERIMENTS/sampling/ISING
#for SEED in 1 2 3
#do
#  for DIM in 40 #10 25 50
#  do
#    for SIGMA in .1 .15 .2 .25 .3 .35 .4 .45 .5 1.0
#    do
#      srun --gres=gpu:1 -c 8 --mem=10G -p gpu  \
#      python ising_block.py \
#      --save_dir ${BASE}/dim_${DIM}/sigma_${SIGMA}/seed_${SEED} \
#      --viz_every 1000 --n_steps 100001 \
#      --dim ${DIM} --sigma ${SIGMA} --subsample 1 --burn_in 0.1 --n_samples 10 --n_test_samples 32 \
#      --ess_statistic hamming --seed ${SEED} &
#    done
#  done
#done

#
#BASE=/scratch/gobi2/gwohl/GWG_EXPERIMENTS/protein
#for DATA in OMPR_ECOLI #OPSD_BOVIN #CHEY_ECOLI PF00018 RNH_ECOLI YES_HUMAN BPT1_BOVIN ELAV4_HUMAN THIO_ALIAC CADH1_HUMAN O45418_CAEEL PCBP1_HUMAN
#do
#  for STEPS in 20
#  do
#    for SAMP in gwg gibbs
#    do
#      for L1 in .03 .01 .1
#      do
#        srun --gres=gpu:1 -c 8 --mem=10G -p gpu  \
#        python pcd_potts.py --save_dir ${BASE}/${DATA}/sampler_${SAMP}_steps_${STEPS}_l1_${L1}_REDO \
#        --model dense_potts --data ${DATA} --sampler ${SAMP} \
#        --sampling_steps ${STEPS} --print_every 10 --lr .001 --viz_every 100 --l1 ${L1} \
#        --weight_decay .0001 --n_iters 10000 \
#        --batch_size 256 --buffer_size 256 --contact_cutoff 5 &
#      done
#    done
#  done
#done

#
#BASE=/scratch/gobi2/gwohl/GWG_EXPERIMENTS/protein_low_lr
#for DATA in OMPR_ECOLI OPSD_BOVIN CHEY_ECOLI PF00018 # RNH_ECOLI BPT1_BOVIN ELAV4_HUMAN THIO_ALIAC CADH1_HUMAN O45418_CAEEL PCBP1_HUMAN
#do
#  for STEPS in 20
#  do
#    for SAMP in gwg gibbs plm
#    do
#      for L1 in .03 .01 .1
#      do
#        srun --gres=gpu:1 -c 8 --mem=10G -p gpu  \
#        python pcd_potts.py --save_dir ${BASE}/${DATA}/sampler_${SAMP}_steps_${STEPS}_l1_${L1} \
#        --model dense_potts --data ${DATA} --sampler ${SAMP} \
#        --sampling_steps ${STEPS} --print_every 10 --lr .0001 --viz_every 100 --l1 ${L1} \
#        --weight_decay .0001 --n_iters 10000 \
#        --batch_size 256 --buffer_size 256 --contact_cutoff 5 &
#      done
#    done
#  done
#done

BASE=/scratch/gobi2/gwohl/GWG_EXPERIMENTS/protein_very_big_buffer
for DATA in OMPR_ECOLI OPSD_BOVIN CHEY_ECOLI PF00018 RNH_ECOLI BPT1_BOVIN ELAV4_HUMAN THIO_ALIAC CADH1_HUMAN O45418_CAEEL PCBP1_HUMAN
do
  for STEPS in 20 50
  do
    for SAMP in gwg gibbs
    do
      for L1 in .03 .01 # .1
      do
        srun --gres=gpu:1 -c 8 --mem=10G -p gpu  \
        python pcd_potts.py --save_dir ${BASE}/${DATA}/sampler_${SAMP}_steps_${STEPS}_l1_${L1} \
        --model dense_potts --data ${DATA} --sampler ${SAMP} \
        --sampling_steps ${STEPS} --print_every 10 --lr .001 --viz_every 100 --l1 ${L1} \
        --weight_decay .0001 --n_iters 10000 \
        --batch_size 256 --buffer_size 10000 --contact_cutoff 5 &
      done
    done
  done
done


#BASE=/scratch/gobi2/gwohl/GWG_EXPERIMENTS/protein_high_lr
#for DATA in OMPR_ECOLI OPSD_BOVIN CHEY_ECOLI PF00018 # RNH_ECOLI BPT1_BOVIN ELAV4_HUMAN THIO_ALIAC CADH1_HUMAN O45418_CAEEL PCBP1_HUMAN
#do
#  for STEPS in 20
#  do
#    for SAMP in gwg gibbs plm
#    do
#      for L1 in .03 .01 .1
#      do
#        srun --gres=gpu:1 -c 8 --mem=10G -p gpu  \
#        python pcd_potts.py --save_dir ${BASE}/${DATA}/sampler_${SAMP}_steps_${STEPS}_l1_${L1} \
#        --model dense_potts --data ${DATA} --sampler ${SAMP} \
#        --sampling_steps ${STEPS} --print_every 10 --lr .003 --viz_every 100 --l1 ${L1} \
#        --weight_decay .0001 --n_iters 10000 \
#        --batch_size 256 --buffer_size 256 --contact_cutoff 5 &
#      done
#    done
#  done
#done
#
#BASE=/scratch/gobi2/gwohl/GWG_EXPERIMENTS/protein
#for DATA in OMPR_ECOLI OPSD_BOVIN CHEY_ECOLI PF00018 RNH_ECOLI BPT1_BOVIN ELAV4_HUMAN THIO_ALIAC CADH1_HUMAN O45418_CAEEL PCBP1_HUMAN
#do
#  for STEPS in 20
#  do
#    for SAMP in plm
#    do
#      for L1 in .03 .01 .1
#      do
#        #srun --gres=gpu:1 -c 8 --mem=10G -p gpu  \
#        python pcd_potts.py --save_dir ${BASE}/${DATA}/sampler_${SAMP}_steps_${STEPS}_l1_${L1}_weighted \
#        --model dense_potts --data ${DATA} --sampler ${SAMP} \
#        --sampling_steps ${STEPS} --print_every 10 --lr .001 --viz_every 100 --l1 ${L1} \
#        --weight_decay .0001 --n_iters 10000 \
#        --batch_size 256 --buffer_size 256 --contact_cutoff 5
#      done
#    done
#  done
#done