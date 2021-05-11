# Generate lattice ising data
for DIM in 10 25 50
do
  for SIGMA in -.1 0.0 .1 .25 .5 1.
    do
      srun --gres=gpu:1 -c 8 --mem=10G -p gpu python pcd.py \
            --save_dir /scratch/gobi2/gwohl/GWG_DATASETS/ising_dim_${DIM}_sigma_${SIGMA} \
            --model lattice_ising --data_model lattice_ising --dim ${DIM} --sigma ${SIGMA} &
  done
done


# Generate lattice potts data
for DIM in 10 25 50
do
  for SIGMA in -.1 0.0 .1 .25 .5 1.
    do
      srun --gres=gpu:1 -c 8 --mem=10G -p gpu python pcd.py \
            --save_dir /scratch/gobi2/gwohl/GWG_DATASETS/potts_dim_${DIM}_sigma_${SIGMA} \
            --model lattice_potts --data_model lattice_potts --dim ${DIM} --sigma ${SIGMA} &
  done
done


# Generate Erdos-Renyi Ising data
for SEED in 1111 2222 3333 4444 5555
do
  echo python pcd.py --save_dir /scratch/gobi2/gwohl/GWG_DATASETS/ising_er_nodes_100_conn_2_seed_${SEED} \
  --data_model er_ising --dim 100 --gt_steps 1000000 --degree 2 --seed ${SEED}
done

for SEED in 1111 2222 3333 4444 5555
do
  echo python pcd.py --save_dir /scratch/gobi2/gwohl/GWG_DATASETS/ising_er_nodes_200_conn_4_seed_${SEED} \
  --data_model er_ising --dim 200 --gt_steps 1000000 --degree 4 --seed ${SEED}
done
