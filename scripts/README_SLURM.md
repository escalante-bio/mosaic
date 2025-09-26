SLURM quick start (single node, 4 GPUs)

1) Environment (on a node or baked into your image)

   module load cuda/12.4
   bash scripts/setup_slurm_env.sh
   export XLA_FLAGS="--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found --xla_gpu_cuda_data_dir=/usr/local/cuda"
   export BOLTZ_CACHE=/scratch/$USER/boltz_cache

2) Run one job

   srun -N1 -n1 --gpus-per-node=4 --cpus-per-task=8 \
     python scripts/run_protrl_csv_slurm.py \
       --csv _external/selected_designs_getbest_Chai1.csv \
       --output /scratch/$USER/protrl_csv_run \
       --model-id AI4PD/ZymCTRL \
       --iterations 2 \
       --designs 2 \
       --reward "total_score=1.0,efield_score=1.0,ncaa_interface_score=1.0,boltz2_plddt=1.0,boltz2_iptm=1.0,length100=-1.0" \
       --csv-predictors --csv-cols "total_score,efield_score,ncaa_interface_score" \
       --clean-proxy --clean-ec-label "3.1.1.102" \
       --ligand-smiles "[C@H](O)(c1ccc(C(=O)O)cc1)Oc1ccc(cc1)N(=O)=O" \
       --reward-devices "0,1,2,3" \
       --xla-cuda-dir /usr/local/cuda

3) SLURM array for multiple seeds

   # seeds 0..9
   sbatch --array=0-9%4 <<'EOF'
   #!/usr/bin/env bash
   #SBATCH -N 1
   #SBATCH --gpus-per-node=4
   #SBATCH --cpus-per-task=8
   set -euo pipefail
   module load cuda/12.4
   export XLA_FLAGS="--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found --xla_gpu_cuda_data_dir=/usr/local/cuda"
   export BOLTZ_CACHE=/scratch/$USER/boltz_cache
   SEED=${SLURM_ARRAY_TASK_ID}
   OUT=/scratch/$USER/protrl_csv_run_seed${SEED}
   mkdir -p "$OUT"
   python scripts/run_protrl_csv_slurm.py \
     --csv _external/selected_designs_getbest_Chai1.csv \
     --output "$OUT" \
     --model-id AI4PD/ZymCTRL \
     --iterations 2 \
     --designs 2 \
     --reward "total_score=1.0,efield_score=1.0,ncaa_interface_score=1.0,boltz2_plddt=1.0,boltz2_iptm=1.0,length100=-1.0" \
     --csv-predictors --csv-cols "total_score,efield_score,ncaa_interface_score" \
     --clean-proxy --clean-ec-label "3.1.1.102" \
     --ligand-smiles "[C@H](O)(c1ccc(C(=O)O)cc1)Oc1ccc(cc1)N(=O)=O" \
     --reward-devices "0,1,2,3" \
     --xla-cuda-dir /usr/local/cuda
   EOF

Notes
- Trainer runs on GPU 0; reward workers run on listed GPUs.
- Use smaller ESM (t6_8M) for fast CSV predictor fitting.
- For true CLEAN scoring provide a proper CLEAN scorer; the "--clean-proxy" flag uses an ESM-based cosine to the EC centroid.


