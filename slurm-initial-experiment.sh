#!/bin/bash
#
#SBATCH --job-name=mdvrptw-batch
#SBATCH --output=logs/%A.out
#SBATCH --error=logs/%A.err
#
#SBATCH --time=2-00:00:00
#SBATCH --nodelist=komputasi03

nix-user-chroot ~/.nix bash
nix-shell ~/sds-rl/env.nix
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sds-rl

cd ~/sds-rl/sds-rl

# 8 hosts
# GAIA
python main.py --title timeout-8hosts-GAIA-60mins --platform-path platform.xml --workload-path workloads-gaia-cleaned-8host.json --timeout 3600 &
python main.py --title timeout-8hosts-GAIA-30mins --platform-path platform.xml --workload-path workloads-gaia-cleaned-8host.json --timeout 1800 &
wait;
python main.py --title timeout-8hosts-GAIA-15mins --platform-path platform.xml --workload-path workloads-gaia-cleaned-8host.json --timeout 900 &
python main.py --title baseline-8hosts-GAIA --platform-path platform.xml --workload-path workloads-gaia-cleaned-8host.json --is-baseline True &
wait;
# NASA
python main.py --title timeout-8hosts-nasa-60mins --platform-path platform.xml --workload-path workloads-nasa-cleaned-8host.json --timeout 3600 &
python main.py --title timeout-8hosts-nasa-30mins --platform-path platform.xml --workload-path workloads-nasa-cleaned-8host.json --timeout 1800 &
wait;
python main.py --title timeout-8hosts-nasa-15mins --platform-path platform.xml --workload-path workloads-nasa-cleaned-8host.json --timeout 900 &
python main.py --title baseline-8hosts-nasa --platform-path platform.xml --workload-path workloads-nasa-cleaned-8host.json --is-baseline True &
wait;
# LLNL
python main.py --title timeout-8hosts-llnl-60mins --platform-path platform.xml --workload-path workloads-llnl-cleaned-8host.json --timeout 3600 &
python main.py --title timeout-8hosts-llnl-30mins --platform-path platform.xml --workload-path workloads-llnl-cleaned-8host.json --timeout 1800 &
wait;
python main.py --title timeout-8hosts-llnl-15mins --platform-path platform.xml --workload-path workloads-llnl-cleaned-8host.json --timeout 900 &
python main.py --title baseline-8hosts-llnl --platform-path platform.xml --workload-path workloads-llnl-cleaned-8host.json --is-baseline True &
wait;


# 128 hosts
# GAIA
python main.py --title timeout-128hosts-GAIA-60mins --platform-path platform.xml --workload-path workloads-gaia-cleaned-128host.json --timeout 3600 &
python main.py --title timeout-128hosts-GAIA-30mins --platform-path platform.xml --workload-path workloads-gaia-cleaned-128host.json --timeout 1800 &
wait;
python main.py --title timeout-128hosts-GAIA-15mins --platform-path platform.xml --workload-path workloads-gaia-cleaned-128host.json --timeout 900 &
python main.py --title baseline-128hosts-GAIA --platform-path platform.xml --workload-path workloads-gaia-cleaned-128host.json --is-baseline True &
wait;
# NASA
python main.py --title timeout-128hosts-nasa-60mins --platform-path platform.xml --workload-path workloads-nasa-cleaned-128host.json --timeout 3600 &
python main.py --title timeout-128hosts-nasa-30mins --platform-path platform.xml --workload-path workloads-nasa-cleaned-128host.json --timeout 1800 &
wait;
python main.py --title timeout-128hosts-nasa-15mins --platform-path platform.xml --workload-path workloads-nasa-cleaned-128host.json --timeout 900 &
python main.py --title baseline-128hosts-nasa --platform-path platform.xml --workload-path workloads-nasa-cleaned-128host.json --is-baseline True &
wait;
# LLNL
python main.py --title timeout-128hosts-llnl-60mins --platform-path platform.xml --workload-path workloads-llnl-cleaned-128host.json --timeout 3600 &
python main.py --title timeout-128hosts-llnl-30mins --platform-path platform.xml --workload-path workloads-llnl-cleaned-128host.json --timeout 1800 &
wait;
python main.py --title timeout-128hosts-llnl-15mins --platform-path platform.xml --workload-path workloads-llnl-cleaned-128host.json --timeout 900 &
python main.py --title baseline-128hosts-llnl --platform-path platform.xml --workload-path workloads-llnl-cleaned-128host.json --is-baseline True &
wait;
