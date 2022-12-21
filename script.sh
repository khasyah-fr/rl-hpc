#!/bin/bash
python main.py --title timeout-128hosts-llnl-60mins --platform-path platform-128.xml --workload-path workloads-llnl-cleaned-128host.json --timeout 3600
python main.py --title timeout-128hosts-llnl-30mins --platform-path platform-128.xml --workload-path workloads-llnl-cleaned-128host.json --timeout 1800
python main.py --title timeout-128hosts-llnl-15mins --platform-path platform-128.xml --workload-path workloads-llnl-cleaned-128host.json --timeout 900
python main.py --title baseline-128hosts-llnl --platform-path platform-128.xml --workload-path workloads-llnl-cleaned-128host.json --is-baseline True

