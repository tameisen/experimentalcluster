#!/bin/bash
echo "Setting CPU frequency to 1.5 GHz..."
for cpu in $(seq 0 $(($(nproc --all) - 1))); do
    sudo cpufreq-set -c $cpu -u 1.5GHz
    echo "CPU $cpu: $(cat /sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_max_freq)"
done
echo "CPU frequency set completed."
