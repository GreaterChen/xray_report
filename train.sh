#!/bin/bash
current_time=$(date "+%Y-%m-%d_%H:%M:%S")
nohup python -u train_full.py >> ./logfiles/$current_time.log 2>&1 &
jobs -l >> ./logfiles/$current_time.log 2>&1
tail -f ./logfiles/$current_time.log