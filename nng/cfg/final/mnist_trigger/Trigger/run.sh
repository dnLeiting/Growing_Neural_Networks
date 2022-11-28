#! /usr/bin/env bash
# this is a script

export WANDB_API_KEY=2ff970f2f3e90f08e7499d6ec2f9e7a384e0dfce

echo "Start"

for f in $(ls *.yaml);
do
	echo $f
	python3 ../../../../../main.py --config-file "$f" > "${f%.*}.log"
done