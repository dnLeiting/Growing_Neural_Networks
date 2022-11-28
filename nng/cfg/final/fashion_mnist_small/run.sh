export WANDB_API_KEY=2ff970f2f3e90f08e7499d6ec2f9e7a384e0dfce

echo "started"

for a in */ ; do
    echo "$a"
    for d in "$a"*/ ; do
      echo "$d"
      for c in "$d"*.yaml ; do
        echo "$c"
        python3 ../../../main.py --config-file "$c"
      done
    done
done
