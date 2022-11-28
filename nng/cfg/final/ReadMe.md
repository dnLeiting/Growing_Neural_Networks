# Final Tests
In here the config files for the final experiments are located.
All final experiments have been executed on a GCP VM e2-custom (4 vCPUs, 8 GB memory).
The folder structure is setup in the following way: \
Dataset -> When Function -> Where Function -> How Function.yaml 

For running the experiments shell scripts in the dataset folders have been created. Each shell script will run all experiment within a dataset folder.

```
cd mnist

bash run.sh
``` 

For running the experiments with Input Based Gaussian as weight initialization method a seed has been used. Hence, the results are not  directly reproducible for this how method. Since different initialization weights will influence the performance of the model, this has a direct influence on the growth of the model. Therefore, it can accrue that even with the same HOW, WHEN and WHERE methods the final models will end up with a different model architecture. Hence, this is subject of improvement if further work will be done on this project.
