# Creating an environment

## Docker

We provide a docker image to ensure repetibility of our results.
To build the image, you can execute the following commends in the project folder:

```bash
# build docker image, using the Dockerfile of this project, and tag it as xaicu113
docker build -t xaicu113 .
```

Once the image has been created, the docker container can be executed locally while mounting the project directory into it. Please make sure that the docker engine is cuda compatible.

```bash
# run image and mount current porject dir
docker run -d --rm --name xaicu113 --shm-size 1G --gpus all -v $(pwd):/home/testbench_xai xaicu113 bash
```

This command will run the container in an detached mode. To connect to it, run:

```bash
# run image and mount current porject dir
docker attach xaicu113
```

To disconnect from it, without terminating the container execution, you can press: 
ctrl + p, then press ctrl + q .

Finally, to terminate the container, execute the ```exit``` command while connected to it.

## Python environment

The environment for the execution of the current project is based on python 3.9.6, and requires cuda 11.3 (both present in the docker container). To create, activate, install, or deactivate the python environment, execute:

# setup venv
```bash
# creates a virtualenv
python3 -m venv venv
# activates the virtualenv
source venv/bin/activate
# install requirements
pip install -r requirements.txt
# deactivate virtual environment
deactivate
```

# Executing processes

Given the resource intensive nature of each step in the experimentation process, each run has been divided in smaller steps, which can be executed through the command line. The used configurations for each execution are stored in ```./data/configs/```. All functions are exposed thorugh the run.py script, and use the fire library to enable a straightforward execution.

## Training a model

To train a single model, two resorces must exist in the project folder.
First, the dataset must be available in ```./data/datasets/```.
Second, the configurations for the model architecture, dataset and training regime must be present at ```./data/configs```
After ensuring that these are present, execute:

```bash
python run.py train --path_model_config="./data/configs/model_${model}.json" --path_dataset_config="./data/configs/dataset_${dataset}.json" --path_training_config="./data/configs/training_${regime}.json" --path_output="./data/models/${dataset}_${model}_${regime}_${seed}" --seed=${seed} 2>&1 | tee -a "./data/logs/training_${dataset}_${model}_${regime}_${seed}.txt"

# where:
# ${model}: model architecture to train
# ${dataset}: name of the dataset to use for training
# ${regime}: training regime, all synthetic dataset experiments were run with the plateau (reduce on plateau) training regime.
# ${seed}: random seed to use while training
```

For example, to train a resnet18, for the AB dataset, with a plateau regime, with a random seed of 0, you can execute:

```bash
python run.py train --path_model_config="./data/configs/model_resnet18.json" --path_dataset_config="./data/configs/dataset_AB.json" --path_training_config="./data/configs/training_plateau.json" --path_output="./data/models/AB_resnet18_plateau_0" --seed=0 2>&1 | tee -a "./data/logs/training_AB_resnet18_plateau_0.txt"
```

## Executing ECLAD, ACE, and ConceptShap

Similar to the training, the configurations for the model, dataset, training and xai method must be present in ```./data/configs```.
To execute any of the methods, a model must have been previously trained. Then, execute:

```bash
# Execute ACE 
python run.py ace_analysis --path_model="./data/models/${dataset}_${model}_${regime}_${seed}" --path_output="./data/results/ace_${dataset}_${model}_${regime}_${seed}" --path_ace_config="./data/configs/ace_default.json" --seed=${seed} 2>&1 | tee -a "./data/logs/ace_${dataset}_${model}_${regime}_${seed}.txt"

# Execute ECLAD
python run.py eclad_analysis --path_model="./data/models/${dataset}_${model}_${regime}_${seed}" --path_output="./data/results/eclad_${dataset}_${model}_${regime}_${seed}_${variant}" --path_eclad_config="./data/configs/eclad_${variant}.json" --seed=${seed} 2>&1 | tee -a "./data/logs/eclad_${dataset}_${model}_${regime}_${seed}_${variant}.txt"

# Execute ConceptShap
python run.py cshap_analysis --path_model="./data/models/${dataset}_${model}_${regime}_${seed}" --path_output="./data/results/cshap_${dataset}_${model}_${regime}_${seed}_${variant}" --path_cshap_config="./data/configs/cshap_${variant}.json" --seed=${seed} 2>&1 | tee -a "./data/logs/cshap_${dataset}_${model}_${regime}_${seed}_${variant}.txt"

# where:
# ${model}: model architecture to train
# ${dataset}: name of the dataset to use for training
# ${regime}: training regime, all synthetic dataset experiments were run with the plateau (reduce on plateau) training regime.
# ${seed}: random seed to use while training
# ${variant}: refers to the configuration variant of the analysis method
```

For example, to analyse a resnet18 trained in the AB dataset, with a plateau regime, with a random seed of 0, you can execute:

```bash
# Execute ACE 
python run.py ace_analysis --path_model="./data/models/AB_resnet18_plateau_0" --path_output="./data/results/ace_AB_resnet18_plateau_0" --path_ace_config="./data/configs/ace_default.json" --seed=0 2>&1 | tee -a "./data/logs/ace_AB_resnet18_plateau_0.txt"

# Execute ECLAD
python run.py eclad_analysis --path_model="./data/models/AB_resnet18_plateau_0" --path_output="./data/results/eclad_AB_resnet18_plateau_0_${variant}" --path_eclad_config="./data/configs/eclad_n10s.json" --seed=0 2>&1 | tee -a "./data/logs/eclad_AB_resnet18_plateau_0_n10s.txt"

# Execute ConceptShap
python run.py cshap_analysis --path_model="./data/models/AB_resnet18_plateau_0" --path_output="./data/results/cshap_AB_resnet18_plateau_0_${variant}" --path_cshap_config="./data/configs/cshap_L7.json" --seed=0 2>&1 | tee -a "./data/logs/cshap_AB_resnet18_plateau_0_L7.txt"
```

## Associating concepts to primitives 

Once a model has been trained, the association can be performed with the following command (example for ECLAD):

```bash
# associate concepts and primitives
python run.py associate_CE --path_dataset_config="./data/configs/dataset_${dataset}.json" --path_model="./data/models/${dataset}_${model}_${regime}_${seed}" --path_output="./data/association/eclad_${dataset}_${model}_${regime}_${seed}_${variant}" --path_analysis="./data/results/eclad_${dataset}_${model}_${regime}_${seed}_${variant}" --force=True 2>&1 | tee -a "./data/logs/eclad_association_${dataset}_${model}_${regime}_${seed}_${variant}.txt"

# where:
# ${model}: model architecture to train
# ${dataset}: name of the dataset to use for training
# ${regime}: training regime, all synthetic dataset experiments were run with the plateau (reduce on plateau) training regime.
# ${seed}: random seed to use while training
# ${variant}: refers to the configuration variant of the analysis method
```

An examle for executing this association for one ECLAD analyis:

```bash
python run.py scatterplot_report_CE --path_dataset_config="./data/configs/dataset_AB.json" --path_model="./data/models/AB_resnet18_plateau_0" --path_output="./data/reports/cshap_AB_resnet18_plateau_0_n10s" --path_analysis="./data/results/cshap_AB_resnet18_plateau_0_n10s" --path_association="data/association/cshap_AB_resnet18_plateau_0_n10s" 2>&1 | tee -a "./data/logs/cshap_reports_AB_resnet18_plateau_0_n10s.txt"
```

## Scatter plots

A similar script exists for creating the scatter plots shown in the paper.

```bash
# create a scatter plot
python run.py scatterplot_report_CE --path_dataset_config="./data/configs/dataset_${dataset}.json" --path_model="./data/models/${dataset}_${model}_${regime}_${seed}" --path_output="./data/reports/eclad_${dataset}_${model}_${regime}_${seed}_${variant}" --path_analysis="./data/results/eclad_${dataset}_${model}_${regime}_${seed}_${variant}" --path_association="data/association/eclad_${dataset}_${model}_${regime}_${seed}_${variant}" 2>&1 | tee -a "./data/logs/eclad_reports_${dataset}_${model}_${regime}_${seed}_${variant}.txt"

# where:
# ${model}: model architecture to train
# ${dataset}: name of the dataset to use for training
# ${regime}: training regime, all synthetic dataset experiments were run with the plateau (reduce on plateau) training regime.
# ${seed}: random seed to use while training
# ${variant}: refers to the configuration variant of the analysis method
```

## Comparison (boxplots)

Finally, the boxplot comparison and extended visualizations can be generated by executing:

```bash
# boxplots
python ./scripts/boxplots.py
# reports
python ./scripts/reports.py
```

## Example pipeline

An example pipeline for the executin of these steps for multiple random seeds can be seen in ```run.sh```
