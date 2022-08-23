#!/bin/bash
#source activate xai
source venv/bin/activate

# which python is being used?
python -c "import sys; print(f'Using python: {sys.executable}')"

for seed in 0 1 2 3 4 5
do  
    for model in resnet18 resnet34 densenet121 efficientnet_b0 vgg16
    do
        for dataset in AB ABplus BigSmall CO colorGB isA metal_nut leather
        do
            case $dataset in
                "concrete_cracks") regime="plateaushort" ;;
                "casting_data") regime="plateaufastcolor" ;;
                "aptos") regime="plateaufast512" ;;
                 *) regime="plateau" ;;
            esac
            
            echo "$processing: {dataset} ${model} ${regime} ${seed}"
            python run.py train --path_model_config="./data/configs/model_${model}.json" --path_dataset_config="./data/configs/dataset_${dataset}.json" --path_training_config="./data/configs/training_${regime}.json" --path_output="./data/models/${dataset}_${model}_${regime}_${seed}" --seed=${seed} 2>&1 | tee -a "./data/logs/training_${dataset}_${model}_${regime}_${seed}.txt"
            python run.py model_eval --path_model="./data/models/${dataset}_${model}_${regime}_${seed}" --path_output="./data/models_eval/${dataset}_${model}_${regime}_${seed}" --seed=${seed} 2>&1 | tee -a "./data/logs/model_eval_${dataset}_${model}_${regime}_${seed}.txt"
            
            python run.py ace_analysis --path_model="./data/models/${dataset}_${model}_${regime}_${seed}" --path_output="./data/results/ace_${dataset}_${model}_${regime}_${seed}" --path_ace_config="./data/configs/ace_default.json" --seed=${seed} 2>&1 | tee -a "./data/logs/ace_${dataset}_${model}_${regime}_${seed}.txt"
            python run.py associate_CE --path_dataset_config="./data/configs/dataset_${dataset}.json" --path_model="./data/models/${dataset}_${model}_${regime}_${seed}" --path_output="./data/association/ace_${dataset}_${model}_${regime}_${seed}" --path_analysis="./data/results/ace_${dataset}_${model}_${regime}_${seed}" --force=True 2>&1 | tee -a "./data/logs/ace_association_${dataset}_${model}_${regime}_${seed}.txt"
            python run.py scatterplot_report_CE --path_dataset_config="./data/configs/dataset_${dataset}.json" --path_model="./data/models/${dataset}_${model}_${regime}_${seed}" --path_output="./data/reports/ace_${dataset}_${model}_${regime}_${seed}" --path_analysis="./data/results/ace_${dataset}_${model}_${regime}_${seed}" --path_association="data/association/ace_${dataset}_${model}_${regime}_${seed}" 2>&1 | tee -a "./data/logs/ace_reports_${dataset}_${model}_${regime}_${seed}.txt"

            for variant in n10s
            do
                echo "analyzing eclad: ${variant}"
                python run.py eclad_analysis --path_model="./data/models/${dataset}_${model}_${regime}_${seed}" --path_output="./data/results/eclad_${dataset}_${model}_${regime}_${seed}_${variant}" --path_eclad_config="./data/configs/eclad_${variant}.json" --seed=${seed} 2>&1 | tee -a "./data/logs/eclad_${dataset}_${model}_${regime}_${seed}_${variant}.txt"
                python run.py associate_CE --path_dataset_config="./data/configs/dataset_${dataset}.json" --path_model="./data/models/${dataset}_${model}_${regime}_${seed}" --path_output="./data/association/eclad_${dataset}_${model}_${regime}_${seed}_${variant}" --path_analysis="./data/results/eclad_${dataset}_${model}_${regime}_${seed}_${variant}" --force=True 2>&1 | tee -a "./data/logs/eclad_association_${dataset}_${model}_${regime}_${seed}_${variant}.txt"
                python run.py scatterplot_report_CE --path_dataset_config="./data/configs/dataset_${dataset}.json" --path_model="./data/models/${dataset}_${model}_${regime}_${seed}" --path_output="./data/reports/eclad_${dataset}_${model}_${regime}_${seed}_${variant}" --path_analysis="./data/results/eclad_${dataset}_${model}_${regime}_${seed}_${variant}" --path_association="data/association/eclad_${dataset}_${model}_${regime}_${seed}_${variant}" 2>&1 | tee -a "./data/logs/eclad_reports_${dataset}_${model}_${regime}_${seed}_${variant}.txt"
            done

            for variant in L7
            do
                echo "analyzing cshap: ${variant}"
                python run.py cshap_analysis --path_model="./data/models/${dataset}_${model}_${regime}_${seed}" --path_output="./data/results/cshap_${dataset}_${model}_${regime}_${seed}_${variant}" --path_cshap_config="./data/configs/cshap_${variant}.json" --seed=${seed} 2>&1 | tee -a "./data/logs/cshap_${dataset}_${model}_${regime}_${seed}_${variant}.txt"
                python run.py associate_CE --path_dataset_config="./data/configs/dataset_${dataset}.json" --path_model="./data/models/${dataset}_${model}_${regime}_${seed}" --path_output="./data/association/cshap_${dataset}_${model}_${regime}_${seed}_${variant}" --path_analysis="./data/results/cshap_${dataset}_${model}_${regime}_${seed}_${variant}" --force=True 2>&1 | tee -a "./data/logs/cshap_association_${dataset}_${model}_${regime}_${seed}_${variant}.txt"
                python run.py scatterplot_report_CE --path_dataset_config="./data/configs/dataset_${dataset}.json" --path_model="./data/models/${dataset}_${model}_${regime}_${seed}" --path_output="./data/reports/cshap_${dataset}_${model}_${regime}_${seed}_${variant}" --path_analysis="./data/results/cshap_${dataset}_${model}_${regime}_${seed}_${variant}" --path_association="data/association/cshap_${dataset}_${model}_${regime}_${seed}_${variant}" 2>&1 | tee -a "./data/logs/cshap_reports_${dataset}_${model}_${regime}_${seed}_${variant}.txt"
            done
        done
    done
done