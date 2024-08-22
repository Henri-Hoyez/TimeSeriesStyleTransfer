#!/bin/bash

exp_names=("0" "2" "4" "6" "8" "10" "12" "14" "16" "18")
# exp_names=("6")

domain_shift="time_shift"
dataset_folder="data/simulated_dataset/"$domain_shift


for exp_name in ${exp_names[@]}
do 
    python train_style_transferV1.py --style1_dataset $dataset_folder"/8.h5" --style2_dataset $dataset_folder"/"$exp_name".h5" --exp_name $exp_name --exp_folder $domain_shift --epochs 100
    python eval.py --exp_folder $domain_shift --exp_name $exp_name
done 

# python train_style_transferV1.py --style1_dataset "data/simulated_dataset/time_shift/8.h5" --style2_dataset "data/simulated_dataset/time_shift/4.h5" --exp_name "4" --exp_folder "time_shift"
