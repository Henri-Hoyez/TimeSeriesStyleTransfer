#!/bin/bash

exp_names=("0.25" "0.50" "0.75" "1.00" "1.25" "1.50" "1.75" "2.00" "2.25" "2.50")
# exp_names=("1.75")

domain_shift="output_noise"
dataset_folder="data/simulated_dataset/"$domain_shift


for exp_name in ${exp_names[@]}
do 
    python train_style_transferV1.py --style1_dataset $dataset_folder"/1.25.h5" --style2_dataset $dataset_folder"/"$exp_name".h5" --exp_name $exp_name --exp_folder $domain_shift
    python eval.py --exp_folder $domain_shift --exp_name $exp_name
done 

# python train_style_transferV1.py --style1_dataset "data/simulated_dataset/output_noise/1.25.h5" --style2_dataset "data/simulated_dataset/output_noise/1.75.h5" --exp_name "1.75" --exp_folder "output_noise"
