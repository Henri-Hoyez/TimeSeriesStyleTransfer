#!/bin/bash

exp_names=("1.0_1.0" "2.0_2.0" "3.0_3.0" "4.0_4.0" "5.0_5.0" "6.0_6.0" "7.0_7.0" "8.0_8.0" "9.0_9.0" "10.0_10.0")
# exp_names=("2.0_2.0")

domain_shift="amplitude_shift"
dataset_folder="data/simulated_dataset/"$domain_shift


for exp_name in ${exp_names[@]}
do 
    python train_style_transferV1.py --style1_dataset $dataset_folder"/3.0_3.0.h5" --style2_dataset $dataset_folder"/"$exp_name".h5" --exp_name $exp_name --exp_folder $domain_shift
    python eval.py --exp_folder $domain_shift --exp_name $exp_name
done    

# python train_style_transferV1.py --style1_dataset "data/simulated_dataset/amplitude_shift/3.0_3.0.h5" --style2_dataset "data/simulated_dataset/amplitude_shift/2.0_2.0.h5" --exp_name "2.0_2.0 Adam thr 0.9 incr lr" --exp_folder "2.0_2.0"
