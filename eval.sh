#!/bin/bash

exp_names=("0.5_0.5" "1.0_1.0" "1.5_1.5" )

for exp_name in ${exp_names[@]}
do 
    python eval.py --exp_name $exp_name
done 