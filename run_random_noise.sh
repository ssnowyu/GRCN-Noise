#!/bin/bash
# remember to add --save when you want to save the experiment log

GPU=$1

DATASET=$3
MODELS=$2
ARGS="$4 $5 $6"
# edge sampling ratio
SAMPLES=(1.0)
# number of training instances per class
TN=("20")
# hidden dimensions for graph revision module
HIDG=("100:10")

# SEED_ARRAY=(633 684 806 350 208 857 369 153 329 363)
SEED_ARRAY=(0 1 2 3 4 5 6 7 8 9)
# DATASEED_ARRAY=(103 219 977 678 527 549 368 945 573 920)
DATASEED_ARRAY=(0 1 2 3 4 5 6 7 8 9)

NOISE_RATE_ARRAY=(0.6 0.7 0.8 0.9 1.0)
SIGMA_ARRAY=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 10 20 30 50 100 200)

if [ ${MODELS} == 'GRCN' ]; then
    if [ ${DATASET} == "Cora" ]; then
        PARA=("30")
    elif [ ${DATASET} == "CiteSeer" ]; then
        PARA=("100")
    elif [ ${DATASET} == "PubMed" ]; then
        PARA=("5")
    elif [ ${DATASET} == "CoraFull" ]; then
        PARA=("20")
    elif [ ${DATASET} == "Computers" ]; then
        PARA=("30")
    elif [ ${DATASET} == "CS" ]; then
        PARA=("5")
    fi
elif [[ ${MODELS} == *"GRCN"* ]]; then
    if [ ${DATASET} == "Cora" ]; then
        PARA=("30")
    elif [ ${DATASET} == "CiteSeer" ]; then
        PARA=("100")
    elif [ ${DATASET} == "PubMed" ]; then
        PARA=("5")
    elif [ ${DATASET} == "CoraFull" ]; then
        PARA=("20")
    elif [ ${DATASET} == "Computers" ]; then
        PARA=("30")
    elif [ ${DATASET} == "CS" ]; then
        PARA=("5")
    fi
fi

for tn in "${TN[@]}"; do
    for sample in "${SAMPLES[@]}"; do
        for para in "${PARA[@]}"; do
            for hidg in "${HIDG[@]}"; do
                for i in {0..4}; do
                    for noise_rate in "${NOISE_RATE_ARRAY[@]}"; do
                        for sigma in "${SIGMA_ARRAY[@]}"; do
                            SEED=${SEED_ARRAY[i]}
                            DATASEED=${DATASEED_ARRAY[i]}
                            option="--dataset ${DATASET} --sample ${sample} --complete ${MODELS}
                                --compl_param ${para} --seed ${SEED} --dataseed ${DATASEED} ${ARGS}
                                --hid_graph ${hidg} --train_num ${tn} --noise_rate ${noise_rate} --sigma ${sigma}"
                            cmd="CUDA_VISIBLE_DEVICES=${GPU} python main_ours.py ${option}"
                            echo $cmd
                            eval $cmd
                        done
                    done
                done
            done
        done
    done
done
