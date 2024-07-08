SEEDS=(2304 220 329 12 6064)
LERNING_RATES=(1E-2 5E-3 2E-3 1E-3 5E-4 2E-4 1E-4)
BETAS=(1E-0 5E-1 25E-2 1E-2 5E-2 2E-2 1E-2)

for seed in ${SEEDS[@]}
do
    for lr in ${LERNING_RATES[@]}
    do
        for beta in ${BETAS[@]}
        do
            JOB_NAME="deep_pro_mp_seed_${seed}_beta_${beta}_lr_${lr}"
            SAVE_FOLDER="./output/seed_${seed}/lr_${lr}/beta_${beta}"
            params="--rnd_seed $seed --learning_rate $lr --beta $beta --save_path $SAVE_FOLDER"
            echo "$params"
            echo "$JOB_NAME"
            # python main_sinus.py $params
            sbatch --job_name "$JOB_NAME" --output "$JOB_NAME.out" run_sinus_main.sh $params
        done
    done    
done    