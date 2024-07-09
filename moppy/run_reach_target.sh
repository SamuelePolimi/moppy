SEEDS=(2304 220 329 12 6064)
LERNING_RATES=(1E-2 5E-3 2E-3 1E-3 5E-4 2E-4 1E-4)
BETAS=(1E-0 5E-1 25E-2 1E-2 5E-2 2E-2 1E-2)
LATENT_VAR=(2 4 8 10)
ACTIVATION_FUNCS=("tanh" "relu", "sigmoid")

EPOCHS=1000

for ld in ${LATENT_VAR[@]}
do
    for af in ${ACTIVATION_FUNCS[@]}
    do
        for seed in ${SEEDS[@]}
        do
            for lr in ${LERNING_RATES[@]}
            do
                for beta in ${BETAS[@]}
                do
                    JOB_NAME="deep_pro_mp_seed_${seed}_beta_${beta}_lr_${lr}_ld_${ld}_af_${af}"
                    SAVE_FOLDER="./output/lr_${lr}/beta_${beta}/ld_${ld}/af_${af}/seed_${seed}"
                    params="--rnd_seed $seed --learning_rate $lr --beta $beta --save_path $SAVE_FOLDER --latent_dim $ld --activation_var $af"
                    echo "$params"
                    echo "$JOB_NAME"
                    # python main_sinus.py $params
                    sbatch --job-name="$JOB_NAME" --output="./out/$JOB_NAME.out" run_sinus_main.sh "$params"
                done
            done    
        done 
    done
done   