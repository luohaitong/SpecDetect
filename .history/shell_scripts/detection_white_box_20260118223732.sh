# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
datasets_path=datasets/human_llm_data_for_experiment
perturbation_datasets_path=datasets/perturbation_data_detectgpt_npr
regeneration_datasets_path=datasets/regeneration_data_dnagpt
# results folders
statistic_detection_results_path=experiment_results/statistic_detection_results
detectgpt_results_path=experiment_results/detectgpt_detection_results
npr_results_path=experiment_results/npr_detection_results
dnagpt_results_path=experiment_results/dna_gpt_detection_results
fast_detectgpt_detection_results_path=experiment_results/fast_detectgpt_detection_results
lastde_doubleplus_detection_results_path=experiment_results/lastde_doubleplus_detection_results
specdetect_doubleplus_detection_results_path=experiment_results/specdetect_detection_results

# open-source
datasets="xsum squad writing" # xsum squad writing
source_models="gpt2_xl gptneo_2.7b opt_2.7b llama1_13b llama2_13b llama3_8b opt_13b bloom_7b phi-4 Qwen3-8B gemma3-1b gptj_6b"
scenarios="white"

# statistic methods：likelihood, logrank, entropy, lrr, lastde
for D in $datasets; do
for M in $source_models; do
   echo `date`, Preparing dataset ${D}_${M} ...
   CUDA_VISIBLE_DEVICES=6 python py_scripts/baselines/statistic_detect.py --dataset_file ${datasets_path}/${D}_${M} --output_file ${statistic_detection_results_path}/${scenarios}  --scoring_model_name ${M}
 done
done


# perturbation methods: detectgpt、npr、dna_gpt
for D in $datasets; do
 for M in $source_models; do
   echo `date`, Preparing dataset ${D}_${M} ...
   CUDA_VISIBLE_DEVICES=7 python py_scripts/baselines/detect_gpt.py --dataset_file ${perturbation_datasets_path}/${D}_${M} --output_file ${detectgpt_results_path}/${scenarios}  --main_results --scoring_model_name ${M}
 done
done


# for D in $datasets; do
 for M in $source_models; do
   echo `date`, Preparing dataset ${D}_${M} ...
   CUDA_VISIBLE_DEVICES=5 python py_scripts/baselines/detect_npr.py --dataset_file ${perturbation_datasets_path}/${D}_${M} --output_file ${npr_results_path}/${scenarios} --main_results  --scoring_model_name ${M}
 done
done


# for D in $datasets; do
 for M in $source_models; do
   echo `date`, Preparing dataset ${D}_${M} ...
   CUDA_VISIBLE_DEVICES=5 python py_scripts/baselines/dna_gpt.py  --dataset_file ${regeneration_datasets_path}/${D}_${M} --output_file ${dnagpt_results_path}/${scenarios}  --scoring_model_name ${M} --scenario ${scenarios}
 done
done

# sampling methods: fast_detectgpt, lastde++
for D in $datasets; do
 for M in $source_models; do
   echo `date`, Preparing dataset ${D}_${M} ...
   CUDA_VISIBLE_DEVICES=6 python py_scripts/baselines/fast_detect_gpt.py --dataset_file ${datasets_path}/${D}_${M} --output_file ${fast_detectgpt_detection_results_path}/${scenarios} --reference_model_name ${M} --scoring_model_name ${M}
 done
done


# for D in $datasets; do
 for M in $source_models; do
   echo `date`, Preparing dataset ${D}_${M} ...
   CUDA_VISIBLE_DEVICES=6 python py_scripts/baselines/lastde_doubleplus.py --dataset_file ${datasets_path}/${D}_${M} --output_file ${lastde_doubleplus_detection_results_path}/${scenarios} --reference_model_name ${M} --scoring_model_name ${M}  --embed_size 4 --epsilon 8 --tau_prime 15
 done
done

for D in $datasets; do
 for M in $source_models; do
   echo `date`, Preparing dataset ${D}_${M} ...
   CUDA_VISIBLE_DEVICES=6 python py_scripts/baselines/specdetect_doubleplus.py --dataset_file ${datasets_path}/${D}_${M} --output_file ${specdetect_doubleplus_detection_results_path}/${scenarios} --reference_model_name ${M} --scoring_model_name ${M}
 done
done
exit