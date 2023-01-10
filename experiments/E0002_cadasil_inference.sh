#!/bin/bash
experiment_name="E0002_cadasil_inference"
criterion_=( "MSELoss" )
train_dataset_=( "None" )
train_dataset_args_=( "[]" )
train_dataset_kwargs_=( '{"N_subs":15}' )
test_dataset_=( "cadasil" )
test_dataset_args_=( "[]" )
test_dataset_kwargs_=( '{"N_timepoints":448}' )
model_=( "bilstm" )
model_args_=( "[448,1378]" )
model_kwargs_=( "{}" )
inference_model_=( "logs/E0001_first_model/run_0/checkpoints/model.best.pth" "logs/E0001_first_model/run_1/checkpoints/model.best.pth" "logs/E0001_first_model/run_2/checkpoints/model.best.pth" "logs/E0001_first_model/run_3/checkpoints/model.best.pth" "logs/E0001_first_model/run_4/checkpoints/model.best.pth" "logs/E0001_first_model/run_5/checkpoints/model.best.pth" "logs/E0001_first_model/run_6/checkpoints/model.best.pth" "logs/E0001_first_model/run_7/checkpoints/model.best.pth" "logs/E0001_first_model/run_8/checkpoints/model.best.pth" "logs/E0001_first_model/run_9/checkpoints/model.best.pth" )
optimizer_=( "Adam" )
optim_kwargs_=( "{}" )
batch_size_=( "64" )
lr_=( "1e-6" )
weight_decay_=( "0" )
num_folds_=( "10" )
epoch_=( "100" )
train_metrics_=( '["loss","correlation"]' )
test_metrics_=( '["loss","correlation"]' )
seed_=( "314159" )
k_=( 0 )
run=0
for criterion in "${criterion_[@]}"; do
for train_dataset in "${train_dataset_[@]}"; do 
for train_dataset_args in "${train_dataset_args_[@]}"; do 
for train_dataset_kwargs in "${train_dataset_kwargs_[@]}"; do
for test_dataset in "${test_dataset_[@]}"; do 
for test_dataset_args in "${test_dataset_args_[@]}"; do 
for test_dataset_kwargs in "${test_dataset_kwargs_[@]}"; do 
for model in "${model_[@]}"; do
for model_args in "${model_args_[@]}"; do
for model_kwargs in "${model_kwargs_[@]}"; do 
for inference_model in "${inference_model_[@]}"; do 
for optimizer in "${optimizer_[@]}"; do 
for optim_kwargs in "${optim_kwargs_[@]}"; do 
for batch_size in "${batch_size_[@]}"; do 
for lr in "${lr_[@]}"; do 
for weight_decay in "${weight_decay_[@]}"; do
for num_folds in "${num_folds_[@]}"; do 
for epoch in "${epoch_[@]}"; do 
for train_metrics in "${train_metrics_[@]}"; do 
for test_metrics in "${test_metrics_[@]}"; do
for seed in "${seed_[@]}"; do 
for k in "${k_[@]}"; do 
args=""
args=$args"--criterion "$criterion" "
args=$args"--train-dataset "$train_dataset" "
args=$args"--train-dataset-args "$train_dataset_args" "
args=$args"--train-dataset-kwargs "$train_dataset_kwargs" "
args=$args"--test-dataset "$test_dataset" "
args=$args"--test-dataset-args "$test_dataset_args" "
args=$args"--test-dataset-kwargs "$test_dataset_kwargs" "
args=$args"--model "$model" "
args=$args"--model-args "$model_args" "
args=$args"--model-kwargs "$model_kwargs" "
args=$args"--inference-model "$inference_model" "
args=$args"--optimizer "$optimizer" "
args=$args"--optim-kwargs "$optim_kwargs" "
args=$args"--batch-size "$batch_size" "
args=$args"--lr "$lr" "
args=$args"--weight-decay "$weight_decay" "
args=$args"--num-folds "$num_folds" "
args=$args"--epoch "$epoch" "
args=$args"--train-metrics "$train_metrics" "
args=$args"--test-metrics "$test_metrics" "
args=$args"--seed "$seed" "
args=$args"--k "$k" "
logdir="logs/"$experiment_name"/run_"$run
args=$args"--logdir "$logdir" "
sbatch -J $experiment_name"-"$run -e "slurm/logs/"$experiment_name"-"$run".err" -o "slurm/logs/"$experiment_name"-"$run".out" slurm/gpu_runner.sh $args
run=$((run+1))
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done 
done
done
done
done
done