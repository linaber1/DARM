#python -m pts.train.generate_latents

#cd /home/abdulrahman.mahmoud/HEAKL/PTS
#source ~/.bashrc
#conda activate .heakl
#module load nvidia/cuda/11.8

#set -a
#source .venv
#set +a


source llada_env/bin/activate
cd /home/berrayan/planner/planner_executor_DiscreteDiffusion/

export PYTHONPATH="$(pwd):${PYTHONPATH}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0 python pts/train/generate_latents.py --dataset dart-1 --num_samples 1000 --start_index 3500 --output_suffix dart1.3 &
CUDA_VISIBLE_DEVICES=1 python pts/train/generate_latents.py --dataset dart-2 --num_samples 1000 --start_index 3500 --output_suffix dart4.3 &
CUDA_VISIBLE_DEVICES=2 python pts/train/generate_latents.py --dataset dart-3 --num_samples 1000 --start_index 3500 --output_suffix dart3.3 &
CUDA_VISIBLE_DEVICES=3 python pts/train/generate_latents.py --dataset dart-4 --num_samples 1000 --start_index 3500 --output_suffix dart2.3 &
#CUDA_VISIBLE_DEVICES=4 python pts/train/generate_latents.py --dataset dart-1 --num_samples 1000 --start_index 0 --output_suffix dart1 &

wait

#export CUDA_VISIBLE_DEVICES=1
#python pts/train/generate_latents.py --dataset "dart-5" --num_samples 5000

# for dataset in "arc_challenge" "dart-1" "dart-2" "dart-3" "dart-4" "dart-5" "gsm8k":
# do
#export CUDA_VISIBLE_DEVICES=0
#python pts/train/generate_latents.py --dataset "dart-4" --num_samples 5000


#export CUDA_VISIBLE_DEVICES=1
#python pts/train/generate_latents.py --dataset "dart-5" --num_samples 5000

#export CUDA_VISIBLE_DEVICES=1
#python pts/train/generate_latents.py --dataset "dart-2" --num_samples 5000


#export CUDA_VISIBLE_DEVICES=3
#python pts/train/generate_latents.py --dataset "dart-3" --num_samples 5000