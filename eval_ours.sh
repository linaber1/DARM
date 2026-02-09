

cd /home/berrayan/planner/planner_executor_DiscreteDiffusion
export PYTHONPATH=/home/berrayan/planner/planner_executor_DiscreteDiffusion
export CUDA_VISIBLE_DEVICES=0,1,2,3
#for name_architecture in "dl_dual"; do
#    for src_subset in "dart-1" "dart-2" "dart-3" "dart-4" "dart-5"  ; do
#        python pts/eval/eval_ours.py \
#            --config configs/dual_pipeline_dream.yaml \
#            --dataset $src_subset \
#            --num_samples 200 \
#            --name_architecture "$name_architecture" 
#    done
#done

for percentage in 100; do
    for name_architecture in "dl_dual"; do
        for src_subset in "dart-2"  ; do
            python pts/eval/eval_ours_.py \
                --config configs/dual_pipeline.yaml \
                --dataset $src_subset \
                --num_samples 200 \
                --name_architecture "$name_architecture" \
                --stop_early \
                --percentage $percentage
        done
    done
done