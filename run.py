import os

print('---------------------------------------------------------------------------------')
cmd = f'torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
        --bsz 4 \
        --images images_4 \
        --eval --llffhold 83 \
        -s /data2/jtx/data/rubble \
        -m output/rubble_4_2 \
        --iterations 200000 \
        --densify_from_iter 1000 --densify_until_iter 50000 \
        --densification_interval 200 \
        --densify_grad_threshold 0.00013 \
        --percent_dense 0.003 \
        --opacity_reset_interval 9000 \
        --test_iterations 7000 30000 50000 80000 110000 120000 140000 160000 180000 200000 \
        --save_iterations 30000 50000 120000 200000 \
        --checkpoint_iterations 50000 120000 200000'
#print(cmd)
#os.system(cmd)

print('---------------------------------------------------------------------------------')
cmd = f'python render.py \
        -m output/rubble_4_2 \
        --iteration 119997 \
        --skip_train \
        --llffhold 83'
print(cmd)
os.system(cmd)

print('---------------------------------------------------------------------------------')
cmd = f'python metrics.py \
        -m output/rubble_4_2 \
        --mode test'
print(cmd)
os.system(cmd)


# for idx, scene in enumerate(['DJI_qixingpark_crossroad', 'DJI_village_building0_1_sq0', 'DJI_village_building1_1_sq0']):
#     print('---------------------------------------------------------------------------------')
#     cmd = f'torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py --bsz 4 -s /data2/liuzhi/Dataset/3DGS_Dataset/input/{scene}/train -m output/{scene}'
#     print(cmd)
#     os.system(cmd)
#
#     print('---------------------------------------------------------------------------------')
#     cmd = f'python render.py -m output/{scene}'
#     print(cmd)
#     os.system(cmd)
#
#     print('---------------------------------------------------------------------------------')
#     cmd = f'python metrics.py -m output/{scene}'
#     print(cmd)
#     os.system(cmd)
