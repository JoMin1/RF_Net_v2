import os
os.system(r"cd E:\Anomlay_detection\skip-ganomaly-master")

# classes = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
classes = ['pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']


epoch = 300
nz = 500
isize = 256
ch = 64
gpu_ids = 0
batchsize = 1
sif = 100
model = "skipganomaly"

""" Unsupervised """


""" Semi_supervised """
# ocmd = r'python train.py --dataset Un_{}_{} --dataroot E:\sample\mvtec_semi_Abnormal\{} --nz {} --ngf {} --ndf {} --niter {} --isize {} --gpu_ids {} --verbose --lr 0.0002  --batchsize {} --save_image_freq {} --model {}'
ocmd = r'python train.py --dataset F-Net_semi2_{}_{} --dataroot E:\sample\mvtec_semi_Abnormal\{} --nz {} --ngf {} --ndf {} --niter {} --isize {} --gpu_ids {} --verbose --lr 0.0002  --batchsize {} --save_image_freq {} --model {}'

for i in range(1):
	count = 1
	for c in classes:
		cmd = ocmd.format(c, i, c,nz,ch,ch,epoch,isize, gpu_ids, batchsize, sif, model)
		print(cmd)
		os.system(cmd)
		count += 1


classes = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
ocmd = r'python train.py --dataset F-Net_Un_{}_{} --dataroot E:\sample\mvtec_semi_NoAbnormal\{} --nz {} --ngf {} --ndf {} --niter {} --isize {} --gpu_ids {} --verbose --lr 0.0002  --batchsize {} --save_image_freq {} --model {}'

for i in range(2):
	count = 1
	for c in classes:
		cmd = ocmd.format(c, i, c,nz,ch,ch,epoch,isize, gpu_ids, batchsize, sif, model)
		print(cmd)
		os.system(cmd)
		count += 1