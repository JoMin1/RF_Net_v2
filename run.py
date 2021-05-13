import os
os.system(r"cd E:\Anomlay_detection\skip-ganomaly-master")

classes = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
# classes = ['carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw']
# classes = ['screw']

# classes = ['bottle', 'cable', 'capsule', 'carpet']
classes = ['grid', 'hazelnut', 'leather', 'metal_nut']
# classes = ['pill', 'screw', 'tile', 'toothbrush']
# classes = ['transistor', 'wood', 'zipper']

epoch = 300
nz = 500
isize = 256
ch = 64
gpu_ids = 1
batchsize = 1
sif = 100000

""" Unsupervised """


""" Semi_supervised """
# ocmd = r'python train.py --dataset Un_{}_{} --dataroot H:\anomaly\mvtec_semi_NoAbnormal\{} --nz {} --ngf {} --ndf {} --niter {} --isize {} --gpu_ids {} --verbose --lr 0.0002  --batchsize {}'
ocmd = r'python train.py --dataset semi_{}_{} --dataroot H:\anomaly\mvtec_semi_Abnormal\{} --nz {} --ngf {} --ndf {} --niter {} --isize {} --gpu_ids {} --verbose --lr 0.0002  --batchsize {} --save_image_freq {}'

for i in range(3):
	count = 1
	for c in classes:
		cmd = ocmd.format(c, i + 2, c,nz,ch,ch,epoch,isize, gpu_ids, batchsize, sif)
		print(cmd)
		os.system(cmd)
		count += 1