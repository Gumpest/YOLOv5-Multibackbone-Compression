# nohup python -m torch.distributed.launch --nproc_per_node 2 --master_port 2000 train.py --data VisDrone.yaml --weights yolov5s.pt --cfg models/yolov5s.yaml --epochs 300 --batch-size 8 --img 608 --nosave --device 2,3 --sync-bn >> yolov5s.out &
# nohup python -m torch.distributed.launch --nproc_per_node 2 --master_port 2001 train.py --data VisDrone.yaml --weights yolov5l.pt --cfg models/yolov5l.yaml --epochs 300 --batch-size 8 --img 608 --nosave --device 0,3 --sync-bn >> yolov5l.out &
# nohup python -m torch.distributed.launch --nproc_per_node 2 --master_port 2002 train.py --data VisDrone.yaml --weights yolov5x.pt --cfg models/yolov5x.yaml --epochs 300 --batch-size 8 --img 608 --nosave --device 0,3 --sync-bn >> yolov5x.out &
# nohup python -m torch.distributed.launch --nproc_per_node 1 --master_port 2002 train.py --data VisDrone.yaml --weights yolov5x.pt --cfg models/yolov5xP2.yaml --epochs 300 --batch-size 8 --img 640 --nosave --device 0,3 --sync-bn >> yolov5xP2.out &
# nohup python -m torch.distributed.launch --nproc_per_node 2 --master_port 2000 train.py --data VisDrone.yaml --weights yolov5x.pt --cfg models/yolov5xP2CBAM.yaml --epochs 300 --batch-size 8 --img 640 --nosave --device 2,3 --sync-bn >> yolov5xP2CBAM.out &
# nohup python -m torch.distributed.launch --nproc_per_node 2 --master_port 2000 train.py --data VisDrone.yaml --weights yolov5x.pt --cfg models/accModels/yolov5xP2CBAM-TPH-BiFPN-SPP.yaml --hyp data/hyps/hyp.visdrone.yaml --epochs 80 --batch-size 4 --img 1536 --device 0,1 --sync-bn --adam --name v5x-TPH >> yolov5xTPH.out &
# nohup python -m torch.distributed.launch --nproc_per_node 2 --master_port 2005 train.py --data VisDrone.yaml --weights yolov5l.pt --cfg models/yolov5lGhost.yaml --epochs 300 --batch-size 8 --img 640 --nosave --device 2,3 --sync-bn >> yolov5lGhost_V6.out &
# nohup python -m torch.distributed.launch --nproc_per_node 2 --master_port 2005 train.py --data VisDrone.yaml --weights yolov5l.pt --cfg models/yolov5lShffule.yaml --epochs 300 --batch-size 8 --img 640 --nosave --device 0,1 --sync-bn >> yolov5lShffule.out &
# nohup python -m torch.distributed.launch --nproc_per_node 2 --master_port 2006 train.py --data VisDrone.yaml --weights yolov5l.pt --cfg models/lightModels/yolov5lMobilenetv3Small.yaml --epochs 300 --batch-size 8 --img 640 --nosave --device 0,1 --sync-bn >> yolov5lMobilenetv3Small.out &
# nohup python -m torch.distributed.launch --nproc_per_node 2 --master_port 2002 train.py --data VisDrone.yaml --weights yolov5l.pt --cfg models/lightModels/yolov5lEfficientLite.yaml --epochs 300 --batch-size 8 --img 640 --nosave --device 0,2 --sync-bn >> yolov5lEfficientLite.out &
# nohup python -m torch.distributed.launch --nproc_per_node 2 --master_port 2000 train.py --data VisDrone.yaml --weights yolov5l.pt --cfg models/lightModels/yolov5lPP-LC.yaml --epochs 300 --batch-size 8 --img 640 --nosave --device 0,3 --sync-bn >> yolov5lPP-LC.out &
# nohup python -m torch.distributed.launch --nproc_per_node 2 --master_port 2003 train.py --data VisDrone.yaml --weights runs/train/exp152/weights/last-pruned.pt --cfg models/prunModels/yolov5s-0.6-pruned.yaml --epochs 220 --batch-size 8 --img 608 --nosave --device 0,1 --sync-bn >> yolov5spruned.out &
nohup python -m torch.distributed.launch --nproc_per_node 2 --master_port 2004 train.py --data VisDrone.yaml --weights runs/train/v5x-TPH/weights/best.pt --cfg models/accModels/yolov5xP2CBAM-Swin-BiFPN-SPP.yaml --hyp data/hyps/hyp.visdrone.yaml --epochs 80 --batch-size 4 --img 1536 --device 0,1 --sync-bn --adam --nohalf --name v5x-Swin >> yolov5xSwin.out &

# Prune Script
# nohup python pruneEagleEye.py --weights /home/zy/DVSTrack/yolov5/weights/yolon608.pt --cfg models/prunModels/yolov5n-pruning.yaml --data data/DVSPerson.yaml --path yolov5n-DVSPerson-pruned.yaml --max_iter 1400 --remain_ratio 0.6 --delta 0.15 >> DVSsearch.out &
# nohup python pruneEagleEye.py --weights runs/train/exp152/weights/last.pt --cfg models/prunModels/yolov5s-pruning.yaml --data data/VisDrone.yaml --path yolov5s-pruned.yaml --max_iter 1200 --remain_ratio 0.6 --delta 0.05 >> yolosSearch.out &
# python -m torch.distributed.launch --nproc_per_node 2 --master_port 2000 train.py --data DVSPerson.yaml --weights /home/zy/DVSTrack/yolov5/weights/yolon608-pruned.pt --cfg models/prunModels/yolov5n-DVSPerson-pruned.yaml --epochs 12 --batch-size 4 --img 608 --nosave --device 0,1 --sync-bn

# Prune Slim
# Step1:
# python train.py --weights /home/zy/DVSTrack/yolov5/weights/yolon608.pt --cfg models/prunModels/yolov5n-pruning.yaml --data data/DVSPerson.yaml --epochs 10 --batch-size 4 --img 608 --nosave --device 2,3 --sync-bn --sparse --name sparseDVSTrack

# Step2:
# python pruneSlim.py --weights runs/train/sparseDVSTrack/weights/best.pt --cfg models/prunModels/yolov5n-pruning.yaml --data data/DVSPerson.yaml --path yolov5n-DVSPerson-pruned.yaml --global_percent 0.6 --device 3

# Step3:
# python -m torch.distributed.launch --nproc_per_node 2 --master_port 2006 train.py --weights runs/train/sparseDVSTrack/weights/best-Slimpruned.pt --cfg models/prunModels/yolov5n-DVSPerson-prunedSlim.yaml --data data/DVSPerson.yaml --epochs 16 --batch-size 8 --img 608 --device 2,3 --sync-bn --name slimDVSTrack
