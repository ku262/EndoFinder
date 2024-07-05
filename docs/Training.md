# Training EndoFinder models

We run `EndoFinder/polyp_pretrain.py` to train EndoFinder models.

The train command is as follows:
```
MASTER_ADDR="<first worker hostname>" python polyp_pretrain.py \
  --data_path=/path/to/images/train \
  --mask_path=/path/to/labels/train \
  --mask_ratio=0.5 --batch_size=32 --entropy_weight=5\
  --mse_weight=0.5 --augmentations=ADVANCED --mixup=true \
  --output_dir=/path/to/train/output \
  --log_dir=/path/to/train/output
```

