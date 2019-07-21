# MineInfoGan
mine + infogan

```bash
python infogan.py --help
usage: infogan.py [-h] [--n_epochs N_EPOCHS] [--batch_size BATCH_SIZE]
                  [--lr LR] [--b1 B1] [--b2 B2] [--n_cpu N_CPU]
                  [--latent_dim LATENT_DIM] [--code_dim CODE_DIM]
                  [--n_classes N_CLASSES] [--img_size IMG_SIZE]
                  [--channels CHANNELS] [--sample_interval SAMPLE_INTERVAL]

optional arguments:
  -h, --help            show this help message and exit
  --n_epochs N_EPOCHS   number of epochs of training
  --batch_size BATCH_SIZE
                        size of the batches
  --lr LR               adam: learning rate
  --b1 B1               adam: decay of first order momentum of gradient
  --b2 B2               adam: decay of first order momentum of gradient
  --n_cpu N_CPU         number of cpu threads to use during batch generation
  --latent_dim LATENT_DIM
                        dimensionality of the latent space
  --code_dim CODE_DIM   latent code
  --n_classes N_CLASSES
                        number of classes for dataset
  --img_size IMG_SIZE   size of each image dimension
  --channels CHANNELS   number of image channels
  --sample_interval SAMPLE_INTERVAL
                        interval between image sampling
```

# Run experiment
```bash
python infogan.py --n_epochs=200 --batch_size 64 --lr 0.0002 --n_cpu 8 --latent_dim 62 --code_dim 2 
```