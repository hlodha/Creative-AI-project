export CUDA_VISIBLE_DEVICES='0,1' # edit this if you want to limit yourself to GPU
python3 amb_disc.py --dataset wikiart --dataroot ~/Pytorch_experiments/datasets/wikiart_256 --outf ambiguitiy_dscriminator_exp2 --imageSize 256 --cuda --ngpu 2 --batchSize 32 --niter 100 --lr 0.00004 --n_class 27 --ndf 128 --workers 8 #--netD ./ambiguitiy_dscriminator_from_scratch/netD_epoch_0.pth
