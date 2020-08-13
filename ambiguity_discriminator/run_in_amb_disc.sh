export CUDA_VISIBLE_DEVICES='1,2' # edit this if you want to limit yourself to GPU
python3 in_amb_disc.py --dataset wikiart --dataroot ~/Pytorch_experiments/datasets/wikiart_256 --outf ambiguitiy_dscriminator_exp3_imagenet --imageSize 224 --cuda --ngpu 2 --batchSize 16 --niter 100 --lr 0.00002 --n_class 27 --ndf 128 --workers 8 --netD ./ambiguitiy_dscriminator_exp3_imagenet/netD_epoch_8.pth
