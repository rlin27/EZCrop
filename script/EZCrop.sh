# CIFAR-10
python evaluate_cifar.py \
--data_dir [Your path to the dataset] \
--arch [Choose the model architecture] \
--job_dir [Your path to save trained models] \
--pretrain_dir [Your path to Pretrained model] \
--ratio_conv_prefix [Ratio conv file folder] \
--compress_rate [Compress rate of each conv] \
--gpu 0,1 \

# ImageNet
python evaluate.py \
--data_dir [Your path to the dataset] \
--arch [Choose the model architecture] \
--job_dir [Your path to save trained models] \
--pretrain_dir [Your path to Pretrained model] \
--ratio_conv_prefix [Ratio conv file folder] \
--compress_rate [Compress rate of each conv] \
--gpu 0,1 \
