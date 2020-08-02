# Classification of KMNIST (Kuzushiji-MNIST) with SoftTriple Loss

## Requirements
This code is confirmed to run with Python 3.7 and PyTorch 1.6.0.

## Usage
1. Download datasets
```
bash download_dataset.sh
```
or, make directory "dataset" and download original NumPy format files from https://github.com/rois-codh/kmnist.

2. Train

```
mkdir path/to/checkpoint_dir
python train.py --loss_type softtriple --checkpoint_dir path/to/checkpoint_dir
```
You can choose loss_type from `softtriple`, `softmaxnorm` and `crossentropy`.
For other options, please refer helps: `python train.py -h`.

3. Test

When your training is done, model parameter files `path/to/checkpoint_dir/model_final.pth` and `path/to/checkpoint_dir/criterion_final.pth` will be generated.
```
python test.py --loss_type softtriple --params_path_m path/to/checkpoint_dir/model_final.pth --params_path_c path/to/checkpoint_dir/criterion_final.pth
```
If you specify/change some hyperparameters in training, you must use the same hyperparameters also in testing.

## Dataset
KMNIST (http://codh.rois.ac.jp/kmnist/)
- 『KMNISTデータセット』（CODH作成） 『日本古典籍くずし字データセット』（国文研ほか所蔵） doi:10.20676/00000341
- Tarin Clanuwat, Mikel Bober-Irizar, Asanobu Kitamoto, Alex Lamb, Kazuaki Yamamoto, David Ha, "Deep Learning for Classical Japanese Literature", arXiv:1812.01718.

## References
- Qi Qian, Lei Shang, Baigui Sun, Juhua Hu, Hao Li, Rong Jin, "SoftTriple Loss: Deep Metric Learning Without Triplet Sampling", ICCV 2019.
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Deep Residual Learning for Image Recognition", CVPR 2016.

# Acknowledgements
This implementation is based on https://github.com/idstcv/SoftTriple.