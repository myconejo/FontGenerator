# FontGenerator
Alphabet font generator using GAN, inspired from zi2zi, pix2pix and other similar GAN projects. 

## Project Objective
The objective of the project is to make a new font by using the model. Making a new font style is a very expensive job. Also, there needs to be a lot of creativity to implement new fonts. In the case of the alphabet, there are 52 letters and more than 30 signs need to be defined by the font file. By generating a new image of letters and signs which adopt the new style of font or human font style, we can reduce the costs to make new fonts. According to GAN’s style interpolation, we may create new kinds of fonts which separately adapt different fonts’ styles. 

### Set conda environment
```
$ cd ./FontGenerator
$ conda env create --file conda_requirements.yaml
$ conda activate FontProject

```
### Make Directory
```
$ mkdir ./results
$ mkdir ./results/ckpt
$ mkdir ./results/ckpt/pre_train
$ mkdir ./results/ckpt/finetune
$ mkdir ./results/fake-image/pre_train
$ mkdir ./results/fake-image/pre_train/valid
$ mkdir ./results/fake-image/finetune
$ mkdir ./results/fake-image/finetune/valid
```
### 1. Install Font files and convert to images
```
https://drive.google.com/file/d/1iRYDXJbH_x4Kabr52LudkvE8JnspLaN6/view?usp=sharing
Extract font files in Util/Font
$ python Util/FontImageGerator.py
```
### 2. Install Pretrain model file
```
https://drive.google.com/file/d/1nu4fKGHy5HVb_s_k425R8LquvBHgXxVJ/view?usp=share_link
Extract font files in ./results/ckpt/pre_train
```
### 3. Pretrain
If trying to reproduce pretrain model
If don't trying to train and just use trained model parameters pass.
```
$ python ./train/train.py
```

### 4. Pretrain Validation(Inference)
```
$ python ./train/valid.py
```
validation results are in results/pre_train/valid-*.png
Target results are results/pre_train/valid-treu-*.png

### 5. Finetuning
```
$ python ./train/finetune.py
```

### 6. Result Images
```
Check images in ./results/fake-image/finetune
```

## References
* [**zi2zi**](https://github.com/kaonashi-tyc/zi2zi/)
* [**zi2zi-blog**](https://kaonashi-tyc.github.io/2017/04/06/zi2zi.html)
* [**GAN-handwriting-styler**](https://github.com/jeina7/GAN-handwriting-styler)
* [**GAN-handwriting-styler-blog**](https://jeinalog.tistory.com/15)
* [**GAN-goodfellow**](https://arxiv.org/pdf/1406.2661.pdf)
