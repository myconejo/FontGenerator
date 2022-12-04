# FontGenerator
Alphabet font generator using GAN, inspired from zi2zi, pix2pix and other similar GAN projects. 

## Project Objective
The objective of the project is to make a new font by using the model. Making a new font style is a very expensive job. Also, there needs to be a lot of creativity to implement new fonts. In the case of the alphabet, there are 52 letters and more than 30 signs need to be defined by the font file. By generating a new image of letters and signs which adopt the new style of font or human font style, we can reduce the costs to make new fonts. According to GAN’s style interpolation, we may create new kinds of fonts which separately adapt different fonts’ styles. 

### Set conda environment
```
$ conda env create --file conda_requirements.yaml
```

### 1. Install Font files
```
https://drive.google.com/file/d/1iRYDXJbH_x4Kabr52LudkvE8JnspLaN6/view?usp=sharing
extract font files in Util/Font
```
### 2. Install Pretrain model file
```
extract font files in ./results/ckpt/pre_train
```
### 3. Pretrain
If trying to reproduce pretrain model
```
$ python ./train/train.py
```

### 4. Finetuning
```
$ python ./train/finetune.py
```

### 5. Result Images
```
Check images in ./results/fake-image/finetune
```

## References
* [**zi2zi**](https://github.com/kaonashi-tyc/zi2zi/)
* [**zi2zi-blog**](https://kaonashi-tyc.github.io/2017/04/06/zi2zi.html)
* [**GAN-handwriting-styler**](https://github.com/jeina7/GAN-handwriting-styler)
* [**GAN-handwriting-styler-blog**](https://jeinalog.tistory.com/15)
* [**GAN-goodfellow**](https://arxiv.org/pdf/1406.2661.pdf)
