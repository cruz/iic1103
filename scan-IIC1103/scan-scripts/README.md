## Scripts to recognize digits


### Convert PDFs to PNGs
The script `convert.sh` takes all PDFs in `INPUT_TEST` directory,
and outputs a PNG using the `convert` tool from the `Imagemagick` package

```
./convert.sh
```


### Cutting digits
`getDigits.py` cuts a fixed-size upper right corner of the PNG and
extracts all squares.
```
python getDigits.py
```
TODO: Extract corner based on the position of the black square, instead of a fixed size.
Then, extracting the squares should be easier, as they will be in a fixed relative position
to the black square.

TODO: Sort the squares according to their position in the image

### Training classifier
`train-mnist.py` trains a LinearSVM using the MINST training data. If the dataset
is not locally available, it is downloaded (but this happens only the first time).
The resulting classifier model is stored as `digits_cls.pkl`

```
python train-mnist.py train
```

### Reading digits
`train-mnist.py` uses the trained model to identify digits on all files ending as `*-cut.png`.

```
python train-mnist.py
```

TODO: This should be integrated in a single script that cuts the image, recognizes digits, and outputs
the numbers.

