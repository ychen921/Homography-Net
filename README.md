# MyAutoPano

## Usage

Use the following code to train the homography network, the defualt is unsupervised homography net.
```
python3 ./Train.py
```

To train the supervised homography net and set the Number of training epoch by `--NumEpochs`, you can use this code
```
python3 ./Train.py --ModelType Sup --NumEpochs 10
```