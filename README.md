# MyAutoPano
This project was completed by Yi-Chung Chen, Ji Liu, and Shreyas Acharya.

## Phase 1
To stitch images, you can run following commend.
```
python3 ./Wrapper.py --BasePath {Path to your dataset}
```

You can use `--Dataset` to set training set or testing set, also `--Set` to select which subset of images will be stitched.
```
python3 ./Wrapper.py --BasePath {Path to your dataset} --Dataset Train --Set Set1
```

## Phase 2
### Train
Use the following code to train the homography network, the defualt is unsupervised homography net and you can choose the model type by `--ModelType`.
```
python3 ./Train.py --BasePath {directory to your dataset} --ModelType Unsup
```

To train the supervised homography net and set the Number of training epoch by `--NumEpochs`, you can use this code.
```
python3 ./Train.py --ModelType Sup --NumEpochs 10
```

### Test
To test the model, you can use this commend. This will load the checkpoint and compute the average EPE (average L2 error between predicted and ground truth homographies) at last. Use `--ModelPath` to set the directory of the checkpoint and `--BasePath` to set the path to your path to the test set.

```
python3 ./Test.py --ModelPath {directory to your checkpoint} --BasePath {directory to your testing dataset} --ModelType Unsup
```


`Wrapper.py`: we are not able to complete image stitching due to the time limitation