# MyAutoPano
This project was completed by Yi-Chung Chen, Ji Liu, and Shreyas Acharya.

## Phase 1
To stitch images, you can run the following command.
```
python3 ./Wrapper.py --BasePath {Path to your dataset}
```

You can use `--Dataset` to set the training set or testing set, and also `--Set` to select which subset of images will be stitched.
```
python3 ./Wrapper.py --BasePath {Path to your dataset} --Dataset Train --Set Set1
```

## Phase 2
### Train
Use the following code to train the homography network, the default is unsupervised homography net and you can choose the model type by `--ModelType`.
```
python3 ./Train.py --BasePath {directory to your dataset} --ModelType Unsup
```

To train the supervised homography net and set the Number of training epochs by `--NumEpochs`, you can use this code.
```
python3 ./Train.py --ModelType Sup --NumEpochs 10
```

### Test
To test the model, you can use this command. This will load the checkpoint and compute the average EPE (average L2 error between predicted and ground truth homographies) at last. Use `--ModelPath` to set the directory of the checkpoint and `--BasePath` to set the path to your path to the test set.

```
python3 ./Test.py --ModelPath {directory to your checkpoint} --BasePath {directory to your testing dataset} --ModelType Unsup
```
## Reference
1. DeTone, Daniel, Tomasz Malisiewicz, and Andrew Rabinovich. "Deep image homography estimation." arXiv preprint arXiv:1606.03798 (2016).
2. Nguyen, Ty, et al. "Unsupervised deep homography: A fast and robust homography estimation model." IEEE Robotics and Automation Letters 3.3 (2018): 2346-2353.
