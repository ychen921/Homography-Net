# Homography Net
This project was completed by Yi-Chung Chen, Ji Liu, and Shreyas Acharya. We developed two deep learning approaches for estimating tomography: supervised and unsupervised. We have achieved the average corner error (EPE) of 5.159 pixels for our supervised model, and 10.213 pixels for our unsupervised model. For more details, please look at the `report.pdf` and [project website](https://cmsc733.github.io/2022/proj/p1/) 

## Data

## Architecture

## Usage
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

### Visualize
```
python3 ./visualization.py --SupCheckPointPath {directory to supervised model checkpoint} --UnsupCheckPointPath {directory to unsupervised model checkpoint} ----TestPath {directory to your testing dataset}
```

## Visualization & Performance

![](./figure/vis.png)


| Model       | Training L2     | Validation L2    |  Test L2        |
| :---        |    :----:       |     :----:       |    :----:       |
| Supervised  | 5.15 $\pm$ 2.45 | 5.17 $\pm$ 2.44  | 5.23 $\pm$ 2.62 |
| Unsupervised| 10.20 $\pm$ 4.74| 9.99 $\pm$ 4.92  | 10.21 $\pm$ 4.85|

## Reference
1. DeTone, Daniel, Tomasz Malisiewicz, and Andrew Rabinovich. "Deep image homography estimation." arXiv preprint arXiv:1606.03798 (2016).
2. Nguyen, Ty, et al. "Unsupervised deep homography: A fast and robust homography estimation model." IEEE Robotics and Automation Letters 3.3 (2018): 2346-2353.
