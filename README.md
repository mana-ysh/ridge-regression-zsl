# ridge-regression-zsl
This code is implementations of Regression-based Zero Shot Learning.
Traditional ridge-regression, which is the model with the projection from example to label space, and inversed ridge-regression, which projects from label to example space, are available.

If you run this code, you can reproduce the result of the synthetic data experiment in [Shigeto+. 2015]. Additionally, Online learning setting with Tensorflow, not closed-form, is also available.

## Run
```
⟩⟩⟩ python train.py -h
usage: train.py [-h] [--mode MODE] [--method METHOD] [--lr LR] [--epoch EPOCH]
                [--log LOG] [--l L] [--l1 L1] [--l1_ratio L1_RATIO] [--skewx]
                [--stdx] [--info INFO] [--sk] [--abs] [--r2_flg]

optional arguments:
  -h, --help           show this help message and exit
  --mode MODE          training mode ["online", "closed", "cd"]
  --method METHOD      method ["ridgex", "ridgey"]
  --lr LR              learning rate
  --epoch EPOCH        number of epochs
  --log LOG            output log dir
  --l L                regularizer
  --l1 L1              L1 regularizer
  --l1_ratio L1_RATIO
  --skewx              force to skew x in synthetic data
  --stdx               standardize x in synthetic data
  --info INFO          informative dimention of scikit-learn
  --sk                 use scikit-learn dataset
  --abs                take abs in reg2
  --r2_flg             enable L2 reg for r2
```

## Dependencies
- numpy
- scikit-learn
- tensorflow

## References
- Ridge Regression, Hubness, and Zero-Shot Learning. Yutaro Shigeto, Ikumi Suzuki, Kazuo Hara, Masashi Shimbo, and Yuji Matsumoto. ECML/PKDD 2015. 
