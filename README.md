# Enhanced 3D Object Detection via Global-Local Feature Fusion Network in Point Clouds and Images

## Main Results

This one is what we get on the sunrgbd dataset.

|        Method        | Point Backbone | Input  |   mAP@0.25    |    mAP@0.5    |
| :------------------: | :------------: | :----: | :-----------: | :-----------: |
| GLFF3D(FCAF3D based) |    HDResNet    | PC+RGB | 69.14 (69.08) | 50.77 (50.64) |

## 1. Prerequisites

The code is tested with Python3.7, PyTorch == 1.8, CUDA == 11.1, mmdet3d == 0.18.1, mmcv_full == 1.3.18 and mmdet == 2.14. We recommend you to use anaconda to make sure that all dependencies are in place. Note that different versions of the library may cause changes in results.

**Step 1.** Create a conda environment and activate it.

```bash
conda create --name glff python=3.7
conda activate glff
```

**Step 2.** Install [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) following the instruction [here](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/getting_started.md).

**Step 3.** Prepare SUN RGB-D Data following the procedure [here](https://github.com/open-mmlab/mmdetection3d/tree/master/data/sunrgbd).

## 2. Joint Training.

To start training, run dist_train.sh with `GLFF3D` configs:

```shell
bash tools/dist_train.sh configs/GLFF3D/base.py 2
```

## 3. Testing

Test pre-trained model using dist_test with `GLFF3D` configs:

```bash
python tools/test.py configs/GLFF3D/base.py \
    work_dirs/GLFF3D_sunrgbd-3d-10class/latest.pth --eval mAP
```

## 4. Visualization

Visualizations can be created with `test.py` script. 

```bash
python tools/test.py configs/GLFF3D/base.py \
    work_dirs/GLFF3D_sunrgbd-3d-10class/latest.pth --eval mAP --show \
    --show-dir work_dirs/GLFF3D_sunrgbd-3d-10class
```

## 5. Citation

This code is part of the manuscript currently under submission to The Visual Computer.Please kindly cite this paper in your publications if it helps your research:

```
@article{,
  title={Enhanced 3D Object Detection via Global-Local Feature Fusion Network in Point Clouds and Images},
  author={Du, Haishun and Zhang, Zhengyang and Zhang, Wenzhe and Cao, Linbing},
  journal={},
  pages={},
  year={},
  publisher={}
}
```

## 6. References

[1] Chen, K., Wang, J., Pang, J., Cao, Y., Xiong, Y., Li, X., Sun, S., Feng, W., Liu, Z., Xu, J., Zhang, Z., Cheng, D., Zhu, C., Cheng, T., Zhao, Q., Li, B., Lu, X., Zhu, R., Wu, Y., Liu, K., Dai, J., Wang, J., Shi, J., Ouyang, W., Loy, C.C., & Lin, D. (2019). MMDetection: Open MMLab Detection Toolbox and Benchmark. *ArXiv, abs/1906.07155*.

[2] S. Song, S. P. Lichtenberg and J. Xiao, "SUN RGB-D: A RGB-D scene understanding benchmark suite," *2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Boston, MA, USA, 2015, pp. 567-576.
