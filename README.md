# DirectMHP for deployment
This package simplifies [DirectMHP](https://github.com/hnuzhy/DirectMHP) for deployment in my project.

## Table of contents
<!--ts-->
- [Installation](#installation)
- [Inference](#inference)
- [References](#references)
- [Licenses](#licenses)
<!--te-->

## Installation
TODO

## Usage
Using the model for inference
```python
from DirectMHP_Inference_seoy import DirectMHP

model = DirectMHP(weights='path/to/weights', device=device)
```

To save model in onnx format
```python
from DirectMHP_Inference_seoy import save_with_weights

save_with_weights(weights='path/to/weights', device='device')
```
or run the save_model script in the package
  
## References

* [YOLOv5 🚀 in PyTorch > ONNX > CoreML > TFLite](https://github.com/ultralytics/yolov5)
* [BMVC 2020 - WHENet: Real-time Fine-Grained Estimation for Wide Range Head Pose](https://github.com/Ascend-Research/HeadPoseEstimation-WHENet)
* [CVPR 2021 - img2pose: Face Alignment and Detection via 6DoF, Face Pose Estimation](https://github.com/vitoralbiero/img2pose)
* [ICIP 2022 - 6D Rotation Representation for Unconstrained Head Pose Estimation](https://github.com/thohemp/6DRepNet)
* We also thank public datasets [AGORA](https://agora.is.tue.mpg.de/) and [CMU-Panoptic](http://domedb.perception.cs.cmu.edu/) for their excellent works.


## Licenses

Our work is based on public code and datasets. If you plan to add our work to your business project, please obtain the following enterprise licenses.
* **DirectMHP:** GNU General Public License v3.0 (GPL-3.0 License): See [LICENSE](./LICENSE.txt) file for details. 
* **YOLOv5:** To request an Enterprise License please complete the form at [Ultralytics Licensing](https://ultralytics.com/license)
* **AGORA-HPE:** Data & Software Copyright License for non-commercial scientific research purposes [AGORA License](https://agora.is.tue.mpg.de/license.html)
* **CMU-HPE:** CMU Panoptic Studio dataset is shared only for research purposes, and this cannot be used for any commercial purposes. The dataset or its modified version cannot be redistributed without permission from dataset organizers [CMU Panoptic Homepage](http://domedb.perception.cs.cmu.edu/)