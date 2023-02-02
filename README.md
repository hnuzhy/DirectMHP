# DirectMHP

Code for my paper [A Simple Baseline for Direct 2D Multi-Person Head Pose Estimation with Full-range Angles]() (submitted to TMM)

## Abstract
> Existing head pose estimation (HPE) mainly focuses on single person with pre-detected frontal heads, which limits their applications in real complex scenarios with multi-persons. We argue that these single HPE methods are fragile and inefficient for Multi-Person Head Pose Estimation (MPHPE) since they rely on the separately trained face detector that cannot generalize well to full viewpoints, especially for heads with invisible face areas. In this paper, we focus on the full-range MPHPE problem, and propose a direct end-to-end simple baseline named DirectMHP. Due to the lack of datasets applicable to the full-range MPHPE, we firstly construct two benchmarks by extracting ground-truth labels for head detection and head orientation from public datasets AGORA and CMU Panoptic. They are rather challenging for having many truncated, occluded, tiny and unevenly illuminated human heads. Then, we design a novel end-to-end trainable one-stage network architecture by joint regressing locations and orientations of multi-head to address the MPHPE problem. Specifically, we regard pose as an auxiliary attribute of the head, and append it after the traditional object prediction. Arbitrary pose representation such as Euler angles is acceptable by this flexible design. Then, we jointly optimize these two tasks by sharing features and utilizing appropriate multiple losses. In this way, our method can implicitly benefit from more surroundings to improve HPE accuracy while maintaining head detection performance. We present comprehensive comparisons with state-of-the-art single HPE methods on public benchmarks, as well as superior baseline results on our constructed MPHPE datasets.


## Illustrations

* **Fig. 1.** Example images of two constructed challenging datasets: AGORA-HPE (*top row*) and CMU-HPE (*middle row*), and two widely used single HPE datasets: 300W-LP & AFLW2000 (*left-down*) and BIWI (*right-down*).
![example1](./materials/datasetexamples.png)

* **Fig. 2.** The pipeline illustration of existing HPE methods, which are all based on two separate stages, and our proposed one-stage MPHPE.
![example2](./materials/illustration.png)


## Installation

* **Environment:** Anaconda, Python3.8, PyTorch1.10.0(CUDA11.2), wandb
  ```bash
  $ git clone https://github.com/hnuzhy/DirectMHP.git
  $ pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

  # Codes are only evaluated on GTX3090 + CUDA11.2 + PyTorch1.10.0.
  $ pip3 install torch==1.10.0+cu111 torchvision==0.11.1+cu111 torchaudio==0.10.0+cu111 \
    -f https://download.pytorch.org/whl/cu111/torch_stable.html
  ```

* **Sim3DR:** Install the renderer, which is used to visualize predictions.
  ```bash
  $ cd Sim3DR
  $ sh build_sim3dr.sh
  ```

## Dataset Preparing


## Training and Testing

* **Yaml:** Please refer these `./data/\*.yaml` files to config your own .yaml file

* **Pretrained weights:** For YOLOv5 weights, please download the version 5.0 that we have used. And put them under the `./weights/` folder
  ```
  yolov5s6.pt [https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s6.pt]
  yolov5m6.pt [https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5m6.pt]
  yolov5l6.pt [https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5l6.pt]
  ```

* **Training:**

* **Testing:**

* **Inference:**


## References

* [YOLOv5 🚀 in PyTorch > ONNX > CoreML > TFLite](https://github.com/ultralytics/yolov5)
* [img2pose: Face Alignment and Detection via 6DoF, Face Pose Estimation - CVPR 2021](https://github.com/vitoralbiero/img2pose)

### Citation

If you use our datasetd (AGORA-HPE and CMU-HPE) or models in your research, please cite with:
```
@inproceedings{joo2015panoptic,
  title={Panoptic studio: A massively multiview system for social motion capture},
  author={Joo, Hanbyul and Liu, Hao and Tan, Lei and Gui, Lin and Nabbe, Bart and Matthews, Iain and Kanade, Takeo and Nobuhara, Shohei and Sheikh, Yaser},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={3334--3342},
  year={2015}
}

@inproceedings{patel2021agora,
  title={AGORA: Avatars in geography optimized for regression analysis},
  author={Patel, Priyanka and Huang, Chun-Hao P and Tesch, Joachim and Hoffmann, David T and Tripathi, Shashank and Black, Michael J},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={13468--13478},
  year={2021}
}

```
