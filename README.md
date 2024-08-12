# Fish detection

## 1、主要代码文件

### （1）PC端需要准备的文件
yolov5官方库,预训练文件yolov5s.pt. 

代码运行命令：
```
conda activate yolov5
```

```
python train.py --data fish.yaml --epochs 300 --weights 'yolov5s.pt' --cfg yolov5s.yaml  --batch-size 32
```

在鱼类数据集上微调后，可利用yolov5官网库中的脚本将pt文件转换为onnx文件:
```
python export.py --weights 训练好的pt模型路径 --include onnx --opset 11
```
注意：

> 1. 在训练过程中,fish.yaml是经过重写的,参考"data/fish.yaml"
> 2. 在模型转换过程中,考虑青云设备的算子支持,必须设置--opset 11
> 3. 训练及转换后的权重文件存放路径："yolov5/runs/train/exp2"

### （2）post_process.py

相较于 test_stage_DCSPM.py，加入分割结果的后处理和可视化代码。

代码运行命令：

```
python post_process.py --data_path='./data' --bs=1 --shape=128 --post=True
```

注意：

> 1. 代码中的 **mask_pred** 是分割结果，可以取出做后续处理；
> 2. post 参数是对模型预测分割结果 **mask_pred** 的后处理开关，默认False关闭，可以在外部命令或者get_args()中修改；
> 3. visualization() 和 visualization_with_mask_true() 函数均是对 mask_pred 的处理。visualization() 将预测结果可视化在原图上，并resize成(256×256)；visualization_with_mask_true() 将标注和预测结果可视化在原图上，并resize成(256×256)；
> 4. 其它参数和函数参考 test_stage_DCSPM.py 。

###(2)NPU测试所需文件
infer_project文件夹：

```
infer_project
├── benchmark.aarch64
├── common
│   ├── eval.sh
│   ├── onnx2om.sh
│   ├── pth2om.sh
│   ├── quantize
│   │   ├── __init__.py
│   │   ├── amct.sh
│   │   ├── calibration_scale.py
│   │   ├── config.cfg
│   │   ├── generate_data.py
│   │   └── img_info_amct.txt
│   ├── util
│   │   ├── __init__.py
│   │   ├── acl_net.py
│   │   ├── aipp_yolov5s.cfg
│   │   ├── atc.sh
│   │   ├── fusion.cfg
│   │   └── modify_model.py
│   ├── world_cup.jpg
│   ├── yolov5_camera.ipynb
│   ├── yolov5_image.ipynb
│   └── yolov5_video.ipynb
├── config.yaml
├── data.yaml
├── edge_infer
│   ├── DeepSortDetector.py
│   ├── acl_image.py
│   ├── acl_model.py
│   ├── acl_net_dynamic.py
│   ├── acl_resource.py
│   ├── coco_names.txt
│   ├── constants.py
│   ├── deepsort
│   │   ├── AIDetector_pytorch.py
│   │   ├── LICENSE
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── deep_sort
│   │   │   ├── configs
│   │   │   │   └── deep_sort.yaml
│   │   │   ├── deep_sort
│   │   │   │   ├── README.md
│   │   │   │   ├── __init__.py
│   │   │   │   ├── deep
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── evaluate.py
│   │   │   │   │   ├── feature_extractor.py
│   │   │   │   │   ├── model.py
│   │   │   │   │   ├── original_model.py
│   │   │   │   │   ├── test.py
│   │   │   │   │   ├── train.jpg
│   │   │   │   │   └── train.py
│   │   │   │   ├── deep_sort.py
│   │   │   │   └── sort
│   │   │   │       ├── __init__.py
│   │   │   │       ├── detection.py
│   │   │   │       ├── iou_matching.py
│   │   │   │       ├── kalman_filter.py
│   │   │   │       ├── linear_assignment.py
│   │   │   │       ├── nn_matching.py
│   │   │   │       ├── preprocessing.py
│   │   │   │       ├── track.py
│   │   │   │       └── tracker.py
│   │   │   └── utils
│   │   │       ├── __init__.py
│   │   │       ├── asserts.py
│   │   │       ├── draw.py
│   │   │       ├── evaluation.py
│   │   │       ├── io.py
│   │   │       ├── json_logger.py
│   │   │       ├── log.py
│   │   │       ├── parser.py
│   │   │       └── tools.py
│   │   ├── main.py
│   │   ├── requirements.txt
│   │   ├── tracker.py
│   │   └── utils
│   │       ├── BaseDetector.py
│   │       └── __init__.py
│   ├── det_utils.py
│   ├── fusion_result.json
│   ├── lablenames.txt
│   ├── mAP
│   │   ├── gt_generate.py
│   │   ├── main.py
│   │   └── predicted
│   │       ├── cat12.txt
│   │       ├── cat15.txt
│   │       ├── cat5.txt
│   │       ├── cat6.txt
│   │       ├── dog11.txt
│   │       ├── dog14.txt
│   │       └── dog8.txt
│   ├── utils.py
│   ├── v5_object_detect.py
│   ├── video.py
│   └── yolov5_infer.py
├── fusion_result.json
├── models
│   ├── __init__.py
│   ├── common.py
│   ├── experimental.py
│   ├── hub
│   │   ├── anchors.yaml
│   │   ├── yolov3-spp.yaml
│   │   ├── yolov3-tiny.yaml
│   │   ├── yolov3.yaml
│   │   ├── yolov5-bifpn.yaml
│   │   ├── yolov5-fpn.yaml
│   │   ├── yolov5-p2.yaml
│   │   ├── yolov5-p34.yaml
│   │   ├── yolov5-p6.yaml
│   │   ├── yolov5-p7.yaml
│   │   ├── yolov5-panet.yaml
│   │   ├── yolov5l6.yaml
│   │   ├── yolov5m6.yaml
│   │   ├── yolov5n6.yaml
│   │   ├── yolov5s-LeakyReLU.yaml
│   │   ├── yolov5s-ghost.yaml
│   │   ├── yolov5s-transformer.yaml
│   │   ├── yolov5s6.yaml
│   │   └── yolov5x6.yaml
│   ├── segment
│   │   ├── yolov5l-seg.yaml
│   │   ├── yolov5m-seg.yaml
│   │   ├── yolov5n-seg.yaml
│   │   ├── yolov5s-seg.yaml
│   │   └── yolov5x-seg.yaml
│   ├── tf.py
│   ├── yolo.py
│   ├── yolov5l.yaml
│   ├── yolov5m.yaml
│   ├── yolov5n.yaml
│   ├── yolov5s.yaml
│   └── yolov5x.yaml
├── om_infer.py
├── onnx2om.py
├── output
│   └── yolov5s_bs1.om
├── run.py
├── test
│   ├── images
│   │   ├── 00000005.jpg
│   │   └── 00000020.jpg
│   ├── labels
│   │   ├── 00000005.txt
│   │   └── 00000020.txt
│   └── test.json
├── utils
│   ├── __init__.py
│   ├── activations.py
│   ├── augmentations.py
│   ├── autoanchor.py
│   ├── autobatch.py
│   ├── aws
│   │   ├── __init__.py
│   │   ├── mime.sh
│   │   ├── resume.py
│   │   └── userdata.sh
│   ├── callbacks.py
│   ├── dataloaders.py
│   ├── docker
│   │   ├── Dockerfile
│   │   ├── Dockerfile-arm64
│   │   └── Dockerfile-cpu
│   ├── downloads.py
│   ├── flask_rest_api
│   │   ├── README.md
│   │   ├── example_request.py
│   │   └── restapi.py
│   ├── general.py
│   ├── google_app_engine
│   │   ├── Dockerfile
│   │   ├── additional_requirements.txt
│   │   └── app.yaml
│   ├── loggers
│   │   ├── __init__.py
│   │   ├── clearml
│   │   │   ├── README.md
│   │   │   ├── __init__.py
│   │   │   ├── clearml_utils.py
│   │   │   └── hpo.py
│   │   ├── comet
│   │   │   ├── README.md
│   │   │   ├── __init__.py
│   │   │   ├── comet_utils.py
│   │   │   ├── hpo.py
│   │   │   └── optimizer_config.json
│   │   └── wandb
│   │       ├── README.md
│   │       ├── __init__.py
│   │       ├── log_dataset.py
│   │       ├── sweep.py
│   │       ├── sweep.yaml
│   │       └── wandb_utils.py
│   ├── loss.py
│   ├── metrics.py
│   ├── plots.py
│   ├── segment
│   │   ├── __init__.py
│   │   ├── augmentations.py
│   │   ├── dataloaders.py
│   │   ├── general.py
│   │   ├── loss.py
│   │   ├── metrics.py
│   │   └── plots.py
│   ├── torch_utils.py
│   └── triton.py
├── yolov5s.onnx
└── yolov5s.pt
```



首先将保存的训练好的权重文件（pt格式）转为onnx模型（.onnx文件），然后在infer_project目录下执行以下命令：

```
python3 onnx2om.py
```

将onnx模型转为om模型，可通过config.yaml文件修改路径和相关参数。
om模型将保存在./output目录下

进行推理：首先将标签填入./edge_infer/lablenames.txt文件中，然后将./edge_infer/yolov5_infer.py中模型文件路径及标签文件路径填入，并选择推理模式，如使用图片或视频，请填写其路径。

修改完毕后，在该目录下执行命令：

```
python3 yolov5_infer.py
```

进行推理
