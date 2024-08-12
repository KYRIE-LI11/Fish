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
