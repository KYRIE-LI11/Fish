# Fish detection

## 1、主要代码文件

### （1）PC端需要准备的文件
yolov5官方库,预训练文件yolov5s.pt. 

```python
# 模型输入格式 [batch, channel, height, width] 
# 训练使用数据 (batch, 1, 128, 128)
# 模型输出格式 [batch, 2, height, width], 2 是二分类。
```

代码运行命令：

```
python test_stage_DCSPM.py --data_path='./data' --bs=1 --shape=128
```

注意：

> 1.  运行时 data_path='./data' 中的 ' ./data ' 需要更换为自己的数据集路径；
> 2.  指标计算函数 evaluate_mc_statistic 按照 batch 计算，**bs=1** 时计算结果准确，不要改这个；
> 3.  预训练好的 DCSPM_v0.pth 固定输入大小为 **(128*128)**，因此运行时 shape 也需要是 128 ；
> 4.  get_args() 可以增加外部命令传入参数，可以自行添加其它控制；
> 5.  Alone_Dataset_mc 类 用于加载数据集，可以自行编写替换；
> 6.  evaluate_mc_statistic 函数中执行前向推理预测分割结果，返回计算指标 dice, iou, precision, recall, hd95, acc等。

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
