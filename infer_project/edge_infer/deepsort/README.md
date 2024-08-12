# 使用说明
用于视频目标检测跟踪推理，检测模型参考yolov5训练，跟踪模型可用自带预训练模型。

# 运行main.py：
帮助信息
```bash
python main.py -h
```
  --yolov5_weights pt模型地址.

  --objects OBJECTS  输出跟踪目标类别，比如： ”cat，dog“.

  --input_video_path 视频地址

  --save_output_path 输出结果视频地址

执行实例
```angular2html
python main.py --yolov5_weights ./weights/yolov5s.pt --objects car,person --input_video_path ./videov/test_person.mp4
```

遵循 GNU General Public License v3.0 协议，标明目标检测部分来源：https://github.com/ultralytics/yolov5/
