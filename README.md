## 安装环境
mmcv [https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html]

下载符合条件的whl包
mmcv-2.2.0-cp311-cp311-manylinux1_x86_64.whl

```bash
uv add mmcv-2.2.0-cp311-cp311-manylinux1_x86_64.whl
uv sync
```

## 模型部署
使用脱离openmm框架方式进行模型导出，只导出模型，不带框解码
```python
python ./projects/easydeploy/tools/export_onnx.py \
    configs/rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py \
    rtmdet_tiny_syncbn_fast_8xb32-300e_coco_20230102_140117-dbb1dc83.pth \
    --work-dir rtmdet_tiny \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
    --opset 13 \
    --model-only
```

onnx 推理测试
```python
python ./projects/easydeploy/examples/main_onnxruntime.py \
	bus.jpg \
	rtmdet_tiny/rtmdet_tiny_syncbn_fast_8xb32-300e_coco_20230102_140117-dbb1dc83.onnx \
	--type rtmdet 
```
端到端模型
```python
python -m projects.easydeploy.tools.image-demo \ 
demo/dog.jpg \
configs/rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py \
rtmdet_tiny/rtmdet_tiny_syncbn_fast_8xb32-300e_coco_20230102_140117.onnx \
--device cpu
```

使用mmdeploy转换模型
mmyolo rtmdet

``` bash
python3 tools/deploy.py \
    configs/mmdet/detection/detection_onnxruntime_static.py \
    ../mmyolo/configs/rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py  \
    mmyolo/rtmdet_tiny_syncbn_fast_8xb32-300e_coco_20230102_140117-dbb1dc83.pth \
    demo/resources/det.jpg \
    --work-dir work_dir \
    --show \
    --device cpu
```

mmdetection rtmdet
``` bash
python3 tools/deploy.py \
    configs/mmdet/detection/detection_onnxruntime_static.py \
    ../mmdetection/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py  \
    mmdetection/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth \
    demo/resources/det.jpg \
    --work-dir work_dir \
    --device cpu
```

mmpose rtmpose
``` bash
python3 tools/deploy.py \
    configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py \
    ../mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-t_8xb256-420e_coco-256x192.py \
    /mnt/quedoulin/github_code/mmpose/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth \
    demo/resources/det.jpg \
    --work-dir Pose_work_dir \
    --device cpu
```