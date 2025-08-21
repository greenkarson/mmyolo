import argparse
from io import BytesIO

import onnx
import torch
import torch.nn.functional as F
from mmdet.apis import init_detector
from torch import nn


def build_model_from_cfg(config_path: str, checkpoint_path: str, device):
    model = init_detector(config_path, checkpoint_path, device=device)
    model.eval()
    return model

class RTMDetModel(torch.nn.Module):
    def __init__(self, model: nn.Module, device) -> None:
        super().__init__()
        self.model = model
        self.device = device

        self.prior_generate = self.model.bbox_head.prior_generator.grid_priors
        self.num_base_priors = self.model.bbox_head.num_base_priors
        self.featmap_strides = self.model.bbox_head.featmap_strides
        self.num_classes = self.model.bbox_head.num_classes


    def forward(self, x):
        ib = x.shape[0]
        clas, bbox = self.model(x)

        dtype = clas[0].dtype

        featmap_sizes = [cls_score.shape[2:] for cls_score in clas]
        mlvl_priors = self.prior_generate(featmap_sizes, dtype=dtype, device=self.device)
        flatten_priors = torch.cat(mlvl_priors)
        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size[0] * featmap_size[1] * self.num_base_priors,),stride) 
                for featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        # flatten cls_scores, bbox_preds
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(ib, -1, self.num_classes)
            for cls_score in clas
        ]
        cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()

        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(ib, -1, 4)
            for bbox_pred in bbox
        ]
        bbox_preds = torch.cat(flatten_bbox_preds, dim=1)

        priors = flatten_priors[None]
        stride = flatten_stride[None, :, None]
        bbox_preds *= stride
        tl_xy = (priors[..., :2] - bbox_preds[..., :2])
        br_xy = (priors[..., :2] + bbox_preds[..., 2:4])
        decoded_bboxes = torch.cat([tl_xy, br_xy], -1)

        # 使用 torch.split 拆分预测值
        # dxdy, dwdh = torch.split(bbox_preds, [2, 2], dim=-1)  # dxdy: [..., 2], dwdh: [..., 2]
        # 计算解码后的坐标
        # tl_xy = priors[..., :2] - dxdy  # 左上角 = 中心 - 偏移
        # br_xy = priors[..., :2] + dwdh  # 右下角 = 中心 + 缩放
        # decoded_bboxes = torch.cat([tl_xy, br_xy], dim=-1)  # [..., 4]
        
        output = torch.cat([decoded_bboxes, cls_scores], -1)
        return output


def parse_args():
    parser = argparse.ArgumentParser(
        description='convert rtmdet model to ONNX.')
    parser.add_argument(
        '--config', type=str, 
        default='configs/rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py',
        help='rtmdet config file path from mmdetection.')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='rtmdet_tiny_syncbn_fast_8xb32-300e_coco_20230102_140117-dbb1dc83.pth',
        help='rtmdet checkpoint path from mmdetection.')
    parser.add_argument('--output', type=str, default='rtmdet_tiny_syncbn_fast.onnx', help='output filename.')
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device used for inference')
    parser.add_argument(
        '--input-name', type=str, default='images', help='ONNX input name.')
    parser.add_argument(
        '--output-name', type=str, default='output', help='ONNX output name.')
    parser.add_argument(
        '--opset', type=int, default=11, help='ONNX opset version.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    model = build_model_from_cfg(args.config, args.checkpoint, args.device)
    rtmdet = RTMDetModel(model, args.device)
    rtmdet.eval()


    x = torch.randn((1, 3, 640, 640), device=args.device)
    y = rtmdet(x)
    print(y.shape)
    # y = model(x)
    # for i, (cls, box) in enumerate(zip(*y)):
    #     print(i, cls.shape, box.shape)
    #     box = box.permute(0, 2, 3, 1)
    #     print(box.shape)
    # exit()
    # torch.onnx.export(
    #     rtmdet,
    #     x,
    #     args.output,
    #     input_names=[args.input_name],
    #     output_names=[args.output_name],
    #     opset_version=args.opset)

    with BytesIO() as f:
        torch.onnx.export(rtmdet, x, f, verbose=False, opset_version=args.opset,
                            training=torch.onnx.TrainingMode.EVAL,
                            do_constant_folding=True,
                            input_names=[args.input_name],
                            output_names=[args.output_name]
                         )
        f.seek(0)
        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model

        import onnxsim
        print('\nStarting to simplify ONNX...')
        try:
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'assert check failed'
        except Exception as e:
            print(f'Simplifier failure: {e}')
        onnx.save(onnx_model, args.output)
        print(f'ONNX export success, save into {args.output}')