import math
import time
import onnxruntime
import cv2
import numpy as np

CLASS_NAMES = (['person'])

def yolo_preprocess(image, infer_shape):
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    mean = mean.reshape((3, 1, 1))
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    std = std.reshape((3, 1, 1))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    ratio_h, ratio_w = infer_shape[0] / height, infer_shape[1] / width
    image = cv2.resize(
        image, (0, 0),
        fx=ratio_w,
        fy=ratio_h,
        interpolation=cv2.INTER_LINEAR)
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    image = image.astype(np.float32)
    image -= mean
    image /= std
    return image[np.newaxis], (ratio_w, ratio_h)

def nms_boxes(boxes, scores, iou_thres):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        agnostic=False,
        multi_label=False,
        max_det=300,
        nc=0,  # number of classes (optional)
        max_nms=30000,
        max_wh=7680,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Arguments:
        prediction (np.ndarray): A tensor of shape (batch_size, num_boxes, num_classes + 4 + num_masks)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, np.ndarray]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int): (optional) The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels

    Returns:
        (List[np.ndarray]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    prediction = np.transpose(prediction, (0, 2, 1))

    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    # 如果 prediction 是 numpy 数组，则可以这样计算 xc
    xc = prediction[:, 4:mi].max(1) > conf_thres

    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    output = [np.zeros((0, 6 + nm))] * bs

    for xi, x in enumerate(prediction):  # image index, image inference
        x = prediction[xi]

        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x.transpose((1, 0))[xc[xi]]  # confidence
        
        if not x.shape[0]:
            continue

        box, cls, mask = np.split(x, [4, 4 + nc, 4 + nc + nm], axis=1)[:3]

        # box = xywh2xyxy(box)

        conf = np.amax(cls, axis=1, keepdims=True)
        j = np.expand_dims(np.argmax(cls, axis=1), axis=1)
        x = np.concatenate([box, conf, j.astype(np.float32), mask], 1)[conf.reshape(-1) > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[np.argsort(x[:, 4])[::-1]][:max_nms]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = nms_boxes(boxes, scores, iou_thres)  # NMS

        i = i[:max_det]  # limit detections

        output[xi] = x[i]

    return output

def predict():
    img_path = '/mnt/quedoulin/codeup_workspace/AlgManagerImpl/person.jpg'
    model_file = '/mnt/quedoulin/github_code/mmyolo/rtmdet_nano_person.onnx'
    NUM_CLASSES = 1
    box_conf_threshold = 0.3
    box_iou_threshold = 0.3
    shape = (640, 640)
    img = cv2.imread(img_path)
    image_h, image_w = img.shape[:2]
    for i in range(1):
        t = time.time()
        blob, (ratio_w, ratio_h) = yolo_preprocess(img, shape)
        session = onnxruntime.InferenceSession(model_file, None)
        output = session.run(None, {session.get_inputs()[0].name: blob})
        preds = non_max_suppression(output[0],
                                    box_conf_threshold,
                                    box_iou_threshold,
                                    nc=NUM_CLASSES
                                    )
        pred = preds[0]
        # raw_shape  = img.shape
        # pred[:, :4] = scale_boxes(shape, pred[:, :4], raw_shape[:2]).round()
        boxes = pred[:, :6]
        for box in boxes:
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            x0 = math.floor(min(max(x0 / ratio_w, 1), image_w - 1))
            y0 = math.floor(min(max(y0 / ratio_h, 1), image_h - 1))
            x1 = math.ceil(min(max(x1 / ratio_w, 1), image_w - 1))
            y1 = math.ceil(min(max(y1 / ratio_h, 1), image_h - 1))
            print(x0, y0, x1, y1, box[4], box[5])
            cv2.rectangle(img, (x0, y0), (x1, y1), (255,255,0), 2)
            cv2.putText(img, f'{CLASS_NAMES[int(box[5])]}: {box[4]:.2f}',
                        (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 2)
        t2 = time.time()
        print(f"cost time: {(t2 - t) * 1000} ms")
        cv2.imwrite('test_infer_person.jpg', img)
if __name__ == "__main__":
    predict()