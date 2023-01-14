# Anchors for Object Detections
This repo has two purposes.
1. generate anchors for prediction
2. assign targets to the anchors for obtaining the loss 
  
  
## Anchor-Maker
Anchor-based detection models have been mainly developed over four series.
- Faster R-CNN
- YOLO
- RetinaNet
- SSD

Each anchor-based detection model series has a different manners of creating anchor priors, creating anchors, and combining them with the model's regression predictions.
Despite slight variations from model to model, models included in a series basically share the same manners.

Now let's see the examples of how Anchor-Maker gives its anchors to models 🔥

### YOLO
```python
img_size = 416
anchor_sizes = [
    [(10, 13), (16, 30), (33, 23)], 
    [(30, 61), (62, 45), (59, 119)], 
    [(116, 90), (156, 198), (373, 326)]
]
strides = [8, 16, 32]
anchors = yolo_anchors(img_size, anchor_sizes, strides)

for i, s in enumerate(strides):
    pred[i][..., :2] = torch.sigmoid(pred[i][..., :2]) * s + anchors[i][..., :2]
    pred[i][..., 2:4] = torch.exp(pred[i][..., 2:4]) * anchors[i][..., 2:]
```
Here, we should note that in YOLO, unlike the others, anchors and model predictions are a list of tensors on each stride, not tensors flattened over given strides.

### RetinaNet
```python
img_size = 608
anchor_sizes = [32, 64, 128, 256, 512]
anchor_scales = [1, 2**(1/3), 2**(2/3)]
aspect_ratios = [(2**(1/2), 2**(-1/2)), (1, 1), (2**(-1/2), 2**(1/2))]
strides = [8, 16, 32, 64, 128]

anchors = retinanet_anchors(img_size, anchor_sizes, anchor_scales, aspect_ratios, strides)

pred[..., :2] = anchors[..., :2] + (pred[..., :2] * anchors[..., 2:])
pred[..., 2:4] = torch.exp(pred[..., 2:4]) * anchors[..., 2:]
```

### SSD
```python
img_size = 300
anchor_sizes = [21, 45, 99, 153, 207, 261]
upper_sizes = [45, 99, 153, 207, 261, 315]
strides = [8, 16, 32, 64, 100, 300]
num_anchors = [4, 6, 6, 6, 4, 4]

anchors = ssd_anchors(img_size, anchor_sizes, upper_sizes, strides, num_anchors)

pred[..., :2] = anchors[..., :2] + (center_variance * pred[..., :2] * anchors[..., 2:])
pred[..., 2:4] = torch.exp(size_variance * pred[..., 2:4]) * anchors[..., 2:]
```

Note that the important thing is that anchor priors and anchors can be generated seperately. The above cases are merely examples that are generally used.  
  

## Anchor-Assigner
It is very necessary to map the predictions and target labels for obtaining the detection models' loss. Anchor-Assigner object assigns targets to the anchors, and then return indices of target-assigned anchors and lables of assigned targets.
  
This implementation can be commonly applied to all series of detection models.

```python
assigner = retinanet_assigner(0.5, 0.4)
assigns = assigner(labels, anchors)

for pred, assign in zip(preds, assigns):
    fore_idx, fore_label = assign['foreground']
    back_idx, _ = assign['background']

    cls_loss = self.focal_loss(pred[..., 4:], fore_idx, back_idx, fore_label[..., 4:])
    reg_loss = self.smooothL1_loss(pred[..., :4], anchors, fore_idx, fore_label[..., :4])
``` 

## License
BSD 3-Clause License Copyright (c) 2022, Kwon Taewan
