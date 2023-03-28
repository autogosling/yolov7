
# YOLOv7 for Autogosling

This repository is an adapted codebase from https://github.com/WongKinYiu/yolov7 that allows the training of images and bounding boxes that contain multiple labels. The original model can only predict bounding boxes that contain  1 label each, while this fork aims to predict bounding boxes that contain at least one label from each different "type" (marks, orientation, and layout).

## Features

- Pre-configured for Autogosling datasets
- Multiple labels per bounding boxes
- Adjusted loss function to take care of multihot vectors instead of one hot vector (multiple labels fix)
- Contains post-processing to re-convert model output into desired format (list of labels)


## Installation

Download the Anaconda environment called `autogosling` by following the instructions found on the other repositories such as https://github.com/autogosling/autogosling-tool.

Make sure that `data/gosling.yaml` points to the correct datasets. Currently, the YAML file contains the following information.

```yaml
train: ../model/data/splits/split-42-0.2-0.1/yolov7-42-0.2-0.1/images/train
val: ../model/data/splits/split-42-0.2-0.1/yolov7-42-0.2-0.1/images/valid
test: ../model/data/splits/split-42-0.2-0.1/yolov7-42-0.2-0.1/images/test
 
# Classes
nc: 18 # number of classes

# names: [] # class names
names: ["area", "bar", "brush", "circular", "heatmap", "horizontal", "ideogram", "line", "linear", "point", "rect", "rule",  "text", "triangleBottom", "triangleLeft", "triangleRight", "vertical", "withinLink"] # class names
```

The `train`, `val`, and `test` attributes point to all the **images** of the dataset's corresponding split and expects the corresponding labels to be in a folder following this relative path: `../../labels/[SPLIT]`.

`nc` indicates the number of classes and `names` is simply the corresponding list of names. Given that we are using multi-label predictions for every bounding box, these labels will not directly be used by the model when training, instead, they will only be used when obtaining the final result.

The base model can be found at https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt.


## Usage/Examples

### Training
```sh
bash train.sh
```

Currently, train.sh follows the following structure:
```sh
export CUDA_LAUNCH_BLOCKING=1
CUDA_LAUNCH_BLOCKING=1 ipython --pdb train.py -- --device 0 --epochs 200 --workers 0 --batch 16 --data data/gosling.yaml --cfg cfg/training/gosling-training.yaml --weights 'yolov7_training.pt' --name yolov7_gosling_fixed_res --hyp data/hyp.scratch.custom.yaml
```

The `CUDA_LAUNCH_BLOCKING` flag is there to help debug CUDA-related errors, while the `ipython --pdb` opens a debugger whenever an error occurs during training for easier debugging. We use device 0, the GPU, 200 epochs over the dataset, a batch size of 16, and use the corresponding yaml files such as `data/gosling.yaml`, `cfg/training/gosling-training.yaml`, and `data/hyp.scratch.custom.yaml`, which simply details the correct paths to use and the correct model configurations.

### Testing
```sh
bash test.sh
```

Here is the current structure of test.sh
```sh
ipython test.py --pdb -- --weights runs/train/yolov7_gosling_fixed_res66/weights/best.pt --task test --data data/gosling.yaml 
```

Again, we use `ipython --pdb` to help debug the program in case of errors, use the corresponding model we want to evaluate (in this case, the best model from the 66th experiment of yolov7_gosling_fixed_res), and pass in the paths to the dataset as detailed by `data/gosling.yaml`.


### Detecting
```sh
bash detect.sh
```

The file currently contains this:
```sh
MODEL_PATH=runs/train/yolov7_gosling_fixed_res66/weights/best.pt
python detect.py --weights $MODEL_PATH --conf 0.25 --img-size 640 --source demo.png
```

We set the `MODEL_PATH` to point to the desired Pytorch model, and run `detect.py` on the image called `demo.png` (which can be changed for any other image) and get the outputted detection in `yolov7/runs/detect`.

### Exporting
```sh
bash export.sh
```

The file currently looks like this:
```sh
MODEL_PATH=runs/train/yolov7_gosling_fixed_res66/weights/best.pt
python export.py  --weights $MODEL_PATH --grid --end2end --simplify \
        --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
```
This allows you to export an ONNX model that can be used with autogosling-flask. Simply modify the `MODEL_PATH` variable to point towards the desired model and run the bash script to obtain the exported model.


## Code Changes

**Original** (yolov7/loss.py:117)

```py
target_bins[range(n), bin_idx] = self.cp
```

**Modified**
```py
multihot_mask = multihot(bin_idx,self.nc).long()
target_bins[multihot_mask] = self.cp
```

**Original**
```py
original = F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
gt_cls_per_image = (
    F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
    original
```

**Modified**
```py
test = multihot(this_target[:,1],self.nc).cuda().to(torch.int64)
# original = F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
gt_cls_per_image = (
    F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
    test
```
This change allows us to "label" the "correct" entries even when using multiple labels. Instead of only labelling the "single" correct label for each bounding box, we construct a Tensor that contains a grid of 0s and 1s, where the 1s correspond to positive classes.

This change is repeated across multiple references in the code and is usually the general fix that is needed to convert from the One-Hot format to Multi-hot. The pattern can be observed by finding variables of the following structure: `X[range(l), indices]`.

As well, there are some code changes that are related to GPU and CPU mismatch errors when using Pytorch Tensors. Operations on Pytorch Tensors can only be performed when all tensors are on the same device, so to fix the previous device errors, we use:
```py
torch.set_default_tensor_type('torch.cuda.FloatTensor')
```

There are also many other code changes, which relate to small bug fixes that again concern device mismatch errors or the program crashing due to encountering multi-hot encodings instead of one-hot encodings.


## Next Steps

- [ ]   Correct the loss function so that it can improve the mAP score when training/testing. The low accuracy is due to some features for single-label that have not yet been translated into multi-label.
- [ ]   Fix the post-processing step of Yolov7 to handle bounding boxes with multiple labels each. Currently, the program enforces a single label per bounding box at post-processing, which must be corrected by changing the argmax into a simple `np.arange(labels.shape[0])[labels > threshold]` instead, which will get all the indices where the label is deemed present in the bounding box.
## Team

- [@mnqng](https://www.github.com/mnqng)/[@mq-liang](https://github.com/mq-liang)
- [@katrina-liu](https://github.com/katrina-liu)
- [@wangqianwen0418](https://github.com/wangqianwen0418)
- [@ngehlenborg](https://github.com/ngehlenborg)