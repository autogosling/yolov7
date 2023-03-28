MODEL_PATH=runs/train/yolov7_gosling_fixed_res66/weights/best.pt
python detect.py --weights $MODEL_PATH --conf 0.25 --img-size 640 --source demo.png