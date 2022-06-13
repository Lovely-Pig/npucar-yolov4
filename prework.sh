git submodule update --init --recursive

pip install -r requirements.txt

wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights

cd data/custom
python write_img_paths.py
python voc2yolo.py
cd ../..
