# Basketball Shot Tracker
This project shows how to train a YOLOv8 model for basketball ball and basketball rim detection. Then you can use tracker script for shot statistics tracking. Model uses a dataset of ball and rim images from [Roboflow platform](https://universe.roboflow.com/sportai/basketball-p0jcl). YOLOv8 model is trained in 100 epochs on a dataset of 17120 images.

### System Requirements
For the model training, a CUDA GPU is necessary, especially for larger dataset. I trained the model with an Nvidia Geforce GTX 1650. It might take a lot of time to process all of the data. 

### How to run
Before running the project you need to create a virtual environment and install required packages for GPU capability

#### Install packages from a requirements file:
```Bash
pip install -r requirements.txt
```

#### After that you can train YOLOv8 model with:
```Bash
python train.py
```

#### To evaluate the model run:
```Bash
python test.py
```

#### For shot tracking run:
```Bash
python shot_detector.py video_path
```



https://github.com/jovannajdovski/Basketball-Tracker/assets/100165980/6ecc2eac-9fa0-43dc-9201-7d753a8ac7b3

