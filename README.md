# Prerequisite
## Virtual Environment
In folder clone from git, follow the steps below:
* Create venv: `python3 -m venv env`
* Activate venv: `source ./venv/bin/activate`
## Install Dependencies: 
* Tensorflow: `pip install tensorflow`  
* Keras: `pip install keras` 
* OpenCV: `pip install opencv-python`
# Run Test
- In project, run command: `python cli.py image-folder [save]`
- ***Note***: 
    - Use argument `save` to write images with label text in `results` folder after display.
    - If don't have argument `save`, images with label will just display.
    - You must press any key to display next image. 