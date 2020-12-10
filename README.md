# Lane Detection Using Convolution LSTM 
This project is based on "Zou Q, Jiang H, Dai Q, Yue Y, Chen L and Wang Q, Robust Lane Detection from Continuous Driving Scenes Using Deep Neural Networks, IEEE Transactions on Vehicular Technology, 2019." and their code https://github.com/qinnzou/Robust-Lane-Detection.
# Dataset 
You can download this dataset from the link in the 'Dataset-Description-v1.2.pdf' file.
BaiduYun：
https://pan.baidu.com/s/1lE2CjuFa9OQwLIbi-OomTQ passcodes：tf9x Or
Google Drive:
https://drive.google.com/drive/folders/1MI5gMDspzuV44lfwzpK6PX0vKuOHUbb_?usp=sharing

You can also download the pretrained model from the following link,
https://drive.google.com/drive/folders/19dF3sWpM_CRgrI46RxowS86eOo_1X1XU?usp=sharing

# Requirements
PyTorch 0.4.0
Python 3.6
CUDA 8.0

# Training 
Before training, change the paths including "train_path"(for train_index.txt), "val_path"(for val_index.txt), "pretrained_path" in config.py to adapt to your environment.
Choose the models(SegNet-ConvLSTM, UNet-ConvLSTM or SegNet, UNet) and adjust the arguments such as class weights, batch size, learning rate in config.py.
Then simply run:
```
python train.py
```

# Test 
To evlauate the performance of a pre-trained model, please put the pretrained model listed above or your own models into "./LaneDetectionCode/pretrained/" and change "pretrained_path" in config.py at first, then change "test_path" for test_index.txt, and "save_path" for the saved results.
Choose the right model that would be evlauated, and then simply run:
```
python test.py
```




