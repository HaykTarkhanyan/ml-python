**CNN's architecture is**

ConvLayer(32 filters of size (3,3)) --> ReLU --> ConvLayer(64 filters of size (3,3)) --> ReLU -->
--> MaxPooling with filter size 2 and stride 2 --> Dropout(0,25) --> FC(128) --> ReLU  --> Dropout(0.5) --> Softmax(10 classes)

**Data description**

Dataset is in data folder it contains 42.000 train images and 28.000 for test
images are 28x28 grayscale images taken from MNIST dataset


**COMMANDS:**

* python train.py                                     
* python predict.py -inp_dir path/to/your/image       
