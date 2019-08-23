CNN's architecture is 

ConvLayer(32 filters of size (3,3)) --> ReLU --> ConvLayer(64 filters of size (3,3)) --> ReLU -->
--> MaxPooling with filter size 2 and stride 2 --> Dropout(0,25) --> FC(128) --> ReLU  --> Dropout(0.5) --> Softmax(10 classes)

Dataset is in data folder it contains 42.000 train images and 28.000 images for test
images are 28x28 grayscale images taken from MNIST dataset

check_input file is for making sure that network has been given valid image
check_input looks if directory was wrong or image was extremely big or small

COMMANDS:

python train.py -> Trains model and saves results in ckpt folder
python predict.py -inp_dir path/to/your/image -> loades model from ckpt and if input is valid predicts the label 
python CNN.py -> runs full code without spliting it to parts, also visualizes losses ang inputs
