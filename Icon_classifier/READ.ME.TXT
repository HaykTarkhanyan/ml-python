ANN's architecture is

Input - 25x25x3(flatten -> 1875 )
Dense - 100
Dense - 80
Dense - 80
Softmax - 5

Dataset is in data folder it contains classes and ~1000 images of each
Data train, valid split is  80/20

check_input file is for making sure that network has been given valid image
check_input looks if directory was wrong or image was extremely big or small or
has wrong number of channels

Diserable input image has size from 15px to 40px and is in rgb formal

COMMANDS:

python train.py   -> Trains model and saves results in ckpt folder
python predict.py -inp_dir path/to//image   -> loades model from ckpt and
                                             if input is valid predicts the label
python ANN.py -inp_dir path/to//image -> full code for checking
