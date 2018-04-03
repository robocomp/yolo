# yolo

A Yolo server and client for RoboComp


Installing Yolo (see https://pjreddie.com/darknet/yolo/)

sudo apt-get install libopencv-dev nvidia-cuda-dev nvidia-cuda-toolkit 
cd src/yololib make sudo cp libyolo.so /usr/local/lib 
cd ../.. 
cmake . 
make 
cp -r src/yololib/data/ . wget https://pjreddie.com/media/files/yolo.weights -O src/yololib/yolo.weights
