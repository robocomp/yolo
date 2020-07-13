# yolo

A YoloV3 server and client for RoboComp


Installing Yolo (see https://pjreddie.com/darknet/yolo/)

sudo apt-get install libopencv-dev nvidia-cuda-dev nvidia-cuda-toolkit 

cd src/yololib 
make 
sudo cp libyolo.so /usr/local/lib 
cd ../.. 
cmake . 
make 
cp -r src/yololib/data/ . wget https://pjreddie.com/media/files/yolo.weights -O src/yololib/yolo.weights


## WARNING
darknet.h has to be modified with a namespace. Add these lines and the closing line at the end of the file.

~~~~
#ifdef __cplusplus
extern "C" {
   namespace yolo
 {
~~~~
 and at the end
~~~~
 {
 {
#endif
using namespace yolo;
~~~~
