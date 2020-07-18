# yolo

A YoloV3 server and client for RoboComp


Install Yolo (see https://pjreddie.com/darknet/yolo/) in a folder outside RoboComp

Make sure you have a recent version of opencv and CUDA  (10.2 works fine)

[comment]: # cd src/yololib
#make 
#sudo cp libyolo.so /usr/local/lib 
#cd ../.. 
#cmake . 
#make 
#cp -r src/yololib/data/ . wget https://pjreddie.com/media/files/yolo.weights -O src/yololib/yolo.weights


## One 
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
