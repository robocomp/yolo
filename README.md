# yolo

A YoloV3 server and client for RoboComp


Install Yolo (see https://pjreddie.com/darknet/yolo/) in a folder outside RoboComp

Make sure you have a recent version of opencv and CUDA  (10.2 works fine)

You need to make the following modifications in the code before compiling it:

#File: darknet.h

Line23
#ifdef __cplusplus
extern "C" {
	namespace yolo
 {
#endif
 
Line 808

}
using namespace yolo;


#File: yolov3.cfg

batch=1
subdivisions=1
#batch=64
#subdivisions=16

#File: image.h
Line 64:
image make_image(int w, int h, int c);

#File: image_opencv.cpp
Line 63:
//    IplImage ipl = m;
//    image im = ipl_to_image(&ipl);
//    rgbgr_image(im);
      return image();

#File: makefile
GPU=1
CUDNN=1
OPENCV=1
OPENMP=1
DEBUG=0

#ARCH= -gencode arch=compute_30,code=sm_30 \
#      -gencode arch=compute_35,code=sm_35 \
#      -gencode arch=compute_50,code=[sm_50,compute_50] \
#      -gencode arch=compute_52,code=[sm_52,compute_52]
#      -gencode arch=compute_20,code=[sm_20,sm_21] \ This one is deprecated?

# This is what I use, uncomment if you know your arch and want to specify
ARCH= -gencode arch=compute_52,code=compute_52

VPATH=./src/:./examples
SLIB=libdarknet.so
ALIB=libdarknet.a
EXEC=darknet
OBJDIR=./obj/

CC=gcc
CPP=g++
NVCC=nvcc 
AR=ar
ARFLAGS=rcs
OPTS=-O3
LDFLAGS= -lm -pthread 
COMMON= -Iinclude/ -Isrc/
CFLAGS=-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC -mavx2

Set the name of your installation path in CMakeListsSpecific.txt for the library libyolo.so and the include directory.

Create a folder in YOLO folder named "yolodata" and copy there your selection of wights and names. The names are defined in the component.
For example:
cfg  coco.data  coco.names  yolov2.weights  yolov3-tiny.weights  yolov3.weights

cfg is a folder with the configuration file you want to use. The component expects: yolov3.cfg  

Dowload weights with: wget https://pjreddie.com/media/files/yolov3.weights 

Compile to obtain the dynamic library libyolo.so. 



