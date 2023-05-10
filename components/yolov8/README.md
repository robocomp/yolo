# yolov8
Intro to component here

Wrapper to use Yolov8 in TRT format
Dowload weights and scripts to convert from .pt from

https://github.com/Linaom1214/TensorRT-For-YOLO-Series/tree/main



## Configuration parameters
As any other component, *yolov8* needs a configuration file to start. In
```
etc/config
```
you can find an example of a configuration file. We can find there the following lines:
```
EXAMPLE HERE
```

## Starting the component
To avoid changing the *config* file in the repository, we can copy it to the component's home directory, so changes will remain untouched by future git pulls:

```
cd <yolov7_py's path> 
```
```
cp etc/config config
```

After editing the new config file we can run the component:

```
bin/yolov7_py config
```
