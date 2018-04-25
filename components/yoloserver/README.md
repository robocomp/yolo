```
```
#
``` yoloserver
```


sudo apt-get install libopencv-dev nvidia-cuda-dev nvidia-cuda-toolkit
cd src/yololib
make
sudo cp libyolo.so /usr/local/lib
cd ../..
cmake .
make
cp -r src/yololib/data/ .
wget https://pjreddie.com/media/files/yolo.weights -O src/yololib/yolo.weights





## Configuration parameters
As any other component,
``` *yoloserver* ```
needs a configuration file to start. In

    etc/config

you can find an example of a configuration file. We can find there the following lines:

    EXAMPLE HERE

    
## Starting the component
To avoid changing the *config* file in the repository, we can copy it to the component's home directory, so changes will remain untouched by future git pulls:

    cd

``` <yoloserver 's path> ```

    cp etc/config config
    
After editing the new config file we can run the component:

    bin/

```yoloserver ```

    --Ice.Config=config
