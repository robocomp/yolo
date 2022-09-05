# yolov5_sort
Intro to component here

Remove name from model.name in reid_model_factory.py
def get_model_url(model):
    if model in __trained_urls:
        return __trained_urls[model]
    else:
        None


def is_model_in_model_types(model):
    if model.name in __model_types:
        return True
    else:
        return False


def get_model_name(model):
    for x in __model_types:
        if x in model:
            return x
    return None

in stron_sort/sort/track.py   change line 154 from 

if (src.any() or dst.any() is None):

to

if (src or dst is None):

## Configuration parameters
As any other component, *yolov5_sort* needs a configuration file to start. In
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
cd <yolov5_sort's path> 
```
```
cp etc/config config
```

After editing the new config file we can run the component:

```
bin/yolov5_sort config
```
