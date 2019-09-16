# Table of Contents

  * [saliency](#saliency)
  * [saliency.salienc](#saliency.salienc)
    * [AuthenticationError](#saliency.salienc.AuthenticationError)
    * [SaliencyClient](#saliency.salienc.SaliencyClient)

# `saliency`


# `saliency.salienc`


## `AuthenticationError` Objects


## `SaliencyClient` Objects

```python
def __init__(self, username, password, host="https://api.saliency.ai/")
```

The entry point to Saliency.ai. Object of this class will allow you to
* Log-in to your Saliency.ai account
* Upload datapoints
* Create tasks for labeling
* Query data in the requested format
* Train and deploy models

In this document we list all methods available in `SaliencyClient`.

Attributes:
    flight_speed     The maximum speed that such a bird can attain.

### `SaliencyClient.__init__()`

```python
def __init__(self, username, password, host="https://api.saliency.ai/")
```


### `SaliencyClient.login()`

```python
def login(self)
```


### `SaliencyClient.add_annotation_scheme()`

```python
def add_annotation_scheme(self, name, type, labels)
```

TODO: check labels formatting

### `SaliencyClient.add_task()`

```python
def add_task(self, scheme, study)
```

Summary or Description of the Function

Parameters:
argument1 (int): Description of arg1

Returns:
int:Returning value

### `SaliencyClient.add_study()`

```python
def add_study(self, study, study_name)
```


### `SaliencyClient.get_mri()`

```python
def get_mri(self, tmpfile)
```


### `SaliencyClient.get_xray()`

```python
def get_xray(self, tmpfile)
```


### `SaliencyClient.get_xray_anno()`

```python
def get_xray_anno(self, tmpfile)
```


### `SaliencyClient.get_zip_mri_anno()`

```python
def get_zip_mri_anno(self, tmpfile, shape)
```


### `SaliencyClient.get_file()`

```python
def get_file(self, file)
```


### `SaliencyClient.get_annotation()`

```python
def get_annotation(self, file, shape, headers)
```


### `SaliencyClient.create_task()`

```python
def create_task(self, scheme_id, study_id, datatype="xray")
```


### `SaliencyClient.download_datapoints()`

```python
def download_datapoints(self, patient="", study="", datatype="xray")
```


### `SaliencyClient.create_arrays()`

```python
def create_arrays(self, values, width, height)
```


### `SaliencyClient.resize_image()`

```python
def resize_image(self, image, width, height)
```


### `SaliencyClient._list_objects()`

```python
def _list_objects(self, obj, obj_id=None)
```

Method to cover all listings.

param obj: is string, that represents an object name in the API. (e.g. 'tasks')
param obj_id: is actual id of the object in the API

### `SaliencyClient._delete_objects()`

```python
def _delete_objects(self, obj, obj_id)
```

Method to delete object.

param:obj: is string, that represents an object name in the API. (e.g. 'tasks')
param:obj_id is actual id of the object in the API

### `SaliencyClient.list_tasks()`

```python
def list_tasks(self, task_id=None)
```

Method to list all tasks.

:param task_id: if passed function would return information for specific task.

### `SaliencyClient.delete_task()`

```python
def delete_task(self, task_id)
```

Method to delete specific task.

param:task_id is actual id of the task in the API

