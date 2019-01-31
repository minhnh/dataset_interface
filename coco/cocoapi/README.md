# Partial import of [COCO API](https://github.com/cocodataset/cocoapi/)
Here there is only the `subtree` of the Python interface, as the Matlab and LUA interfaces are not used.

* Building requires `cython`:
```
pip install --user cython
```
* To build, run `make` under the `PythonAPI` directory, then the sample notebooks will be able to import `pycocotools`
* To install, execute `setup.py`:
```
python setup.py install --user
```
Note: tested with Python 3.7
