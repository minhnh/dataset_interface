# Partial import of [COCO API](https://github.com/cocodataset/cocoapi/)
Here there is only the `subtree` of the Python interface, as the Matlab and LUA interfaces are not used.

* Building requires `cython`:
```
pip install --user cython
```
* To build, run `make` under the `PythonAPI` directory, then the sample notebooks will be able to import `pycocotools`
* To install only the Python COCO API, execute `setup.py`:
```
python setup.py install --user
```
* Top level [`setup.py`](../../setup.py) will also install `pycocotools` as a package
see [top level README](../../README.md) for more information

Note: tested with Python 3.7
