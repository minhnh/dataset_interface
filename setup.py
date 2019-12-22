from setuptools import setup, Extension
import numpy as np

COCO_API_PATH = 'coco/cocoapi/'

ext_modules = [
    Extension(
        'pycocotools._mask',
        sources=[COCO_API_PATH + 'common/maskApi.c', COCO_API_PATH + 'PythonAPI/pycocotools/_mask.pyx'],
        include_dirs=[np.get_include(), COCO_API_PATH + 'common'],
        extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
    )
]

setup(
    name='dataset_interface',
    packages=['pycocotools', 'dataset_interface', 'dataset_interface.common', 'dataset_interface.coco',
              'dataset_interface.augmentation', 'dataset_interface.object_detection'],
    package_dir={
        'dataset_interface': '.',
        'pycocotools': COCO_API_PATH + 'PythonAPI/pycocotools'
    },
    install_requires=[
        'setuptools>=18.0',
        'cython>=0.27.3',
        'pyyaml>=5.1.0',
        'future',
        'six'
    ],
    version='1.0',
    ext_modules=ext_modules
)
