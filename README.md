The repository is the entire code that was written during our experiments on 
our theis subject  - _efficient hydrodynamic model implementations
  in GPGPU architecture engineering_.

It consists of two parts:
 - src - there are all variants of _saturated flow model_ implementation
  parallel and sequential, written in Python, C, CUDA C and OpenCAL framework
 - tests - a script for compiling and testing programs which run on CPU and GPU,
  it can test CUDA C, C and OpenCAL files, we claim that is versatile enough
   that **can be useful in similar projects**, for more information check
    __settings.py__ and run script with -h option

Authors:
 - Paweł Renc
 - Tomasz Pęcak
