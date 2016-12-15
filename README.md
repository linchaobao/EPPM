EPPM
====================================

Code by [Linchao Bao](https://sites.google.com/site/linchaobao/) (linchaobao@gmail.com)

#### About

EPPM is an implementation of the optical flow algorithm presented in the paper "Fast Edge-Preserving PatchMatch for Large Displacement Optical Flow" in CVPR 2014. 

The source code is different from the original paper in the following ways:

1. The self-similarity propagation scheme proposed in the paper is not used in this implementation. Instead, the implementation uses pixel skipping scheme to reduce patch matching computation. 

2. The parameter setting are not the same as that for benchmarking in the paper. 


#### Compiling

The code relies on NVIDIA CUDA (version > 5.0) and only works on NVIDIA GPUs. To compile the code, run the following command: 

	
	mkdir build
	cd build
	cmake ..
	make


#### Citing

Please cite the following paper if you find the code useful: 


	@inproceedings{bao2014cvpreppm,
	  title={Fast Edge-Preserving PatchMatch for Large Displacement Optical Flow},
	  author={Bao, Linchao and Yang, Qingxiong and Jin, Hailin},
	  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	  year={2014},
	  pages={3534-3541}, 
	  organization={IEEE}
	}


#### License & Disclaimer

The code is released under an [MIT license](https://en.wikipedia.org/wiki/MIT_License) and is developed for academic purpose. Please use the code in your own risk. 

