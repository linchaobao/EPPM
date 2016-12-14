/* This file is part of the EPPM source code package. 
 *
 * Copyright (c) 2013-2016 Linchao Bao (linchaobao@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

/**
 * Basic host code for CUDA. 
 */
#ifndef _BAO_BASIC_CUDA_H_
#define _BAO_BASIC_CUDA_H_


#pragma warning( disable : 4819 ) // code page (936)
#pragma warning( disable : 4305 ) // initialization truncation from 'double' to 'float'
#pragma warning( disable : 4244 ) // conversion from 'double' to 'float'
#include <cuda_runtime.h>
#include <curand_kernel.h> //for generate random numbers on device
#include "../3rdparty/nv-cuda-v5.0/helper_cuda.h"  //common helper functions from NVIDIA

typedef double BAO_FLOAT;

#ifndef _WIN32
#define __max(a,b) (((a) > (b)) ? (a) : (b))
#define __min(a,b) (((a) < (b)) ? (a) : (b))
#include <inttypes.h>
typedef int64_t __int64;
#endif


//////////////////////////////////////////////////////////////////////////
// Timer
#ifndef _WIN32
#include <sys/time.h>
#endif

class bao_timer_gpu
{
public:
    bao_timer_gpu();
    ~bao_timer_gpu();
    void start();
    double stop();
    double time_display(char *disp="",int nr_frame=1);
    double fps_display(char *disp="",int nr_frame=1);
private:
    cudaEvent_t m_start;
    cudaEvent_t m_stop;
};

class bao_timer_gpu_cpu //synchronize between cpu and gpu time
{
public: 
    void start(); 
    double stop(); 
    double time_display(char* disp="", int nr_frame=1); 
    double fps_display(char* disp="", int nr_frame=1); 
private:
#ifdef _WIN32
    double m_pc_frequency; 
    __int64 m_counter_start;
#else
    struct timeval timerStart;
#endif
    
};


//////////////////////////////////////////////////////////////////////////
// Memory allocate and free
inline void bao_cuda_init()
{
    int devID = findCudaDevice();
}

template<typename T>
inline T* bao_cuda_alloc(int len) //pitch is for output
{
    T* img_ptr;
    checkCudaErrors(cudaMalloc(&img_ptr, sizeof(T)*len));
    return img_ptr;
}

template<typename T>
inline T* bao_cuda_alloc(int h, int w) //pitch is for output
{
    T* img_ptr;
    checkCudaErrors(cudaMalloc(&img_ptr, sizeof(T)*w*h));
    return img_ptr;
}

template<typename T>
inline T* bao_cuda_alloc_pitched(size_t& pitchOut, int h, int w) //pitch is for output
{
    T* img_ptr;
    checkCudaErrors(cudaMallocPitch(&img_ptr, &pitchOut, sizeof(T)*w, h));
    return img_ptr;
}

template<typename T>
inline void bao_cuda_free(T* &ptr) 
{
    checkCudaErrors(cudaFree(ptr));
    ptr = NULL;
}

template<typename T>
inline void bao_cuda_free_pitched(T* &ptr) //the same as non-pitched
{
    checkCudaErrors(cudaFree(ptr));
    ptr = NULL;
}

template<typename T>
inline void bao_cuda_memset(T* img_ptr, int len)
{
    checkCudaErrors(cudaMemset(img_ptr, 0, sizeof(T)*len));
}

template<typename T>
inline void bao_cuda_memset(T* img_ptr, int h, int w)
{
    checkCudaErrors(cudaMemset(img_ptr, 0, sizeof(T)*h*w));
}

template<typename T>
inline void bao_cuda_memset_pitched(T* img_ptr, size_t pitch, int h, int w) //for 2D pitched memory
{
    checkCudaErrors(cudaMemset2D(img_ptr, pitch, 0, sizeof(T)*w, h));
}

template<typename T>
inline void bao_cuda_copy_h2d(T* dev_ptr, T* host_ptr, int len)
{
    checkCudaErrors(cudaMemcpy(dev_ptr, host_ptr, sizeof(T)*len, cudaMemcpyHostToDevice));
}

template<typename T>
inline void bao_cuda_copy_h2d(T* dev_ptr, T* host_ptr, int h, int w)
{
    checkCudaErrors(cudaMemcpy(dev_ptr, host_ptr, sizeof(T)*w*h, cudaMemcpyHostToDevice));
}

template<typename T>
inline void bao_cuda_copy_h2d_pitched(T* dev_ptr, size_t dev_pitch, T* host_ptr, int h, int w)
{
    checkCudaErrors(cudaMemcpy2D(dev_ptr, dev_pitch, host_ptr, sizeof(T)*w, sizeof(T)*w, h, cudaMemcpyHostToDevice));
}

template<typename T>
inline void bao_cuda_copy_d2h(T* host_ptr, T* dev_ptr, int len)
{
    checkCudaErrors(cudaMemcpy(host_ptr, dev_ptr, sizeof(T)*len, cudaMemcpyDeviceToHost));
}

template<typename T>
inline void bao_cuda_copy_d2h(T* host_ptr, T* dev_ptr, int h, int w)
{
    checkCudaErrors(cudaMemcpy(host_ptr, dev_ptr, sizeof(T)*w*h, cudaMemcpyDeviceToHost));
}

template<typename T>
inline void bao_cuda_copy_d2h_pitched(T* host_ptr, T* dev_ptr, size_t dev_pitch, int h, int w)
{
    checkCudaErrors(cudaMemcpy2D(host_ptr, sizeof(T)*w, dev_ptr, dev_pitch, sizeof(T)*w, h, cudaMemcpyDeviceToHost));
}

template<typename T>
inline void bao_cuda_copy_d2d(T* dev_ptr1, T* dev_ptr2, int len)
{
    checkCudaErrors(cudaMemcpy(dev_ptr1, dev_ptr2, sizeof(T)*len, cudaMemcpyDeviceToDevice));
}

template<typename T>
inline void bao_cuda_copy_d2d(T* dev_ptr1, T* dev_ptr2, int h, int w)
{
    checkCudaErrors(cudaMemcpy(dev_ptr1, dev_ptr2, sizeof(T)*w*h, cudaMemcpyDeviceToDevice));
}

template<typename T>
inline void bao_cuda_copy_d2d_pitched(T* dev_ptr1, size_t dev_pitch1, T* dev_ptr2, size_t dev_pitch2, int h, int w)
{
    checkCudaErrors(cudaMemcpy2D(dev_ptr1, dev_pitch1, dev_ptr2, dev_pitch2, sizeof(T)*w, h, cudaMemcpyDeviceToDevice));
}


//////////////////////////////////////////////////////////////////////////
// Pyramid operation
template<typename T>
inline T** bao_cuda_pyr_alloc(int nLevels, int* arrH, int* arrW) //NOTE: top level pointer is on host!
{
    T** pPyr = (T**)malloc(nLevels*sizeof(T*)); //NOTE: top level pointer is on host!
    for (int i=0; i<nLevels; i++)
    {
        pPyr[i] = bao_cuda_alloc<T>(arrH[i],arrW[i]);
    }
    return pPyr;
}

template<typename T>
inline T** bao_cuda_pyr_alloc_pitched(size_t* outArrPitch, int nLevels, int* arrH, int* arrW) //NOTE: top level pointer is on host!
{
    T** pPyr = (T**)malloc(nLevels*sizeof(T*)); //NOTE: top level pointer is on host!
    for (int i=0; i<nLevels; i++)
    {
        pPyr[i] = bao_cuda_alloc_pitched<T>(outArrPitch[i], arrH[i],arrW[i]);
    }
    return pPyr;
}

template<typename T>
inline void bao_cuda_pyr_free(T** &p, int nLevels)
{
    if (p == NULL) return;
    for (int i=0; i<nLevels; i++)
    {
        bao_cuda_free(p[i]);
    }
    free(p); //DO NOT use bao_free!
    p = NULL;
}

template<typename T>
inline void bao_cuda_pyr_free_pitched(T** &p, int nLevels) //the same as non-pitched
{
    if (p == NULL) return;
    for (int i=0; i<nLevels; i++)
    {
        bao_cuda_free(p[i]);
    }
    free(p); //DO NOT use bao_free!
    p = NULL;
}


//////////////////////////////////////////////////////////////////////////
// Basic conversion (CPU code)
inline void bao_rgb2rgba(uchar4** img_out, unsigned char*** img_in, int h, int w)
{
    for (int i=0;i<h;i++) for(int j=0;j<w;j++) 
    {
        img_out[i][j].x = img_in[i][j][0];
        img_out[i][j].y = img_in[i][j][1];
        img_out[i][j].z = img_in[i][j][2];
        img_out[i][j].w = 0;
    }
}

inline void bao_rgb2rgba(float4** img_out, unsigned char*** img_in, int h, int w)
{
    for (int i=0;i<h;i++) for(int j=0;j<w;j++) 
    {
        img_out[i][j].x = img_in[i][j][0];
        img_out[i][j].y = img_in[i][j][1];
        img_out[i][j].z = img_in[i][j][2];
        img_out[i][j].w = 0;
    }
}

inline void bao_rgba2rgb(unsigned char*** img_out, uchar4** img_in, int h, int w)
{
    for (int i=0;i<h;i++) for(int j=0;j<w;j++) 
    {
        img_out[i][j][0] = img_in[i][j].x;
        img_out[i][j][1] = img_in[i][j].y;
        img_out[i][j][2] = img_in[i][j].z;
    }
}

inline void bao_rgba2rgb(unsigned char*** img_out, float4** img_in, int h, int w)
{
    for (int i=0;i<h;i++) for(int j=0;j<w;j++) 
    {
        img_out[i][j][0] = __max(0,__min(255,img_in[i][j].x*255));
        img_out[i][j][1] = __max(0,__min(255,img_in[i][j].y*255));
        img_out[i][j][2] = __max(0,__min(255,img_in[i][j].z*255));
    }
}

inline unsigned int bao_div_ceil(unsigned int a, unsigned int b)
{
    return (a+b-1)/b;
}

inline void bao_cuda_sync()
{
    checkCudaErrors(cudaDeviceSynchronize());
}



#endif
