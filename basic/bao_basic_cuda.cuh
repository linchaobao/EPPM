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
 * Basic device code for CUDA. Can only be included in *.cu files. 
 */
#ifndef _BAO_BASIC_CUDA_CUH_
#define _BAO_BASIC_CUDA_CUH_

#include "bao_basic_cuda.h"

#define BLOCK_DIM_X  16
#define BLOCK_DIM_Y  16


template<typename T1, typename T2>
__global__ void _d_bao_add(T1*d_imgout, size_t out_mem_w, T2*d_img1, T2*d_img2, size_t in_mem_w, int h, int w)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    d_imgout[id_y*out_mem_w + id_x] = T1(d_img1[id_y*in_mem_w+id_x] + d_img2[id_y*in_mem_w+id_x]);
}
template<typename T1, typename T2>
inline void bao_cuda_add(T1*d_imgout, T2*d_img1, T2*d_img2, int h, int w)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    _d_bao_add<<<gridSize, blockSize>>>(d_imgout,w,d_img1,d_img2,w,h,w);
}
template<typename T1, typename T2>
inline void bao_cuda_add_pitched(T1*d_imgout, size_t pitch1, T2*d_img1, T2*d_img2, size_t pitch2, int h, int w)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    _d_bao_add<<<gridSize, blockSize>>>(d_imgout,pitch1/sizeof(T1),d_img1,d_img2,pitch2/sizeof(T2),h,w);
}


template<typename T1, typename T2>
__global__ void _d_bao_blending(T1*d_imgout, size_t out_mem_w, T2*d_img1, T2*d_img2, size_t in_mem_w, int h, int w, float weight1=.5f, float weight2=.5f)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    d_imgout[id_y*out_mem_w + id_x] = T1(d_img1[id_y*in_mem_w+id_x] * weight1 + d_img2[id_y*in_mem_w+id_x] * weight2);
}
template<typename T1, typename T2>
inline void bao_cuda_blending(T1*d_imgout, T2*d_img1, T2*d_img2, int h, int w, float weight1=.5f, float weight2=.5f)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    _d_bao_blending<<<gridSize, blockSize>>>(d_imgout,w,d_img1,d_img2,w,h,w,weight1,weight2);
}
template<typename T1, typename T2>
inline void bao_cuda_blending_pitched(T1*d_imgout, size_t pitch1, T2*d_img1, T2*d_img2, size_t pitch2, int h, int w, float weight1=.5f, float weight2=.5f)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    _d_bao_blending<<<gridSize, blockSize>>>(d_imgout,pitch1/sizeof(T1),d_img1,d_img2,pitch2/sizeof(T2),h,w,weight1,weight2);
}


template<typename T1, typename T2>
__global__ void _d_bao_minus(T1*d_imgout, size_t out_mem_w, T2*d_img1, T2*d_img2, size_t in_mem_w, int h, int w)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    d_imgout[id_y*out_mem_w + id_x] = T1(d_img1[id_y*in_mem_w+id_x] - d_img2[id_y*in_mem_w+id_x]);
}
template<typename T1, typename T2>
inline void bao_cuda_minus(T1*d_imgout, T2*d_img1, T2*d_img2, int h, int w)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    _d_bao_minus<<<gridSize, blockSize>>>(d_imgout,w,d_img1,d_img2,w,h,w);
}
template<typename T1, typename T2>
inline void bao_cuda_minus_pitched(T1*d_imgout, size_t pitch1, T2*d_img1, T2*d_img2, size_t pitch2, int h, int w)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    _d_bao_minus<<<gridSize, blockSize>>>(d_imgout,pitch1/sizeof(T1),d_img1,d_img2,pitch2/sizeof(T2),h,w);
}


template<typename T1, typename T2>
__global__ void _d_bao_multiply_bypixel(T1*d_imgout, size_t out_mem_w, T2*d_img1, T2*d_img2, size_t in_mem_w, int h, int w)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    d_imgout[id_y*out_mem_w + id_x] = T1(d_img1[id_y*in_mem_w+id_x] * d_img2[id_y*in_mem_w+id_x]);
}
template<typename T1, typename T2>
inline void bao_cuda_multiply_bypixel(T1*d_imgout, T2*d_img1, T2*d_img2, int h, int w)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    _d_bao_multiply_bypixel<<<gridSize, blockSize>>>(d_imgout,w,d_img1,d_img2,w,h,w);
}
template<typename T1, typename T2>
inline void bao_cuda_multiply_bypixel_pitched(T1*d_imgout, size_t pitch1, T2*d_img1, T2*d_img2, size_t pitch2, int h, int w)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    _d_bao_multiply_bypixel<<<gridSize, blockSize>>>(d_imgout,pitch1/sizeof(T1),d_img1,d_img2,pitch2/sizeof(T2),h,w);
}


template<typename T1, typename T2>
__global__ void _d_bao_multiply_scalar(T1*d_imgout, size_t out_mem_w, T2*d_imgin, size_t in_mem_w, float scale, int h, int w)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    d_imgout[id_y*out_mem_w + id_x] = T1(d_imgin[id_y*in_mem_w+id_x] * scale);
}
template<typename T1, typename T2>
inline void bao_cuda_multiply_scalar(T1*d_imgout, T2*d_imgin, float scale, int h, int w)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    _d_bao_multiply_scalar<<<gridSize, blockSize>>>(d_imgout,w,d_imgin,w,scale,h,w);
}
template<typename T1, typename T2>
inline void bao_cuda_multiply_scalar_pitched(T1*d_imgout, size_t pitch1, T2*d_imgin, size_t pitch2, float scale, int h, int w)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    _d_bao_multiply_scalar<<<gridSize, blockSize>>>(d_imgout,pitch1/sizeof(T1),d_imgin,pitch2/sizeof(T2),scale,h,w);
}


template<typename T1, typename T2>
__global__ void _d_bao_rgb2gray(T1*d_imgout, size_t out_mem_w, T2*d_imgin, size_t in_mem_w, int h, int w)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    T2 rgba = d_imgin[id_y*in_mem_w+id_x];
    d_imgout[id_y*out_mem_w + id_x] = T1(rgba.x*0.299f + rgba.y*0.587f + rgba.z*0.114f);
}
template<typename T1, typename T2>
inline void bao_cuda_rgb2gray(T1*d_imgout, T2*d_imgin, int h, int w) //T2 has to be rgba format
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    _d_bao_rgb2gray<<<gridSize, blockSize>>>(d_imgout,w,d_imgin,w,h,w);
}
template<typename T1, typename T2>
inline void bao_cuda_rgb2gray_pitched(T1*d_imgout, size_t pitch1, T2*d_imgin, size_t pitch2, int h, int w) //T2 has to be rgba format
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    _d_bao_rgb2gray<<<gridSize, blockSize>>>(d_imgout,pitch1/sizeof(T1),d_imgin,pitch2/sizeof(T2),h,w);
}



template<typename T1, typename T2>
__global__ void _d_bao_multichannels_average(T1*d_imgout, size_t out_mem_w, T2*d_imgin, size_t in_mem_w, int h, int w)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    T2 rgba = d_imgin[id_y*in_mem_w+id_x];
    d_imgout[id_y*out_mem_w + id_x] = T1(rgba.x*0.25f + rgba.y*0.25f + rgba.z*0.25f + rgba.w*0.25f);
}
template<typename T1, typename T2>
inline void bao_cuda_multichannels_average(T1*d_imgout, T2*d_imgin, int h, int w) //T2 has to be rgba format
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    _d_bao_multichannels_average<<<gridSize, blockSize>>>(d_imgout,w,d_imgin,w,h,w);
}
template<typename T1, typename T2>
inline void bao_cuda_multichannels_average_pitched(T1*d_imgout, size_t pitch1, T2*d_imgin, size_t pitch2, int h, int w) //T2 has to be rgba format
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    _d_bao_multichannels_average<<<gridSize, blockSize>>>(d_imgout,pitch1/sizeof(T1),d_imgin,pitch2/sizeof(T2),h,w);
}


template<typename T1, typename T2>
__global__ void _d_bao_deriv_x_5taps(T1*d_imgout, size_t out_mem_w, T2*d_imgin, size_t in_mem_w, int h, int w)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    float xFilter[5]={1.f/12.f, -8.f/12.f, 0, 8.f/12.f, -1.f/12.f};
    T2 val = T2(0);
    for (int dx=-2; dx<=2; dx++)
    {
        int cx = __max(0,__min(w-1,id_x+dx));
        val += d_imgin[id_y*in_mem_w+cx] * xFilter[2+dx]; //TODO: load into shared memory
    }
    d_imgout[id_y*out_mem_w + id_x] = T1(val);
}
template<>
__global__ void _d_bao_deriv_x_5taps(float4*d_imgout, size_t out_mem_w, float4*d_imgin, size_t in_mem_w, int h, int w)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    float xFilter[5]={1.f/12.f, -8.f/12.f, 0, 8.f/12.f, -1.f/12.f};
    float4 val = make_float4(0);
    for (int dx=-2; dx<=2; dx++)
    {
        int cx = __max(0,__min(w-1,id_x+dx));
        val += d_imgin[id_y*in_mem_w+cx] * xFilter[2+dx]; //TODO: load into shared memory
    }
    d_imgout[id_y*out_mem_w + id_x] = float4(val);
}
template<typename T1, typename T2>
__global__ void _d_bao_deriv_x(T1*d_imgout, size_t out_mem_w, T2*d_imgin, size_t in_mem_w, int h, int w)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w-1 || id_y >= h)  //NOTE: w-1 not w!
    {
        if (id_x==w-1 && id_y<h) //make the border to zero
        {
            T2 temp = d_imgin[id_y*in_mem_w+id_x];
            d_imgout[id_y*out_mem_w + id_x] = T1(temp-temp); //zero
        }
        return;
    }
    d_imgout[id_y*out_mem_w + id_x] = T1(d_imgin[id_y*in_mem_w+id_x+1] - d_imgin[id_y*in_mem_w+id_x]);
}
template<typename T1, typename T2>
inline void bao_cuda_deriv_x(T1*d_imgout, T2*d_imgin, int h, int w, bool use_five_points=false)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    if (use_five_points)
    {
        _d_bao_deriv_x_5taps<<<gridSize, blockSize>>>(d_imgout,w,d_imgin,w,h,w);
    }
    else
    {
        _d_bao_deriv_x<<<gridSize, blockSize>>>(d_imgout,w,d_imgin,w,h,w);
    }
}
template<typename T1, typename T2>
inline void bao_cuda_deriv_x_pitched(T1*d_imgout, size_t pitch1, T2*d_imgin, size_t pitch2, int h, int w, bool use_five_points=false)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    if (use_five_points)
    {
        _d_bao_deriv_x_5taps<<<gridSize, blockSize>>>(d_imgout,pitch1/sizeof(T1),d_imgin,pitch2/sizeof(T2),h,w);
    }
    else
    {
        _d_bao_deriv_x<<<gridSize, blockSize>>>(d_imgout,pitch1/sizeof(T1),d_imgin,pitch2/sizeof(T2),h,w);
    }
}


template<typename T1, typename T2>
__global__ void _d_bao_deriv_y_5taps(T1*d_imgout, size_t out_mem_w, T2*d_imgin, size_t in_mem_w, int h, int w)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    float yFilter[5]={1.f/12.f, -8.f/12.f, 0, 8.f/12.f, -1.f/12.f};
    T2 val = T2(0);
    for (int dy=-2; dy<=2; dy++)
    {
        int cy = __max(0,__min(h-1,id_y+dy));
        val += d_imgin[cy*in_mem_w+id_x] * yFilter[2+dy]; //TODO: load into shared memory
    }
    d_imgout[id_y*out_mem_w + id_x] = T1(val);
}
template<>
__global__ void _d_bao_deriv_y_5taps(float4*d_imgout, size_t out_mem_w, float4*d_imgin, size_t in_mem_w, int h, int w)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    float yFilter[5]={1.f/12.f, -8.f/12.f, 0, 8.f/12.f, -1.f/12.f};
    float4 val = make_float4(0);
    for (int dy=-2; dy<=2; dy++)
    {
        int cy = __max(0,__min(h-1,id_y+dy));
        val += d_imgin[cy*in_mem_w+id_x] * yFilter[2+dy]; //TODO: load into shared memory
    }
    d_imgout[id_y*out_mem_w + id_x] = float4(val);
}
template<typename T1, typename T2>
__global__ void _d_bao_deriv_y(T1*d_imgout, size_t out_mem_w, T2*d_imgin, size_t in_mem_w, int h, int w)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h-1) //NOTE: h-1 not h!
    {
        if (id_y==h-1 && id_x<w) //make the border to zero
        {
            T2 temp = d_imgin[id_y*in_mem_w+id_x];
            d_imgout[id_y*out_mem_w + id_x] = T1(temp-temp); //zero
        }
        return;
    }
    d_imgout[id_y*out_mem_w + id_x] = T1(d_imgin[(id_y+1)*in_mem_w+id_x] - d_imgin[id_y*in_mem_w+id_x]);
}
template<typename T1, typename T2>
inline void bao_cuda_deriv_y(T1*d_imgout, T2*d_imgin, int h, int w, bool use_five_points=false)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    if (use_five_points)
    {
        _d_bao_deriv_y_5taps<<<gridSize, blockSize>>>(d_imgout,w,d_imgin,w,h,w);
    }
    else
    {
        _d_bao_deriv_y<<<gridSize, blockSize>>>(d_imgout,w,d_imgin,w,h,w);
    }
}
template<typename T1, typename T2>
inline void bao_cuda_deriv_y_pitched(T1*d_imgout, size_t pitch1, T2*d_imgin, size_t pitch2, int h, int w, bool use_five_points=false)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    if (use_five_points)
    {
        _d_bao_deriv_y_5taps<<<gridSize, blockSize>>>(d_imgout,pitch1/sizeof(T1),d_imgin,pitch2/sizeof(T2),h,w);
    }
    else
    {
        _d_bao_deriv_y<<<gridSize, blockSize>>>(d_imgout,pitch1/sizeof(T1),d_imgin,pitch2/sizeof(T2),h,w);
    }
}


__device__ float _bao_cuda_basic_d_floatval; //device var
template<typename T>
__global__ void _d_bao_vec_inner_product(T*d_img1, T*d_img2, size_t in_mem_w, int h, int w)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x == 0 && id_y == 0) //TODO: only one thread works, too slow! optimize it!
    {
        float res = 0;
        for (int i=0;i<h;i++) for(int j=0;j<w;j++) res += float(d_img1[i*in_mem_w+j] * d_img2[i*in_mem_w+j]);
        _bao_cuda_basic_d_floatval = res; //NOTE: use a global scoped __device__ variable!
    }
}
template<typename T>
inline float bao_cuda_vec_inner_product(T* d_img1, T* d_img2, int h, int w)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    _d_bao_vec_inner_product<<<gridSize, blockSize>>>(d_img1,d_img2,w,h,w);
    float h_res;
    checkCudaErrors(cudaMemcpyFromSymbol(&h_res,_bao_cuda_basic_d_floatval,sizeof(float)));
    return h_res;
}
template<typename T>
inline float bao_cuda_vec_inner_product_pitched(T* d_img1, T* d_img2, size_t pitch, int h, int w)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    _d_bao_vec_inner_product<<<gridSize, blockSize>>>(d_img1,d_img2,pitch/sizeof(T),h,w);
    float h_res;
    checkCudaErrors(cudaMemcpyFromSymbol(&h_res,_bao_cuda_basic_d_floatval,sizeof(float)));
    return h_res;
}


template<typename T1, typename T2>
__global__ void _d_bao_gauss_filter(T1*d_imgout, size_t out_mem_w, T2*d_imgin, size_t in_mem_w, int h, int w, float sigma, float radius)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    
    sigma = sigma*sigma*2;
    T2 val = T2(0);
    float sum = 0;
    for (int dy=-radius; dy<=radius; dy++) for (int dx=-radius; dx<=radius; dx++) //TODO: slow!
    {
        int cy = __max(0,__min(h-1,id_y+dy));
        int cx = __max(0,__min(w-1,id_x+dx));
        float weight = __expf(-(float)(dy*dy+dx*dx)/sigma);
        val += d_imgin[cy*in_mem_w+cx] * weight;
        sum += weight;
    }
    d_imgout[id_y*out_mem_w + id_x] = T1(val/sum);
}
template<>
__global__ void _d_bao_gauss_filter(float4*d_imgout, size_t out_mem_w, float4*d_imgin, size_t in_mem_w, int h, int w, float sigma, float radius)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    
    sigma = sigma*sigma*2;
    float4 val = make_float4(0);
    float sum = 0;
    for (int dy=-radius; dy<=radius; dy++) for (int dx=-radius; dx<=radius; dx++) //TODO: slow!
    {
        int cy = __max(0,__min(h-1,id_y+dy));
        int cx = __max(0,__min(w-1,id_x+dx));
        float weight = __expf(-(float)(dy*dy+dx*dx)/sigma);
        val += d_imgin[cy*in_mem_w+cx] * weight;
        sum += weight;
    }
    d_imgout[id_y*out_mem_w + id_x] = (val/sum);
}
template<>
__global__ void _d_bao_gauss_filter(uchar4*d_imgout, size_t out_mem_w, uchar4*d_imgin, size_t in_mem_w, int h, int w, float sigma, float radius)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    
    sigma = sigma*sigma*2;
    float4 val = make_float4(0);
    float sum = 0;
    uchar4 temp_pix;
    for (int dy=-radius; dy<=radius; dy++) for (int dx=-radius; dx<=radius; dx++) //TODO: slow!
    {
        int cy = __max(0,__min(h-1,id_y+dy));
        int cx = __max(0,__min(w-1,id_x+dx));
        float weight = __expf(-(float)(dy*dy+dx*dx)/sigma);
        temp_pix = d_imgin[cy*in_mem_w+cx];
        val.x += temp_pix.x * weight;
        val.y += temp_pix.y * weight;
        val.z += temp_pix.z * weight;
        val.w += temp_pix.w * weight;
        sum += weight;
    }
    val /= sum;
    uchar4 res_u4;
    res_u4.x = val.x;
    res_u4.y = val.y;
    res_u4.z = val.z;
    res_u4.w = val.w;
    d_imgout[id_y*out_mem_w + id_x] = res_u4;
}
template<typename T>
inline void bao_cuda_gauss_filter(T* d_imgout, T* d_imgin, int h, int w, float sigma, int radius)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    _d_bao_gauss_filter<<<gridSize, blockSize>>>(d_imgout,w,d_imgin,w,h,w,sigma,radius);
}
template<typename T>
inline void bao_cuda_gauss_filter_pitched(T* d_imgout, T* d_imgin, size_t pitch, int h, int w, float sigma, int radius)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    _d_bao_gauss_filter<<<gridSize, blockSize>>>(d_imgout,pitch/sizeof(T),d_imgin,pitch/sizeof(T),h,w,sigma,radius);
}


template<typename T1, typename T2>
__global__ void _d_bao_bilinear_resize(T1*d_imgout, size_t out_mem_w, int outH, int outW, T2*d_imgin, size_t in_mem_w, int h, int w, float ratio)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= outW || id_y >= outH) return;
    float div_scale=1.f/ratio;
    float fx=(float)(id_x+1)*div_scale-1;
    float fy=(float)(id_y+1)*div_scale-1;

    int xx,yy;
    xx=fx;
    yy=fy;
    float dx,dy,s;
    dx=__max(__min(fx-xx,1),0);
    dy=__max(__min(fy-yy,1),0);

    T2 res = T2(0);
    for(int m=0;m<=1;m++) for(int n=0;n<=1;n++)
    {
        int u=__max(0,__min(w-1,xx+m));
        int v=__max(0,__min(h-1,yy+n));
        s=fabs(1-m-dx)*fabs(1-n-dy);
        res += (d_imgin[v*in_mem_w+u]*s);
    }
    d_imgout[id_y*out_mem_w + id_x] = T1(res);
}
template<>
__global__ void _d_bao_bilinear_resize(float2*d_imgout, size_t out_mem_w, int outH, int outW, float2*d_imgin, size_t in_mem_w, int h, int w, float ratio)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= outW || id_y >= outH) return;
    float div_scale=1.f/ratio;
    float fx=(float)(id_x+1)*div_scale-1;
    float fy=(float)(id_y+1)*div_scale-1;

    int xx,yy;
    xx=fx;
    yy=fy;
    float dx,dy,s;
    dx=__max(__min(fx-xx,1),0);
    dy=__max(__min(fy-yy,1),0);

    float2 res = make_float2(0);
    for(int m=0;m<=1;m++) for(int n=0;n<=1;n++)
    {
        int u=__max(0,__min(w-1,xx+m));
        int v=__max(0,__min(h-1,yy+n));
        s=fabs(1-m-dx)*fabs(1-n-dy);
        res += (d_imgin[v*in_mem_w+u]*s);
    }
    d_imgout[id_y*out_mem_w + id_x] = float2(res);
}
template<>
__global__ void _d_bao_bilinear_resize(float4*d_imgout, size_t out_mem_w, int outH, int outW, float4*d_imgin, size_t in_mem_w, int h, int w, float ratio)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= outW || id_y >= outH) return;
    float div_scale=1.f/ratio;
    float fx=(float)(id_x+1)*div_scale-1;
    float fy=(float)(id_y+1)*div_scale-1;

    int xx,yy;
    xx=fx;
    yy=fy;
    float dx,dy,s;
    dx=__max(__min(fx-xx,1),0);
    dy=__max(__min(fy-yy,1),0);

    float4 res = make_float4(0);
    for(int m=0;m<=1;m++) for(int n=0;n<=1;n++)
    {
        int u=__max(0,__min(w-1,xx+m));
        int v=__max(0,__min(h-1,yy+n));
        s=fabs(1-m-dx)*fabs(1-n-dy);
        res += (d_imgin[v*in_mem_w+u]*s);
    }
    d_imgout[id_y*out_mem_w + id_x] = float4(res);
}
template<>
__global__ void _d_bao_bilinear_resize(uchar4*d_imgout, size_t out_mem_w, int outH, int outW, uchar4*d_imgin, size_t in_mem_w, int h, int w, float ratio)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= outW || id_y >= outH) return;
    float div_scale=1.f/ratio;
    float fx=(float)(id_x+1)*div_scale-1;
    float fy=(float)(id_y+1)*div_scale-1;

    int xx,yy;
    xx=fx;
    yy=fy;
    float dx,dy,s;
    dx=__max(__min(fx-xx,1),0);
    dy=__max(__min(fy-yy,1),0);

    float4 res = make_float4(0);
    uchar4 temp_pix;
    for(int m=0;m<=1;m++) for(int n=0;n<=1;n++)
    {
        int u=__max(0,__min(w-1,xx+m));
        int v=__max(0,__min(h-1,yy+n));
        s=fabs(1-m-dx)*fabs(1-n-dy);
        temp_pix = d_imgin[v*in_mem_w+u];
        res.x += (temp_pix.x*s);
        res.y += (temp_pix.y*s);
        res.z += (temp_pix.z*s);
        res.w += (temp_pix.w*s);
    }
    uchar4 res_u4;
    res_u4.x = res.x;
    res_u4.y = res.y;
    res_u4.z = res.z;
    res_u4.w = res.w;
    d_imgout[id_y*out_mem_w + id_x] = res_u4;
}
template<typename T>
inline void bao_cuda_bilinear_resize(T* d_imgout, int outH, int outW, T* d_imgin, int h, int w, float ratio)
{
    dim3 gridSize(bao_div_ceil(outW,BLOCK_DIM_X),bao_div_ceil(outH,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    _d_bao_bilinear_resize<<<gridSize, blockSize>>>(d_imgout,outW,outH,outW,d_imgin,w,h,w,ratio);
}
template<typename T>
inline void bao_cuda_bilinear_resize_pitched(T* d_imgout, size_t pitch1, int outH, int outW, T* d_imgin, size_t pitch2, int h, int w, float ratio)
{
    dim3 gridSize(bao_div_ceil(outW,BLOCK_DIM_X),bao_div_ceil(outH,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    _d_bao_bilinear_resize<<<gridSize, blockSize>>>(d_imgout,pitch1/sizeof(T),outH,outW,d_imgin,pitch2/sizeof(T),h,w,ratio);
}



template<typename T>
inline void bao_cuda_construct_gauss_pyramid(T** pPyr, T* d_img, T**pPyrTemp, int nLevels, int* arrH, int* arrW, float ratio=0.5f)
{
    if (nLevels <= 0) return;
    bao_cuda_copy_d2d(pPyr[0],d_img,arrH[0],arrW[0]);
    float baseSigma=(1/ratio-1);
    int n=log(0.25)/log(ratio);
    float nSigma=baseSigma*n;
    for (int i=1; i<nLevels; i++)
    {
        if(i<=n)
        {
            float sigma=baseSigma*i;
            bao_cuda_gauss_filter(pPyrTemp[0],pPyr[0],arrH[0],arrW[0],sigma,sigma*3);
            bao_cuda_bilinear_resize(pPyr[i],arrH[i],arrW[i],pPyrTemp[0],arrH[0],arrW[0],pow(ratio,i));
        }
        else
        {
            bao_cuda_gauss_filter(pPyrTemp[i-n],pPyr[i-n],arrH[i-n],arrW[i-n],nSigma,nSigma*3);
            bao_cuda_bilinear_resize(pPyr[i],arrH[i],arrW[i],pPyrTemp[i-n],arrH[i-n],arrW[i-n],(float)pow(ratio,i)*arrW[0]/arrW[i-n]);
        }
    }
}
template<typename T>
inline void bao_cuda_construct_gauss_pyramid_pitched(T** pPyr, T* d_img, T**pPyrTemp, int nLevels, int* arrH, int* arrW, size_t* arrPitch, float ratio=0.5f)
{
    if (nLevels <= 0) return;
    if (pPyr[0] != d_img) bao_cuda_copy_d2d_pitched(pPyr[0],arrPitch[0],d_img,arrPitch[0],arrH[0],arrW[0]);
    float baseSigma=(1/ratio-1);
    int n=log(0.25)/log(ratio);
    float nSigma=baseSigma*n;
    for (int i=1; i<nLevels; i++)
    {
        if(i<=n)
        {
            float sigma=baseSigma*i;
            bao_cuda_gauss_filter_pitched(pPyrTemp[0],pPyr[0],arrPitch[0],arrH[0],arrW[0],sigma,sigma*3);
            bao_cuda_bilinear_resize_pitched(pPyr[i],arrPitch[i],arrH[i],arrW[i],pPyrTemp[0],arrPitch[0],arrH[0],arrW[0],pow(ratio,i));
        }
        else
        {
            bao_cuda_gauss_filter_pitched(pPyrTemp[i-n],pPyr[i-n],arrPitch[i-n],arrH[i-n],arrW[i-n],nSigma,nSigma*3);
            bao_cuda_bilinear_resize_pitched(pPyr[i],arrPitch[i],arrH[i],arrW[i],pPyrTemp[i-n],arrPitch[i-n],arrH[i-n],arrW[i-n],(float)pow(ratio,i)*arrW[0]/arrW[i-n]);
        }
    }
}


template<typename T>
__device__ inline T _d_bao_bicubic_interp_pixel(T* d_img, size_t in_mem_w, int h, int w, float x, float y)
{
    T res = 0;
    int xx,yy;
    xx=x;
    yy=y;
    float dx,dy;
    dx=__max(__min(x-xx,1),0);
    dy=__max(__min(y-yy,1),0);

    #pragma unroll
    for(int m=0;m<=1;m++) for(int n=0;n<=1;n++)
    {
        int u=__max(0,__min(w-1,xx+m));
        int v=__max(0,__min(h-1,yy+n));
        float s=fabs(1-m-dx)*fabs(1-n-dy);
        res += (d_img[v*in_mem_w+u]*s);
    }
    return res;
}
template<>
__device__ inline float4 _d_bao_bicubic_interp_pixel(float4* d_img, size_t in_mem_w, int h, int w, float x, float y)
{
    float4 res = make_float4(0);
    int xx,yy;
    xx=x;
    yy=y;
    float dx,dy;
    dx=__max(__min(x-xx,1),0);
    dy=__max(__min(y-yy,1),0);

    #pragma unroll
    for(int m=0;m<=1;m++) for(int n=0;n<=1;n++)
    {
        int u=__max(0,__min(w-1,xx+m));
        int v=__max(0,__min(h-1,yy+n));
        float s=fabs(1-m-dx)*fabs(1-n-dy);
        res += (d_img[v*in_mem_w+u]*s);
    }
    return res;
}
template<typename T1>
__global__ void _d_bao_warping_by_flow(T1* pWarpedImg1, T1* pImg1, T1* pImg2, size_t out_mem_w, float* U, float* V, size_t in_mem_w, int h, int w)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    float xx, yy;
    xx = id_x + U[id_y*in_mem_w+id_x];
    yy = id_y + V[id_y*in_mem_w+id_x];
    if (xx<0 || xx>w-1 || yy<0 || yy>h-1)
    {
        pWarpedImg1[id_y*out_mem_w + id_x] = pImg1[id_y*out_mem_w + id_x]; //fill invalid pixels with img1, to make I_t = 0
    }
    else
    {
        pWarpedImg1[id_y*out_mem_w + id_x] = _d_bao_bicubic_interp_pixel<T1>(pImg2,out_mem_w,h,w,xx,yy);
    }
}
template<typename T1>
inline void bao_cuda_warping_by_flow(T1* pWarpedImg1, T1* pImg1, T1* pImg2, float* U, float* V, int h, int w)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    _d_bao_warping_by_flow<<<gridSize, blockSize>>>(pWarpedImg1,pImg1,pImg2,w,U,V,w,h,w);
}
template<typename T1>
inline void bao_cuda_warping_by_flow_pitched(T1* pWarpedImg1, T1* pImg1, T1* pImg2, size_t pitch1, float* U, float* V, size_t pitch2, int h, int w)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    _d_bao_warping_by_flow<<<gridSize, blockSize>>>(pWarpedImg1,pImg1,pImg2,pitch1/sizeof(T1),U,V,pitch2/sizeof(float),h,w);
}


__device__ int _d_bao_colorwheel_ncols = 0;
__device__ int _d_bao_g_colorwheel[60][3];
__device__ void _d_bao_colorwheel_setcols(int r, int g, int b, int k)
{
    _d_bao_g_colorwheel[k][0] = r;
    _d_bao_g_colorwheel[k][1] = g;
    _d_bao_g_colorwheel[k][2] = b;
}
__global__ void _d_bao_colorwheel_build()
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x==0 && id_y==0)
    {
        int RY = 15;
        int YG = 6;
        int GC = 4;
        int CB = 11;
        int BM = 13;
        int MR = 6;
        _d_bao_colorwheel_ncols = RY + YG + GC + CB + BM + MR;
        int i;
        int k = 0;
        for (i = 0; i < RY; i++) _d_bao_colorwheel_setcols(255,	   255*i/RY,	 0,	       k++);
        for (i = 0; i < YG; i++) _d_bao_colorwheel_setcols(255-255*i/YG, 255,		 0,	       k++);
        for (i = 0; i < GC; i++) _d_bao_colorwheel_setcols(0,		   255,		 255*i/GC,     k++);
        for (i = 0; i < CB; i++) _d_bao_colorwheel_setcols(0,		   255-255*i/CB, 255,	       k++);
        for (i = 0; i < BM; i++) _d_bao_colorwheel_setcols(255*i/BM,	   0,		 255,	       k++);
        for (i = 0; i < MR; i++) _d_bao_colorwheel_setcols(255,	   0,		 255-255*i/MR, k++);
    }
}
__device__ uchar4 _d_bao_compute_flow_color(float fx, float fy)
{
    float rad = __fsqrt_rn(fx * fx + fy * fy);
    float a = atan2(-fy, -fx) / 3.14159f;
    float fk = (a + 1.0f) / 2.0f * (_d_bao_colorwheel_ncols-1);
    int k0 = (int)fk;
    int k1 = (k0 + 1) % _d_bao_colorwheel_ncols;
    float f = fk - k0;
    //f = 0; // uncomment to see original color wheel
    unsigned char pixval[3];
    for (int b = 0; b < 3; b++) 
    {
	    float col0 = _d_bao_g_colorwheel[k0][b] / 255.0f;
	    float col1 = _d_bao_g_colorwheel[k1][b] / 255.0f;
	    float col = (1 - f) * col0 + f * col1;
	    if (rad <= 1)
	        col = 1 - rad * (1 - col); // increase saturation with radius
	    else
	        col *= .75; // out of range
	    pixval[2 - b] = (int)(255.0 * col);
    }
    uchar4 pix;
    pix.x = pixval[2];
    pix.y = pixval[1];
    pix.z = pixval[0];
    return pix;
}
__global__ void _d_bao_convert_flow_to_colorshow(uchar4* rgbflow, size_t out_mem_w, float* flow_x, float* flow_y, size_t in_mem_w, int h, int w, float max_rad)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    float fx = flow_x[id_y*in_mem_w+id_x];
    float fy = flow_y[id_y*in_mem_w+id_x];
    uchar4 rgba = make_uchar4(0,0,0,0);
    if (abs(fx)<999999 && abs(fy)<999999) 
    {
        rgba = _d_bao_compute_flow_color(fx/max_rad, fy/max_rad);
    }
    rgbflow[id_y*out_mem_w + id_x] = rgba;
}
__global__ void _d_bao_convert_flow_to_colorshow(uchar4* rgbflow, size_t out_mem_w, float2* flow_vec, size_t in_mem_w, int h, int w, float max_rad)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    float2 val = flow_vec[id_y*in_mem_w+id_x];
    float fx = val.x;
    float fy = val.y;
    uchar4 rgba = make_uchar4(0,0,0,0);
    if (abs(fx)<999999 && abs(fy)<999999) 
    {
        rgba = _d_bao_compute_flow_color(fx/max_rad, fy/max_rad);
    }
    rgbflow[id_y*out_mem_w + id_x] = rgba;
}
void bao_cuda_convert_flow_to_colorshow(uchar4* rgbflow, float* flow_x, float* flow_y, int h, int w, float max_disp_x=100, float max_disp_y=100)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    float max_disp_rad = sqrt(max_disp_x*max_disp_x + max_disp_y*max_disp_y);
    _d_bao_colorwheel_build<<<1, 1>>>();
    _d_bao_convert_flow_to_colorshow<<<gridSize, blockSize>>>(rgbflow, w, flow_x, flow_y, w, h, w, max_disp_rad);
}
void bao_cuda_convert_flow_to_colorshow(uchar4* rgbflow, float2* flow_vec, int h, int w, float max_disp_x=100, float max_disp_y=100)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    float max_disp_rad = sqrt(max_disp_x*max_disp_x + max_disp_y*max_disp_y);
    _d_bao_colorwheel_build<<<1, 1>>>();
    _d_bao_convert_flow_to_colorshow<<<gridSize, blockSize>>>(rgbflow, w, flow_vec, w, h, w, max_disp_rad);
}




#endif