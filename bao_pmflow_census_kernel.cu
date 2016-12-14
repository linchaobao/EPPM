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


#include "bao_basic_cuda.h"
#include "defs.h"  //for parameters
#include "bicubicTexture_kernel.cuh"

texture<uchar4, 2, cudaReadModeNormalizedFloat> rgbaImg1Tex; //in texture memory
texture<uchar4, 2, cudaReadModeNormalizedFloat> rgbaImg2Tex; //in texture memory


#define BLOCK_DIM_X  16
#define BLOCK_DIM_Y  16

__device__ unsigned char _d_is_larger(float4 a, float4 b)
{
    if (0.3f*a.x + 0.6f*a.y + 0.1f*a.z > 0.3f*b.x + 0.6f*b.y + 0.1f*b.z) return 0x1;
    else return 0x0;
}

__global__ void d_census_transform3x3(unsigned char* d_census1, unsigned char* d_census2, int w, int h, size_t census_mem_w)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    
    float4 centerPix = tex2D(rgbaImg1Tex, id_x, id_y);
    float4 surroundPix[8];
    surroundPix[0] = tex2D(rgbaImg1Tex, id_x-1, id_y-1);
    surroundPix[1] = tex2D(rgbaImg1Tex, id_x, id_y-1);
    surroundPix[2] = tex2D(rgbaImg1Tex, id_x+1, id_y-1);
    surroundPix[3] = tex2D(rgbaImg1Tex, id_x-1, id_y);
    surroundPix[4] = tex2D(rgbaImg1Tex, id_x+1, id_y);
    surroundPix[5] = tex2D(rgbaImg1Tex, id_x-1, id_y+1);
    surroundPix[6] = tex2D(rgbaImg1Tex, id_x, id_y+1);
    surroundPix[7] = tex2D(rgbaImg1Tex, id_x+1, id_y+1);
    unsigned char censusRes = _d_is_larger(surroundPix[0],centerPix);
    censusRes += (_d_is_larger(surroundPix[1],centerPix) << 1);
    censusRes += (_d_is_larger(surroundPix[2],centerPix) << 2);
    censusRes += (_d_is_larger(surroundPix[3],centerPix) << 3);
    censusRes += (_d_is_larger(surroundPix[4],centerPix) << 4);
    censusRes += (_d_is_larger(surroundPix[5],centerPix) << 5);
    censusRes += (_d_is_larger(surroundPix[6],centerPix) << 6);
    censusRes += (_d_is_larger(surroundPix[7],centerPix) << 7);
    d_census1[id_y*census_mem_w + id_x] = censusRes;

    //the second image
    centerPix = tex2D(rgbaImg2Tex, id_x, id_y);
    surroundPix[0] = tex2D(rgbaImg2Tex, id_x-1, id_y-1);
    surroundPix[1] = tex2D(rgbaImg2Tex, id_x, id_y-1);
    surroundPix[2] = tex2D(rgbaImg2Tex, id_x+1, id_y-1);
    surroundPix[3] = tex2D(rgbaImg2Tex, id_x-1, id_y);
    surroundPix[4] = tex2D(rgbaImg2Tex, id_x+1, id_y);
    surroundPix[5] = tex2D(rgbaImg2Tex, id_x-1, id_y+1);
    surroundPix[6] = tex2D(rgbaImg2Tex, id_x, id_y+1);
    surroundPix[7] = tex2D(rgbaImg2Tex, id_x+1, id_y+1);
    censusRes = _d_is_larger(surroundPix[0],centerPix);
    censusRes += (_d_is_larger(surroundPix[1],centerPix) << 1);
    censusRes += (_d_is_larger(surroundPix[2],centerPix) << 2);
    censusRes += (_d_is_larger(surroundPix[3],centerPix) << 3);
    censusRes += (_d_is_larger(surroundPix[4],centerPix) << 4);
    censusRes += (_d_is_larger(surroundPix[5],centerPix) << 5);
    censusRes += (_d_is_larger(surroundPix[6],centerPix) << 6);
    censusRes += (_d_is_larger(surroundPix[7],centerPix) << 7);
    d_census2[id_y*census_mem_w + id_x] = censusRes;
}


extern "C" 
void baoCudaCensusTransform(unsigned char* d_census1, unsigned char* d_census2, uchar4* d_img1, uchar4* d_img2, int w, int h, size_t img_pitch, size_t census_pitch)
{
    //bind imgs
    cudaChannelFormatDesc desc_img = cudaCreateChannelDesc<uchar4>();
    checkCudaErrors(cudaBindTexture2D(0, rgbaImg1Tex, d_img1, desc_img, w, h, img_pitch));
    checkCudaErrors(cudaBindTexture2D(0, rgbaImg2Tex, d_img2, desc_img, w, h, img_pitch));
//     getLastCudaError("Census Bind Texture FAILED");

    //compute census transform
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);

    size_t census_mem_w = census_pitch/sizeof(unsigned char);
//     bao_timer_gpu timer;
//     timer.start();
    d_census_transform3x3<<<gridSize, blockSize>>>(d_census1,d_census2,w,h,census_mem_w); //0.075ms
//     timer.time_display("Pre: Census Transform");
//     getLastCudaError("Census Transform FAILED");
}


__global__ void d_census_transform3x3_bicubic(unsigned char* d_census1, unsigned char* d_census2, int w, int h, size_t census_mem_w, float up_factor)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    
    float4 centerPix = tex2DBicubic<uchar4,float4>(rgbaImg1Tex, id_x*up_factor, id_y*up_factor);
    float4 surroundPix[8];
    surroundPix[0] = tex2DBicubic<uchar4,float4>(rgbaImg1Tex, (id_x-1)*up_factor, (id_y-1)*up_factor);
    surroundPix[1] = tex2DBicubic<uchar4,float4>(rgbaImg1Tex, (id_x)*up_factor, (id_y-1)*up_factor);
    surroundPix[2] = tex2DBicubic<uchar4,float4>(rgbaImg1Tex, (id_x+1)*up_factor, (id_y-1)*up_factor);
    surroundPix[3] = tex2DBicubic<uchar4,float4>(rgbaImg1Tex, (id_x-1)*up_factor, (id_y)*up_factor);
    surroundPix[4] = tex2DBicubic<uchar4,float4>(rgbaImg1Tex, (id_x+1)*up_factor, (id_y)*up_factor);
    surroundPix[5] = tex2DBicubic<uchar4,float4>(rgbaImg1Tex, (id_x-1)*up_factor, (id_y+1)*up_factor);
    surroundPix[6] = tex2DBicubic<uchar4,float4>(rgbaImg1Tex, (id_x)*up_factor, (id_y+1)*up_factor);
    surroundPix[7] = tex2DBicubic<uchar4,float4>(rgbaImg1Tex, (id_x+1)*up_factor, (id_y+1)*up_factor);
    unsigned char censusRes = _d_is_larger(surroundPix[0],centerPix);
    censusRes += (_d_is_larger(surroundPix[1],centerPix) << 1);
    censusRes += (_d_is_larger(surroundPix[2],centerPix) << 2);
    censusRes += (_d_is_larger(surroundPix[3],centerPix) << 3);
    censusRes += (_d_is_larger(surroundPix[4],centerPix) << 4);
    censusRes += (_d_is_larger(surroundPix[5],centerPix) << 5);
    censusRes += (_d_is_larger(surroundPix[6],centerPix) << 6);
    censusRes += (_d_is_larger(surroundPix[7],centerPix) << 7);
    d_census1[id_y*census_mem_w + id_x] = censusRes;

    //the second image
    centerPix = tex2DBicubic<uchar4,float4>(rgbaImg2Tex, id_x*up_factor, id_y*up_factor);
    surroundPix[0] = tex2DBicubic<uchar4,float4>(rgbaImg2Tex, (id_x-1)*up_factor, (id_y-1)*up_factor);
    surroundPix[1] = tex2DBicubic<uchar4,float4>(rgbaImg2Tex, (id_x)*up_factor, (id_y-1)*up_factor);
    surroundPix[2] = tex2DBicubic<uchar4,float4>(rgbaImg2Tex, (id_x+1)*up_factor, (id_y-1)*up_factor);
    surroundPix[3] = tex2DBicubic<uchar4,float4>(rgbaImg2Tex, (id_x-1)*up_factor, (id_y)*up_factor);
    surroundPix[4] = tex2DBicubic<uchar4,float4>(rgbaImg2Tex, (id_x+1)*up_factor, (id_y)*up_factor);
    surroundPix[5] = tex2DBicubic<uchar4,float4>(rgbaImg2Tex, (id_x-1)*up_factor, (id_y+1)*up_factor);
    surroundPix[6] = tex2DBicubic<uchar4,float4>(rgbaImg2Tex, (id_x)*up_factor, (id_y+1)*up_factor);
    surroundPix[7] = tex2DBicubic<uchar4,float4>(rgbaImg2Tex, (id_x+1)*up_factor, (id_y+1)*up_factor);
    censusRes = _d_is_larger(surroundPix[0],centerPix);
    censusRes += (_d_is_larger(surroundPix[1],centerPix) << 1);
    censusRes += (_d_is_larger(surroundPix[2],centerPix) << 2);
    censusRes += (_d_is_larger(surroundPix[3],centerPix) << 3);
    censusRes += (_d_is_larger(surroundPix[4],centerPix) << 4);
    censusRes += (_d_is_larger(surroundPix[5],centerPix) << 5);
    censusRes += (_d_is_larger(surroundPix[6],centerPix) << 6);
    censusRes += (_d_is_larger(surroundPix[7],centerPix) << 7);
    d_census2[id_y*census_mem_w + id_x] = censusRes;
}

extern "C" 
void baoCudaCensusTransform_Bicubic(unsigned char* d_census1, unsigned char* d_census2, int w_up, int h_up, size_t census_pitch, uchar4* d_img1, uchar4* d_img2, int w, int h, size_t img_pitch)
{
    //bind imgs
    cudaChannelFormatDesc desc_img = cudaCreateChannelDesc<uchar4>();
    checkCudaErrors(cudaBindTexture2D(0, rgbaImg1Tex, d_img1, desc_img, w, h, img_pitch));
    checkCudaErrors(cudaBindTexture2D(0, rgbaImg2Tex, d_img2, desc_img, w, h, img_pitch));

    //compute census transform
    dim3 gridSize(bao_div_ceil(w_up,BLOCK_DIM_X),bao_div_ceil(h_up,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);

    float up_factor = float(w) / float(w_up); //e.g., 0.5f
    size_t census_mem_w = census_pitch/sizeof(unsigned char);
    bao_timer_gpu timer;
    timer.start();
    d_census_transform3x3_bicubic<<<gridSize, blockSize>>>(d_census1,d_census2,w_up,h_up,census_mem_w,up_factor); //0.075ms
    timer.time_display("Pre: Census Transform");
    getLastCudaError("Census Transform FAILED");
}
