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
#include "bao_basic_cuda.cuh"
#include "defs.h"  //for parameters
#include <float.h>

#define  CENSUS_MAX_DIFF  8

__constant__ float cWmfGaussian[WMF_RADIUS+1];   //gaussian array in device side
__constant__ float cSubpixGaussian[SUBPIX_PATCH_R+1];   //gaussian array in device side
__constant__ float cCensusNegativeGaussian[CENSUS_MAX_DIFF+1];  //for census transform distance
texture<uchar4, 2, cudaReadModeNormalizedFloat> rgbaImg1Tex; //in texture memory
texture<uchar4, 2, cudaReadModeNormalizedFloat> rgbaImg2Tex; //in texture memory
texture<unsigned char, 2, cudaReadModeElementType> census1Tex; //in texture memory
texture<unsigned char, 2, cudaReadModeElementType> census2Tex; //in texture memory


#define BLOCK_DIM_X  16
#define BLOCK_DIM_Y  16

#define INVALID_LOCATION         -10000


//////////////////////////////////////////////////////////////////////////
// left-right consistency check
#define DIFF_THRESH  0

__global__ void d_left_right_check(short2* d_disp_vec, float* d_cost, short2* d_disp_vec2, float* d_cost2, int w, int h, size_t cost_mem_w, size_t disp_mem_w)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    short2 cur_pix_pos = make_short2(id_x,id_y);
    short2 disp_val = d_disp_vec[id_y*disp_mem_w + id_x];
    if (disp_val.y < 0 || disp_val.y >= h || disp_val.x < 0 || disp_val.x >= w)
    {
        d_disp_vec[id_y*disp_mem_w + id_x] = make_short2(INVALID_LOCATION,INVALID_LOCATION); //set invalid disparity for later fixing
        d_cost[id_y*cost_mem_w + id_x] = FLT_MAX;
    }
    else
    {
        short2 disp_val2 = d_disp_vec2[disp_val.y*disp_mem_w + disp_val.x];
        if (abs(disp_val2.x - cur_pix_pos.x)>DIFF_THRESH || abs(disp_val2.y - cur_pix_pos.y)>DIFF_THRESH)
        {
            d_disp_vec[id_y*disp_mem_w + id_x] = make_short2(INVALID_LOCATION,INVALID_LOCATION); //set invalid disparity for later fixing
            d_cost[id_y*cost_mem_w + id_x] = FLT_MAX;
//             d_disp_vec2[disp_val.y*disp_mem_w + disp_val.x] = make_short2(INVALID_LOCATION,INVALID_LOCATION); //set invalid disparity for later fixing
//             d_cost2[disp_val.y*cost_mem_w + disp_val.x] = FLT_MAX;
        }
    }
}

extern "C" 
void baoCudaLeftRightCheck(short2* d_disp_vec, float* d_cost, short2* d_disp_vec2, float* d_cost2, int w, int h, size_t cost_pitch, size_t disp_pitch)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    
    size_t cost_mem_w = cost_pitch/sizeof(float);
    size_t disp_mem_w = disp_pitch/sizeof(short2);

//     bao_timer_gpu timer;
//     timer.start();
    d_left_right_check<<<gridSize,blockSize>>>(d_disp_vec,d_cost,d_disp_vec2,d_cost2,w,h,cost_mem_w,disp_mem_w); //0.02ms
    d_left_right_check<<<gridSize,blockSize>>>(d_disp_vec2,d_cost2,d_disp_vec,d_cost,w,h,cost_mem_w,disp_mem_w); //0.02ms
//     timer.time_display("Refine: Left-right Check");
}


#define DIFF_THRESH_2  50
__global__ void d_left_right_check_buffered(short2* d_out_disp_vec, float* d_out_cost, short2* d_disp_vec, float* d_cost, short2* d_disp_vec2, float* d_cost2, int w, int h, size_t cost_mem_w, size_t disp_mem_w)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    short2 cur_pix_pos = make_short2(id_x,id_y);
    short2 disp_val = d_disp_vec[id_y*disp_mem_w + id_x];
    if (disp_val.y < 0 || disp_val.y >= h || disp_val.x < 0 || disp_val.x >= w)
    {
        d_out_disp_vec[id_y*disp_mem_w + id_x] = make_short2(INVALID_LOCATION,INVALID_LOCATION); //set invalid disparity for later fixing
        d_out_cost[id_y*cost_mem_w + id_x] = FLT_MAX;
    }
    else
    {
        short2 disp_val2 = d_disp_vec2[disp_val.y*disp_mem_w + disp_val.x];
        if (abs(disp_val2.x - cur_pix_pos.x)>DIFF_THRESH_2 || abs(disp_val2.y - cur_pix_pos.y)>DIFF_THRESH_2)
        {
            d_out_disp_vec[id_y*disp_mem_w + id_x] = make_short2(INVALID_LOCATION,INVALID_LOCATION); //set invalid disparity for later fixing
            d_out_cost[id_y*cost_mem_w + id_x] = FLT_MAX;
        }
        else
        {
            d_out_disp_vec[id_y*disp_mem_w + id_x] = disp_val; //set invalid disparity for later fixing
            d_out_cost[id_y*cost_mem_w + id_x] = d_cost[id_y*cost_mem_w + id_x];
        }
    }
}

extern "C" 
void baoCudaLeftRightCheck_Buffered(short2* d_disp_vec, float* d_cost, short2* d_disp_vec2, float* d_cost2, short2* d_disp_vec_temp, float* d_cost_temp, int w, int h, size_t cost_pitch, size_t disp_pitch)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    
    size_t cost_mem_w = cost_pitch/sizeof(float);
    size_t disp_mem_w = disp_pitch/sizeof(short2);

//     bao_timer_gpu timer;
//     timer.start();
    d_left_right_check_buffered<<<gridSize,blockSize>>>(d_disp_vec_temp,d_cost_temp,d_disp_vec,d_cost,d_disp_vec2,d_cost2,w,h,cost_mem_w,disp_mem_w); //0.02ms
    d_left_right_check_buffered<<<gridSize,blockSize>>>(d_disp_vec2,d_cost2,d_disp_vec2,d_cost2,d_disp_vec,d_cost,w,h,cost_mem_w,disp_mem_w); //0.02ms
    bao_cuda_copy_d2d(d_disp_vec,d_disp_vec_temp,h,w);
    bao_cuda_copy_d2d(d_cost,d_cost_temp,h,w);
//     timer.time_display("Refine: Left-right Check");
}


//////////////////////////////////////////////////////////////////////////
// outlier removal (remove isolated outliers)
//#define  STAT_RADIUS  6  //KITTI 2, SINTEL 4, middlebury 4 //move to def.h
#define  STAT_COUNT_THRESH  ((2*STAT_RADIUS+1)*(2*STAT_RADIUS+1)/2)
#define  STAT_SIM_THRESH 2

__global__ void d_outlier_removal(short2* d_disp_vec, float* d_cost, int w, int h, size_t cost_mem_w, size_t disp_mem_w)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;

    short2 curDisp = d_disp_vec[id_y*disp_mem_w + id_x];
    if (curDisp.x < 0 && curDisp.y < 0) return; //skip occlusion
    curDisp.x -= id_x;
    curDisp.y -= id_y;

    int count_similar_pix = 0;

//#pragma unroll
    for (int dy=-STAT_RADIUS; dy<=STAT_RADIUS; dy++) 
    {
//#pragma unroll
        for (int dx=-STAT_RADIUS; dx<=STAT_RADIUS; dx++)
        {
            int cy = id_y + dy;
            int cx = id_x + dx;
            if (cx < 0 || cy < 0 || cx >= w || cy >= h) continue;
            short2 neigDisp = d_disp_vec[cy*disp_mem_w + cx];
            neigDisp.x -= cx;
            neigDisp.y -= cy;
            if (abs(neigDisp.x-curDisp.x)<=STAT_SIM_THRESH && abs(neigDisp.y-curDisp.y)<=STAT_SIM_THRESH) count_similar_pix++;
        }
    }
    if (count_similar_pix < STAT_COUNT_THRESH) 
    {
        d_disp_vec[id_y*disp_mem_w + id_x] = make_short2(INVALID_LOCATION,INVALID_LOCATION); //set invalid disparity for later fixing
        d_cost[id_y*cost_mem_w + id_x] = FLT_MAX;
    }
}


extern "C" 
void baoCudaOutlierRemoval(short2* d_disp_vec, float* d_cost, int w, int h, size_t cost_pitch, size_t disp_pitch)
{
    size_t cost_mem_w = cost_pitch/sizeof(float);
    size_t disp_mem_w = disp_pitch/sizeof(short2);
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    d_outlier_removal<<<gridSize, blockSize>>>(d_disp_vec,d_cost,w,h,cost_mem_w,disp_mem_w);
}


//////////////////////////////////////////////////////////////////////////
// weighted median filtering
__device__ float _d_wmf_bilateral_weight(float4 a, float4 b, int dx, int dy)
{
    float delta_r = max(max(abs(b.x - a.x), abs(b.y - a.y)), abs(b.z - a.z));
    float coef_r = __expf(-(delta_r*delta_r) / (WMF_SIG_R*WMF_SIG_R));
    float coef_s = cWmfGaussian[dx] * cWmfGaussian[dy] ;
    return coef_r * coef_s;
}

__global__ void d_weighted_median_filtering(short2* d_disp_vec, int w, int h, size_t disp_mem_w, bool is_only_occlusion)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;

    short2 outputDisp = d_disp_vec[id_y*disp_mem_w + id_x]; //TODO: consider load into shared memory
    if (is_only_occlusion && outputDisp.x >= 0 && outputDisp.y >= 0) return; //non-occlusion

    // NOTE: ORIGINAL METHOD (first accumulate weights, then find medain) will run out memory for large displacement!
    // Here use a heavier computational method instead: select the output from all possible candidate which makes the cost minimum!
    float4 centerPix = tex2D(rgbaImg1Tex, id_x, id_y);
    float minCostSum = FLT_MAX;
    for (int dy = -WMF_RADIUS; dy <= WMF_RADIUS; dy++) for (int dx = -WMF_RADIUS; dx <= WMF_RADIUS; dx++) //run through all candidates
    {
        int cy = id_y + dy;
        int cx = id_x + dx;
        if (cx < 0 || cy < 0 || cx >= w || cy >= h) continue;
        short2 candidateDisp = d_disp_vec[cy*disp_mem_w + cx]; //TODO: consider load into shared memory
        if (candidateDisp.x < 0 || candidateDisp.y < 0) continue; //skip invalid disparity
        candidateDisp.x -= cx;
        candidateDisp.y -= cy;
        float costSum = 0.0f;
        float weightSum = 0.0f;
        for (int dy2 = -WMF_RADIUS; dy2 <= WMF_RADIUS; dy2++) 
        {
//#pragma unroll
            for (int dx2 = -WMF_RADIUS; dx2 <= WMF_RADIUS; dx2++)
            {
                int cy2 = id_y + dy2;
                int cx2 = id_x + dx2;
                if (cx2 < 0 || cy2 < 0 || cx2 >= w || cy2 >= h) continue;
                short2 curDisp = d_disp_vec[cy2*disp_mem_w + cx2]; //TODO: consider load into shared memory
                if (curDisp.x < 0 || curDisp.y < 0) continue; //skip invalid disparity
                curDisp.x -= cx2;
                curDisp.y -= cy2;
                float4 curPix = tex2D(rgbaImg1Tex, cx2, cy2);
                float curWeight = _d_wmf_bilateral_weight(centerPix, curPix, abs(dx2), abs(dy2));
                costSum += curWeight * max(abs(candidateDisp.x-curDisp.x),abs(candidateDisp.y-curDisp.y)); //accumulate absolute cost for each candidate
                weightSum += curWeight;
            }
        }
        if (weightSum > 0.0f && costSum < minCostSum)
        {
            minCostSum = costSum;
            outputDisp.x = candidateDisp.x + id_x;
            outputDisp.y = candidateDisp.y + id_y;
        }
    }

    // output
    if (outputDisp.x < 0 || outputDisp.y < 0) return;
    d_disp_vec[id_y*disp_mem_w + id_x] = outputDisp;
}

extern "C" 
void baoCudaWeightedMedianFilter(short2* d_disp_vec, float* d_cost, uchar4* d_img, int w, int h, size_t img_pitch, size_t cost_pitch, size_t disp_pitch, int num_iter, bool is_only_occlusion)
{
    //bind imgs
    cudaChannelFormatDesc desc_img = cudaCreateChannelDesc<uchar4>();
    checkCudaErrors(cudaBindTexture2D(0, rgbaImg1Tex, d_img, desc_img, w, h, img_pitch));

    //init gaussian lookup table
    float fWmfGaussian[WMF_RADIUS+1];
    for (int i=0; i<=WMF_RADIUS; i++)
    {
        fWmfGaussian[i] = expf(-float(i*i)/(WMF_SIG_S*WMF_SIG_S));
    }
    checkCudaErrors(cudaMemcpyToSymbol(cWmfGaussian, fWmfGaussian, sizeof(float)*(WMF_RADIUS+1)));
    
    //launch kernel
    size_t disp_mem_w = disp_pitch/sizeof(short2);
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);

//     bao_timer_gpu timer;
//     timer.start();
    for (int i=0; i<num_iter; i++) d_weighted_median_filtering<<<gridSize, blockSize>>>(d_disp_vec, w, h, disp_mem_w, is_only_occlusion);
//     timer.time_display("Refine: WMF");
//     getLastCudaError("Refine: WMF FAILED");
}


//////////////////////////////////////////////////////////////////////////
// fill holes which are not fixed by wmf
__device__ float _d_rgb_max_dist(float4 a, float4 b)
{
    float mod = max(max(abs(b.x - a.x), abs(b.y - a.y)), abs(b.z - a.z));
    return mod;
}

__global__ void d_fill_holes(short2* d_disp_vec, float* d_cost, int w, int h, size_t cost_mem_w, size_t disp_mem_w)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;

    short2 curDisp = d_disp_vec[id_y*disp_mem_w + id_x];
    if (curDisp.x >= 0 && curDisp.y >= 0) return; //skip non-occlusion

    // look into four directions, find the first valid disparity
    short2 neighDisp[4] = {curDisp,curDisp,curDisp,curDisp}; 
    int neighX[4] = {id_x,id_x,id_x,id_x};
    int neighY[4] = {id_y,id_y,id_y,id_y};
    for (int cx=id_x-1; cx>=0; cx--) //left
    {
        neighDisp[0] = d_disp_vec[id_y*disp_mem_w + cx];
        if (neighDisp[0].x>=0 && neighDisp[0].y>=0) 
        {
            neighX[0] = cx;
            neighY[0] = id_y;
            break;
        }
    }
    for (int cx=id_x+1; cx<w; cx++) //right
    {
        neighDisp[1] = d_disp_vec[id_y*disp_mem_w + cx];
        if (neighDisp[1].x>=0 && neighDisp[1].y>=0) 
        {
            neighX[1] = cx;
            neighY[1] = id_y;
            break;
        }
    }
    for (int cy=id_y-1; cy>=0; cy--) //up
    {
        neighDisp[2] = d_disp_vec[cy*disp_mem_w + id_x];
        if (neighDisp[2].x>=0 && neighDisp[2].y>=0) 
        {
            neighX[2] = id_x;
            neighY[2] = cy;
            break;
        }
    }
    for (int cy=id_y+1; cy<h; cy++) //down
    {
        neighDisp[3] = d_disp_vec[cy*disp_mem_w + id_x];
        if (neighDisp[3].x>=0 && neighDisp[3].y>=0) 
        {
            neighX[3] = id_x;
            neighY[3] = cy;
            break;
        }
    }

    float4 curPix = tex2D(rgbaImg1Tex,id_x,id_y);
    float minPixDiff = FLT_MAX;

//#pragma unroll
    for (int i=0; i<4; i++)
    {
        float4 neighPix = tex2D(rgbaImg1Tex,neighX[i],neighY[i]);
        float pixDiff = _d_rgb_max_dist(curPix,neighPix);
        if (pixDiff < minPixDiff && neighDisp[i].x >= 0 && neighDisp[i].y >= 0)
        {
            minPixDiff = pixDiff;
            curDisp.x = neighDisp[i].x-neighX[i];
            curDisp.y = neighDisp[i].y-neighY[i];
        }
    }

    // output
    curDisp.x += id_x;
    curDisp.y += id_y;
    d_disp_vec[id_y*disp_mem_w + id_x] = curDisp;
}

extern "C" 
void baoCudaFillHole(short2* d_disp_vec, float* d_cost, uchar4* d_img, int w, int h, size_t img_pitch, size_t cost_pitch, size_t disp_pitch)
{
    //bind imgs
    cudaChannelFormatDesc desc_img = cudaCreateChannelDesc<uchar4>();
    checkCudaErrors(cudaBindTexture2D(0, rgbaImg1Tex, d_img, desc_img, w, h, img_pitch));
    
    //launch kernel
    size_t cost_mem_w = cost_pitch/sizeof(float);
    size_t disp_mem_w = disp_pitch/sizeof(short2);
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
//     bao_timer_gpu timer;
//     timer.start();
    d_fill_holes<<<gridSize, blockSize>>>(d_disp_vec,d_cost,w,h,cost_mem_w,disp_mem_w);
//     timer.time_display("Refine: Filling Holes");
//     getLastCudaError("Refine: Filling Holes FAILED");
}


//////////////////////////////////////////////////////////////////////////
// subpixel refinement
#define SUBPIX_NEIG_RADIUS     2   //5*5 neighborhood
#define SUBPIX_UNKNOWNS        6
#define SUBPIX_NUM_EQU        ((2*SUBPIX_NEIG_RADIUS+1)*(2*SUBPIX_NEIG_RADIUS+1))  //5*5=25
#define SUBPIX_POS_2_IDX(x,y) ((y+SUBPIX_NEIG_RADIUS)*(2*SUBPIX_NEIG_RADIUS+1)+x+SUBPIX_NEIG_RADIUS) 

#include "3rdparty/nv-cuda-v5.0/bicubicTexture_kernel.cuh"

__device__ inline void _d_subpix_bilateral_dist(float& cost, float& weight, float4 c1, float4 c2, float4 a, float4 b, unsigned char s1, unsigned char s2, int dx, int dy)
{
    float mod = max(max(abs(b.x - a.x), abs(b.y - a.y)), abs(b.z - a.z));
    unsigned char census_dist = 0;
    unsigned char census_diff = s1^s2;
    while (census_diff) { census_dist++; census_diff &= census_diff-1; };

    float cost_ad = 1 - __expf(-(mod*mod) / (LAMBDA_AD*LAMBDA_AD));
    float cost_census = cCensusNegativeGaussian[census_dist];

    float delta_r1 = max(max(abs(c1.x - a.x), abs(c1.y - a.y)), abs(c1.z - a.z));
    float delta_r2 = max(max(abs(c2.x - b.x), abs(c2.y - b.y)), abs(c2.z - b.z));
    
    float coef_r = __expf(-(delta_r1*delta_r1+delta_r2*delta_r2) / (SUBPIX_SIG_R*SUBPIX_SIG_R));
    float coef_s = cSubpixGaussian[dx] * cSubpixGaussian[dy];

    weight = coef_r * coef_s; //output
    cost = (cost_ad + cost_census)*weight; //output

    return;
}

__device__ inline void _d_subpix_L2_dist(float& cost, float& weight, float4 c1, float4 c2, float4 a, float4 b, unsigned char s1, unsigned char s2, int dx, int dy)
{
    float mod = max(max(abs(b.x - a.x), abs(b.y - a.y)), abs(b.z - a.z));
    unsigned char census_dist = 0;
    unsigned char census_diff = s1^s2;
    while (census_diff) { census_dist++; census_diff &= census_diff-1; };

    float cost_ad = 1 - __expf(-(mod*mod) / (LAMBDA_AD*LAMBDA_AD));
    float cost_census = cCensusNegativeGaussian[census_dist];

    weight = 1.0f; //output
    cost = (cost_ad + cost_census); //output

    return;
}

__device__ inline float _d_calc_subpix_cost(float x1,float y1,float x2,float y2)
{
    float4 centerPix1 = tex2D(rgbaImg1Tex, x1, y1); //data in frame 1
    float4 centerPix2 = tex2D(rgbaImg2Tex, x2, y2); //data in frame 2
    float cost_sum = 0.0f;
    float weight_sum = 0.0f;
    float weight = 0.0f;
    float cost = 0.0f;
    float4 curPix1;
    float4 curPix2;
    unsigned char curCensus1;
    unsigned char curCensus2;
    float ii,jj;
    for (int i = -SUBPIX_PATCH_R; i <= SUBPIX_PATCH_R; i+=2) //skip pixels
    {
//#pragma unroll  //NOTE: unrolling makes it slower in SM3.5!! (??: in SM1.0 makes 2x faster, when R=8)
        for (int j = -SUBPIX_PATCH_R; j <= SUBPIX_PATCH_R; j+=2) //skip pixels
        {
            ii = float(i)*SUBPIX_UP_FACTOR_INV;
            jj = float(j)*SUBPIX_UP_FACTOR_INV;
            curPix1 = tex2DBicubic<uchar4,float4>(rgbaImg1Tex, x1 + jj, y1 + ii); //data in frame 1
            curPix2 = tex2DBicubic<uchar4,float4>(rgbaImg2Tex, x2 + jj, y2 + ii); //data in frame 2
            curCensus1 = tex2D(census1Tex, (x1 + jj)*SUBPIX_UP_FACTOR, (y1 + ii)*SUBPIX_UP_FACTOR);
            curCensus2 = tex2D(census2Tex, (x2 + jj)*SUBPIX_UP_FACTOR, (y2 + ii)*SUBPIX_UP_FACTOR);
            _d_subpix_bilateral_dist(cost, weight, centerPix1, centerPix2, curPix1, curPix2, curCensus1, curCensus2, abs(j), abs(i)); //NOTE: original i and j
            //_d_subpix_L2_dist(cost, weight, centerPix1, centerPix2, curPix1, curPix2, curCensus1, curCensus2, abs(j), abs(i)); //NOTE: original i and j
            cost_sum += cost;
            weight_sum += weight;
        }
    }
    return (cost_sum/weight_sum);
}

__device__ inline void _d_conjugate_gradient_solver(float vecX[SUBPIX_UNKNOWNS],float matAtA[SUBPIX_UNKNOWNS][SUBPIX_UNKNOWNS],float vecAtB[SUBPIX_UNKNOWNS],float bufTemp[SUBPIX_UNKNOWNS*3])
{
    float normb=0.f;
    #pragma unroll
    for (int i=0; i<SUBPIX_UNKNOWNS; i++)
    {
        normb += vecAtB[i]*vecAtB[i];
    }
    normb = __fsqrt_rn(normb);
    
    #pragma unroll
    for (int i=0; i<SUBPIX_UNKNOWNS; i++)
    {
        vecX[i]=0;
        bufTemp[i]=vecAtB[i]; //r
        bufTemp[SUBPIX_UNKNOWNS+i]=vecAtB[i]; //d
    }
    float rtr=normb*normb;
    int niters=0;
    while(__fsqrt_rn(rtr)/normb>1.0e-6 && niters<5) //max iterations 5
    {
        niters=niters+1;
        #pragma unroll
        for (int i=0; i<SUBPIX_UNKNOWNS; i++)
        {
            bufTemp[2*SUBPIX_UNKNOWNS+i]=0; //ad
        }
        #pragma unroll
        for(int i=0;i<SUBPIX_UNKNOWNS;i++) 
        {
            #pragma unroll
            for(int j=0;j<SUBPIX_UNKNOWNS;j++) 
            {
                bufTemp[2*SUBPIX_UNKNOWNS+i] += matAtA[i][j]*bufTemp[SUBPIX_UNKNOWNS+j]; //ad=ata*d
            }
        }
        float dad=0;
        #pragma unroll
        for(int i=0;i<SUBPIX_UNKNOWNS;i++) 
        {
            dad += bufTemp[SUBPIX_UNKNOWNS+i]*bufTemp[2*SUBPIX_UNKNOWNS+i]; //ad*d
        }
        float alpha=rtr/dad;
        #pragma unroll
        for(int i=0;i<SUBPIX_UNKNOWNS;i++)
        {
            vecX[i]+=(float)(alpha*bufTemp[SUBPIX_UNKNOWNS+i]); //d
            bufTemp[i]-=(float)(alpha*bufTemp[2*SUBPIX_UNKNOWNS+i]); //ad
        }
        float rtrold=rtr;
        rtr=0;
        #pragma unroll
        for(int i=0;i<SUBPIX_UNKNOWNS;i++) 
        {
            rtr+=bufTemp[i]*bufTemp[i]; //r*r
        }
        float beta=rtr/rtrold;
        #pragma unroll
        for(int i=0;i<SUBPIX_UNKNOWNS;i++)
        {
            bufTemp[SUBPIX_UNKNOWNS+i]=(float)(bufTemp[i]+beta*bufTemp[SUBPIX_UNKNOWNS+i]); //d=r+beta*d
        }
    }
}

__global__ void d_subpixel_refine(float2* d_flow, short2* d_disp_vec, int w, int h, size_t flow_mem_w, size_t disp_mem_w)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    
    short2 curDisp = d_disp_vec[id_y*disp_mem_w + id_x];
    if (curDisp.x < 0 || curDisp.y < 0 || curDisp.x >= w || curDisp.y >= h) return;

    // compute cost
    float vecB[SUBPIX_NUM_EQU];
    bool is_all_zero = false;
    //#pragma unroll
    for (int dy=-SUBPIX_NEIG_RADIUS; dy<=SUBPIX_NEIG_RADIUS; dy++) 
    {
        //#pragma unroll
        for (int dx=-SUBPIX_NEIG_RADIUS; dx<=SUBPIX_NEIG_RADIUS; dx++)
        {
            float nx = float(curDisp.x) + float(dx)/float(SUBPIX_UP_FACTOR);
            float ny = float(curDisp.y) + float(dy)/float(SUBPIX_UP_FACTOR);
            if (nx<0 || nx>=w || ny<0 || ny>=h) 
            {
                vecB[SUBPIX_POS_2_IDX(dx,dy)] = 2.f;
            }
            else 
            {
                vecB[SUBPIX_POS_2_IDX(dx,dy)] = _d_calc_subpix_cost(id_x,id_y,nx,ny); 
                is_all_zero=true;
            }
        }
    }
    if (is_all_zero == false) return;

    // prepare data for solving linear system
    float matA[SUBPIX_NUM_EQU][SUBPIX_UNKNOWNS];
    float matAtA[SUBPIX_UNKNOWNS][SUBPIX_UNKNOWNS];
    float vecAtB[SUBPIX_UNKNOWNS];
    float vecX[SUBPIX_UNKNOWNS];

    // prepare matrix A
    //#pragma unroll
    for (int y=-SUBPIX_NEIG_RADIUS; y<=SUBPIX_NEIG_RADIUS; y++) 
    {
        //#pragma unroll
        for (int x=-SUBPIX_NEIG_RADIUS; x<=SUBPIX_NEIG_RADIUS; x++)
        {
            matA[SUBPIX_POS_2_IDX(x,y)][0] = float(x*x);
            matA[SUBPIX_POS_2_IDX(x,y)][1] = float(y*y);
            matA[SUBPIX_POS_2_IDX(x,y)][2] = float(x*y);
            matA[SUBPIX_POS_2_IDX(x,y)][3] = float(x);
            matA[SUBPIX_POS_2_IDX(x,y)][4] = float(y);
            matA[SUBPIX_POS_2_IDX(x,y)][5] = 1.0f;
        }
    }

    // prepare matrix AtA
    //#pragma unroll
    for(int y=0;y<SUBPIX_UNKNOWNS;y++) 
    {
        //#pragma unroll
        for(int x=0;x<SUBPIX_UNKNOWNS;x++)
        {
            float vv=0;
            //#pragma unroll
            for(int i=0;i<SUBPIX_NUM_EQU;i++) vv+=matA[i][y]*matA[i][x];
            matAtA[y][x]=float(vv);
        }
    }

    // prepare vector Atb
    //#pragma unroll
    for(int yy=0; yy<SUBPIX_UNKNOWNS; yy++)
    {
        float vv=0;
        //#pragma unroll
        for(int ii=0;ii<SUBPIX_NUM_EQU;ii++) vv+=matA[ii][yy]*vecB[ii];
        vecAtB[yy]=float(vv);
    }

    // solve linear system Ax = b (in fact, solve AtAx = Atb) [conjugate gradient]
    float bufTemp[SUBPIX_UNKNOWNS*3];
    _d_conjugate_gradient_solver(vecX,matAtA,vecAtB,bufTemp);
    
    // calculate output subpix position
    float denorm = vecX[2]*vecX[2] - 4*vecX[0]*vecX[1];
    if (denorm == 0) return;
    float subx = (2*vecX[3]*vecX[1] - vecX[2]*vecX[4]) / denorm; //(2db ¨C ce)/(c^2 - 4ab)
    float suby = (2*vecX[0]*vecX[4] - vecX[2]*vecX[3]) / denorm; //(2ae - dc)/(c^2 - 4ab)
    if (abs(suby)<=3 && abs(subx)<=3) 
    {
        float2 curFlow;
        curFlow.x = (float(curDisp.x-id_x)*SUBPIX_UP_FACTOR + subx)/float(SUBPIX_UP_FACTOR);
        curFlow.y = (float(curDisp.y-id_y)*SUBPIX_UP_FACTOR + suby)/float(SUBPIX_UP_FACTOR);
        // store to global memory
        d_flow[id_y*flow_mem_w + id_x] = curFlow;
    }
}

__global__ void d_convert_nnf_to_flow(float2* d_flow, short2* d_disp_vec, int w, int h, size_t flow_mem_w, size_t disp_mem_w)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    
    short2 curDisp = d_disp_vec[id_y*disp_mem_w + id_x];
    float2 curFlow;
    if (curDisp.x <= INVALID_LOCATION || curDisp.y <= INVALID_LOCATION)
    {
        curFlow.x = UNKNOWN_FLOW;
        curFlow.y = UNKNOWN_FLOW;
    }
    else
    {
        curFlow.x = float(curDisp.x-id_x);
        curFlow.y = float(curDisp.y-id_y);
    }
    d_flow[id_y*flow_mem_w + id_x] = curFlow; 
}

__global__ void d_convert_flow_to_nnf(short2* d_disp_vec, float2* d_flow, int w, int h, size_t flow_mem_w, size_t disp_mem_w)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    
    float2 curFlow = d_flow[id_y*flow_mem_w + id_x];
    short2 curDisp;
    if (curFlow.x > UNKNOWN_FLOW_THRESH || curFlow.y > UNKNOWN_FLOW_THRESH)
    {
        curDisp.x = INVALID_LOCATION;
        curDisp.y = INVALID_LOCATION;
    }
    else
    {
        curDisp.x = short(curFlow.x+id_x);
        curDisp.y = short(curFlow.y+id_y);
    }
    d_disp_vec[id_y*disp_mem_w + id_x] = curDisp; 
}

extern "C" 
void baoCudaSubpixRefine(float2* d_flow, short2* d_disp_vec, uchar4* d_img1, uchar4* d_img2, unsigned char* d_census1_up, unsigned char* d_census2_up, int w, int h, size_t img_pitch, size_t census_pitch_up, size_t disp_pitch, size_t flow_pitch)
{
    //bind imgs
    rgbaImg1Tex.filterMode = cudaFilterModeLinear;
    rgbaImg2Tex.filterMode = cudaFilterModeLinear;
    cudaChannelFormatDesc desc_img = cudaCreateChannelDesc<uchar4>();
    checkCudaErrors(cudaBindTexture2D(0, rgbaImg1Tex, d_img1, desc_img, w, h, img_pitch));
    checkCudaErrors(cudaBindTexture2D(0, rgbaImg2Tex, d_img2, desc_img, w, h, img_pitch));

    //bind the census with texture memory
    census1Tex.filterMode = cudaFilterModePoint;
    census1Tex.normalized = false;
    census2Tex.filterMode = cudaFilterModePoint;
    census2Tex.normalized = false;
    cudaChannelFormatDesc desc_census = cudaCreateChannelDesc<unsigned char>();
    checkCudaErrors(cudaBindTexture2D(0, census1Tex, d_census1_up, desc_census, w*SUBPIX_UP_FACTOR, h*SUBPIX_UP_FACTOR, census_pitch_up));
    checkCudaErrors(cudaBindTexture2D(0, census2Tex, d_census2_up, desc_census, w*SUBPIX_UP_FACTOR, h*SUBPIX_UP_FACTOR, census_pitch_up));

    //init gaussian lookup table
    float fSubpixGaussian[SUBPIX_PATCH_R+1];
    for (int i=0; i<=SUBPIX_PATCH_R; i++)
    {
        fSubpixGaussian[i] = expf(-float(i*i)/(SUBPIX_SIG_S*SUBPIX_SIG_S));
    }
    checkCudaErrors(cudaMemcpyToSymbol(cSubpixGaussian, fSubpixGaussian, sizeof(float)*(SUBPIX_PATCH_R+1)));
    float fNegativeGaussian[CENSUS_MAX_DIFF+1];
    for (int i=0; i<=CENSUS_MAX_DIFF; i++)
    {
        fNegativeGaussian[i] = 1-expf(-float(i*i)/(LAMBDA_CENSUS*CENSUS_MAX_DIFF*LAMBDA_CENSUS*CENSUS_MAX_DIFF));
    }
    checkCudaErrors(cudaMemcpyToSymbol(cCensusNegativeGaussian, fNegativeGaussian, sizeof(float)*(CENSUS_MAX_DIFF+1)));

    //launch kernel
    size_t disp_mem_w = disp_pitch/sizeof(short2);
    size_t flow_mem_w = flow_pitch/sizeof(float2);
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    bao_timer_gpu timer;
    timer.start();
    //d_convert_nnf_to_flow<<<gridSize, blockSize>>>(d_flow,d_disp_vec,w,h,flow_mem_w,disp_mem_w);
    d_subpixel_refine<<<gridSize, blockSize>>>(d_flow,d_disp_vec,w,h,flow_mem_w,disp_mem_w);
    timer.time_display("Refine: Subpixel Refinement");
    getLastCudaError("Refine: Subpixel Refinement FAILED");
}

extern "C" 
void baoCudaNNF2Flow(float2* d_flow, short2* d_disp_vec, int w, int h, size_t disp_pitch, size_t flow_pitch)
{
    //launch kernel
    size_t disp_mem_w = disp_pitch/sizeof(short2);
    size_t flow_mem_w = flow_pitch/sizeof(float2);
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    d_convert_nnf_to_flow<<<gridSize, blockSize>>>(d_flow,d_disp_vec,w,h,flow_mem_w,disp_mem_w);
//     getLastCudaError("Refine: Convert NNF to Flow FAILED");
}

extern "C" 
void baoCudaFlow2NNF(short2* d_disp_vec, float2* d_flow, int w, int h, size_t disp_pitch, size_t flow_pitch)
{
    //launch kernel
    size_t disp_mem_w = disp_pitch/sizeof(short2);
    size_t flow_mem_w = flow_pitch/sizeof(float2);
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    d_convert_flow_to_nnf<<<gridSize, blockSize>>>(d_disp_vec,d_flow,w,h,flow_mem_w,disp_mem_w);
    getLastCudaError("Refine: Convert NNF to Flow FAILED");
}


//////////////////////////////////////////////////////////////////////////
// final flow smoothing
// #define POSTPROC_BLF_SIG_S       5 //for SINTEL 10 , kitti 5  //move to def.h
#define POSTPROC_BLF_SIG_R       0.02f    //move to def.h
#define POSTPROC_BLF_RADIUS      (2*POSTPROC_BLF_SIG_S)
__constant__ float cBlfGaussian[POSTPROC_BLF_RADIUS+1];   //gaussian array in device side

__device__ float _d_bilateral_weight(float4 a, float4 b, int dx, int dy)
{
    float delta_r = max(max(abs(b.x - a.x), abs(b.y - a.y)), abs(b.z - a.z));
    float coef_r = __expf(-(delta_r*delta_r) / (POSTPROC_BLF_SIG_R*POSTPROC_BLF_SIG_R));
    float coef_s = cBlfGaussian[dx] * cBlfGaussian[dy] ;
    return coef_r * coef_s;
}

__global__ void d_flow_bilateral_filtering(float2* d_flow, int w, int h, size_t flow_mem_w)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;

    float4 centerPix = tex2D(rgbaImg1Tex, id_x, id_y);
    float2 newFlow = make_float2(0.f,0.f);
    float weightSum = 0.f;
    //#pragma unroll //NOTE: unrolling in SM3.5 makes it much slower!!
    for (int dy = -POSTPROC_BLF_RADIUS; dy <= POSTPROC_BLF_RADIUS; dy++) 
    {
        //#pragma unroll //NOTE: unrolling in SM3.5 makes it much slower!!
        for (int dx = -POSTPROC_BLF_RADIUS; dx <= POSTPROC_BLF_RADIUS; dx++)
        {
            int cy = id_y + dy;
            int cx = id_x + dx;
            if (cx < 0 || cy < 0 || cx >= w || cy >= h) continue;
            float2 curFlow = d_flow[cy*flow_mem_w + cx]; //TODO: consider load into shared memory
            if (curFlow.x > UNKNOWN_FLOW_THRESH || curFlow.y > UNKNOWN_FLOW_THRESH) continue;
            float4 curPix = tex2D(rgbaImg1Tex, cx, cy);
            float curWeight = _d_bilateral_weight(centerPix, curPix, abs(dx), abs(dy));
            newFlow.x += curWeight * curFlow.x;
            newFlow.y += curWeight * curFlow.y;
            weightSum += curWeight;
        }
    }

    // output
    if (weightSum != 0)
    {
        newFlow.x /= weightSum;
        newFlow.y /= weightSum;
        d_flow[id_y*flow_mem_w + id_x] = newFlow;
    }
}

extern "C" 
void baoCudaFlowSmoothing(float2* d_flow, uchar4* d_img, int w, int h, size_t img_pitch, size_t flow_pitch)
{
    //bind imgs
    cudaChannelFormatDesc desc_img = cudaCreateChannelDesc<uchar4>();
    checkCudaErrors(cudaBindTexture2D(0, rgbaImg1Tex, d_img, desc_img, w, h, img_pitch));

    //init gaussian lookup table
    float fBlfGaussian[POSTPROC_BLF_RADIUS+1];
    for (int i=0; i<=POSTPROC_BLF_RADIUS; i++)
    {
        fBlfGaussian[i] = expf(-float(i*i)/float(POSTPROC_BLF_SIG_S*POSTPROC_BLF_SIG_S));
    }
    checkCudaErrors(cudaMemcpyToSymbol(cBlfGaussian, fBlfGaussian, sizeof(float)*(POSTPROC_BLF_RADIUS+1)));
    
    //launch kernel
    size_t flow_mem_w = flow_pitch/sizeof(float2);
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);

//     bao_timer_gpu timer;
//     timer.start();
    d_flow_bilateral_filtering<<<gridSize, blockSize>>>(d_flow, w, h, flow_mem_w);
//     timer.time_display("Refine: BLF");
//     getLastCudaError("Refine: BLF FAILED");
}


__global__ void d_bilateral_upsample_flow(float2* d_flow_vec, int w, int h, size_t flow_mem_w, float2* d_flow_vec_s, int w_s, int h_s, size_t flow_mem_w_s, float ratio_up)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;

    float4 centerPix = tex2D(rgbaImg1Tex, id_x, id_y);
    float2 newFlow = make_float2(0.f,0.f);
    float weightSum = 0.f;
    //#pragma unroll //NOTE: unrolling in SM3.5 makes it much slower!!
    for (int dy = -POSTPROC_BLF_RADIUS; dy <= POSTPROC_BLF_RADIUS; dy++) 
    {
        //#pragma unroll //NOTE: unrolling in SM3.5 makes it much slower!!
        for (int dx = -POSTPROC_BLF_RADIUS; dx <= POSTPROC_BLF_RADIUS; dx++)
        {
            int cy = id_y + dy;
            int cx = id_x + dx;
            if (cx < 0 || cy < 0 || cx >= w || cy >= h) continue;
            float2 curFlow = d_flow_vec_s[int(cy/ratio_up)*flow_mem_w_s + int(cx/ratio_up)]; //TODO: consider load into shared memory
            if (curFlow.x > UNKNOWN_FLOW_THRESH || curFlow.y > UNKNOWN_FLOW_THRESH) continue;
            float4 curPix = tex2D(rgbaImg1Tex, cx, cy);
            float curWeight = _d_bilateral_weight(centerPix, curPix, abs(dx), abs(dy));
            newFlow.x += curWeight * curFlow.x;
            newFlow.y += curWeight * curFlow.y;
            weightSum += curWeight;
        }
    }

    // output
    if (weightSum != 0)
    {
        newFlow.x /= weightSum;
        newFlow.y /= weightSum;
        newFlow.x *= ratio_up; //NOTE: remember to upscale the value of flow
        newFlow.y *= ratio_up; //NOTE: remember to upscale the value of flow
        d_flow_vec[id_y*flow_mem_w + id_x] = newFlow;
    }
}
extern "C" 
void baoCudaFlowBilteralUpsampling(float2* d_flow_vec, uchar4* d_img, int w, int h, size_t img_pitch, float2* d_flow_vec_small, int w_s, int h_s, float ratio_up)
{
    //bind imgs
    cudaChannelFormatDesc desc_img = cudaCreateChannelDesc<uchar4>();
    checkCudaErrors(cudaBindTexture2D(0, rgbaImg1Tex, d_img, desc_img, w, h, img_pitch));

    //init gaussian lookup table
    float fBlfGaussian[POSTPROC_BLF_RADIUS+1];
    for (int i=0; i<=POSTPROC_BLF_RADIUS; i++)
    {
        fBlfGaussian[i] = expf(-float(i*i)/float(POSTPROC_BLF_SIG_S*POSTPROC_BLF_SIG_S));
    }
    checkCudaErrors(cudaMemcpyToSymbol(cBlfGaussian, fBlfGaussian, sizeof(float)*(POSTPROC_BLF_RADIUS+1)));

    size_t flow_mem_w = w;
    size_t flow_mem_w_s = w_s;

    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    d_bilateral_upsample_flow<<<gridSize, blockSize>>>(d_flow_vec,w,h,flow_mem_w,d_flow_vec_small,w_s,h_s,flow_mem_w_s,ratio_up);
}


__global__ void d_flow_cutoff(float2* d_flow, int w, int h, size_t flow_mem_w, float max_flow_val)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;

    float2 newFlow = d_flow[id_y*flow_mem_w + id_x];
    newFlow.x = __max(-max_flow_val, __min(max_flow_val, newFlow.x)); 
    newFlow.y = __max(-max_flow_val, __min(max_flow_val, newFlow.y));
    d_flow[id_y*flow_mem_w + id_x] = newFlow;
}
extern "C" 
void baoCudaFlowCutoff(float2* d_flow, int w, int h, size_t flow_pitch, float max_flow_val)
{
    //launch kernel
    size_t flow_mem_w = flow_pitch/sizeof(float2);
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);

    d_flow_cutoff<<<gridSize, blockSize>>>(d_flow, w, h, flow_mem_w, max_flow_val);
//     getLastCudaError("Refine: Flow Cutoff FAILED");
}


//////////////////////////////////////////////////////////////////////////
// guide image smoothing
#define POSTPROC_MEDIAN_RADIUS      2
#define POSTPROC_MEDIAN_ROW_NUMPIX  (2*POSTPROC_MEDIAN_RADIUS+1)
#define POSTPROC_MEDIAN_THRESH      ((POSTPROC_MEDIAN_ROW_NUMPIX*POSTPROC_MEDIAN_ROW_NUMPIX)/2)

__global__ void d_median_filtering(uchar4* d_img_smoothed, uchar4* d_img, int w, int h, size_t img_mem_w)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;

    // collect neighbors
    uchar4 neighPix[POSTPROC_MEDIAN_ROW_NUMPIX*POSTPROC_MEDIAN_ROW_NUMPIX];
    //#pragma unroll 
    for (int dy = -POSTPROC_MEDIAN_RADIUS; dy <= POSTPROC_MEDIAN_RADIUS; dy++) 
    {
        //#pragma unroll 
        for (int dx = -POSTPROC_MEDIAN_RADIUS; dx <= POSTPROC_MEDIAN_RADIUS; dx++)
        {
            int cy = id_y + dy;
            int cx = id_x + dx;
            if (cx < 0 || cy < 0 || cx >= w || cy >= h) neighPix[(dy+POSTPROC_MEDIAN_RADIUS)*POSTPROC_MEDIAN_ROW_NUMPIX + dx + POSTPROC_MEDIAN_RADIUS] = make_uchar4(0,0,0,0);
            else neighPix[(dy+POSTPROC_MEDIAN_RADIUS)*POSTPROC_MEDIAN_ROW_NUMPIX + dx + POSTPROC_MEDIAN_RADIUS] = d_img[cy*img_mem_w + cx];
        }
    }

    // sort
    unsigned char tempVal;
    #pragma unroll  //makes it faster!
    for (int i=0; i<POSTPROC_MEDIAN_ROW_NUMPIX*POSTPROC_MEDIAN_ROW_NUMPIX; i++)
    {
        #pragma unroll  //makes it faster!
        for (int j=0; j<POSTPROC_MEDIAN_ROW_NUMPIX*POSTPROC_MEDIAN_ROW_NUMPIX; j++)
        {
            if (neighPix[i].x > neighPix[j].x) 
            {
                tempVal = neighPix[i].x;
                neighPix[i].x = neighPix[j].x;
                neighPix[j].x = tempVal;
            }
            if (neighPix[i].y > neighPix[j].y) 
            {
                tempVal = neighPix[i].y;
                neighPix[i].y = neighPix[j].y;
                neighPix[j].y = tempVal;
            }
            if (neighPix[i].z > neighPix[j].z) 
            {
                tempVal = neighPix[i].z;
                neighPix[i].z = neighPix[j].z;
                neighPix[j].z = tempVal;
            }
        }
    }

    // select median
    uchar4 newPix = neighPix[POSTPROC_MEDIAN_THRESH];
    d_img_smoothed[id_y*img_mem_w + id_x] = newPix;
}

__global__ void d_image_bilateral_filtering(uchar4* d_img_smoothed, uchar4* d_img, int w, int h, size_t img_mem_w)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;

    float4 centerPix = tex2D(rgbaImg1Tex, id_x, id_y);
    float4 fnewPix = make_float4(0,0,0,0);
    float weightSum = 0.f;
    //#pragma unroll //NOTE: unrolling in SM3.5 makes it much slower!!
    for (int dy = -POSTPROC_BLF_RADIUS; dy <= POSTPROC_BLF_RADIUS; dy++) 
    {
        //#pragma unroll //NOTE: unrolling in SM3.5 makes it much slower!!
        for (int dx = -POSTPROC_BLF_RADIUS; dx <= POSTPROC_BLF_RADIUS; dx++)
        {
            int cy = id_y + dy;
            int cx = id_x + dx;
            if (cx < 0 || cy < 0 || cx >= w || cy >= h) continue;
            float4 curPix = tex2D(rgbaImg1Tex, cx, cy);
            float curWeight = _d_bilateral_weight(centerPix, curPix, abs(dx), abs(dy));
            uchar4 neighPix = d_img[cy*img_mem_w + cx]; //TODO: consider load into shared memory
            fnewPix.x += curWeight * neighPix.x;
            fnewPix.y += curWeight * neighPix.y;
            fnewPix.z += curWeight * neighPix.z;
            weightSum += curWeight;
        }
    }

    // output
    if (weightSum != 0)
    {
        fnewPix.x /= weightSum;
        fnewPix.y /= weightSum;
        fnewPix.z /= weightSum;
        uchar4 newPix;
        newPix.x = fnewPix.x;
        newPix.y = fnewPix.y;
        newPix.z = fnewPix.z;
        d_img_smoothed[id_y*img_mem_w + id_x] = newPix;
    }
    else
    {
        d_img_smoothed[id_y*img_mem_w + id_x] = d_img[id_y*img_mem_w + id_x];
    }
}

extern "C" 
void baoCudaImageSmoothing(uchar4* d_img_smoothed, uchar4* d_img, int w, int h, size_t img_pitch)
{
    size_t img_mem_w = img_pitch/sizeof(uchar4);
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    bao_timer_gpu timer;

    /* median filtering */
    timer.start();
    d_median_filtering<<<gridSize, blockSize>>>(d_img_smoothed, d_img, w, h, img_mem_w);
    timer.time_display("Pre: Guide Image Median Filtering");
    getLastCudaError("Pre: Median Filtering FAILED");

    /* bilateral filtering */
    //bind imgs
    cudaChannelFormatDesc desc_img = cudaCreateChannelDesc<uchar4>();
    checkCudaErrors(cudaBindTexture2D(0, rgbaImg1Tex, d_img, desc_img, w, h, img_pitch));

    //init gaussian lookup table
    float fBlfGaussian[POSTPROC_BLF_RADIUS+1];
    for (int i=0; i<=POSTPROC_BLF_RADIUS; i++)
    {
        fBlfGaussian[i] = expf(-float(i*i)/float(POSTPROC_BLF_SIG_S*POSTPROC_BLF_SIG_S));
    }
    checkCudaErrors(cudaMemcpyToSymbol(cBlfGaussian, fBlfGaussian, sizeof(float)*(POSTPROC_BLF_RADIUS+1)));
    
    //launch kernel
    timer.start();
    d_image_bilateral_filtering<<<gridSize, blockSize>>>(d_img_smoothed, d_img, w, h, img_mem_w);
    timer.time_display("Pre: Guide Image Bilateral Filtering");
    getLastCudaError("Pre: Guide Image BLF FAILED");
}




extern "C" void baoCudaCensusTransform(unsigned char* d_census1, unsigned char* d_census2, uchar4* d_img1, uchar4* d_img2, int w, int h, size_t img_pitch, size_t census_pitch);
extern "C" 
void baoCudaPatchMatchMultiscalePrepare(uchar4** pImgPyr1, uchar4** pImgPyr2, unsigned char** pCensusPyr1, unsigned char** pCensusPyr2, uchar4** pTempPyr1, uchar4** pTempPyr2, int* arrH, int* arrW, size_t*arrPitchUchar4, size_t*arrPitchUchar1, int nLevels, uchar4*d_img1, uchar4*d_img2, int h, int w)
{
    bao_cuda_gauss_filter_pitched(pImgPyr1[0],d_img1,arrPitchUchar4[0],h,w,.5f,2);
    bao_cuda_gauss_filter_pitched(pImgPyr2[0],d_img2,arrPitchUchar4[0],h,w,.5f,2);
    bao_cuda_construct_gauss_pyramid_pitched(pImgPyr1,pImgPyr1[0],pTempPyr1,nLevels,arrH,arrW,arrPitchUchar4,PYR_RATIO);
    bao_cuda_construct_gauss_pyramid_pitched(pImgPyr2,pImgPyr2[0],pTempPyr2,nLevels,arrH,arrW,arrPitchUchar4,PYR_RATIO);
    for (int i=0; i<nLevels; i++)
    {
        baoCudaCensusTransform(pCensusPyr1[i],pCensusPyr2[i],pImgPyr1[i],pImgPyr2[i],arrW[i],arrH[i],arrPitchUchar4[i],arrPitchUchar1[i]);
    }
}


extern "C" void baoCudaFlowBilteralUpsampling(float2* d_flow_vec, uchar4* d_img1, int w, int h, size_t img_pitch, float2* d_flow_vec_small, int w_s, int h_s, float ratio_up);
extern "C" void baoCudaBLFCostFilterRefine(float2* d_flow_vec, uchar4* d_img1, uchar4* d_img2, unsigned char* d_census1, unsigned char* d_census2, int w, int h, size_t img_pitch, size_t census_pitch);
extern "C" 
void baoCudaBLF_C2F(float2** pFlowPyr, uchar4** pImgPyr1, uchar4** pImgPyr2, unsigned char** pCensusPyr1, unsigned char** pCensusPyr2, float2** pTempPyr1, float2** pTempPyr2, int* arrH, int* arrW, size_t* arrPitchUchar4, size_t* arrPitchUchar1, int nLayerIdx)
{
    //from nLayerIdx+1 to nLayerIdx
//     baoCudaFlowBilteralUpsampling(pFlowPyr[nLayerIdx],pImgPyr1[nLayerIdx],arrW[nLayerIdx],arrH[nLayerIdx],arrPitchUchar4[nLayerIdx],pFlowPyr[nLayerIdx+1],arrW[nLayerIdx+1],arrH[nLayerIdx+1],PYR_RATIO_UP);

    bao_cuda_bilinear_resize(pFlowPyr[nLayerIdx],arrH[nLayerIdx],arrW[nLayerIdx],pFlowPyr[nLayerIdx+1],arrH[nLayerIdx+1],arrW[nLayerIdx+1],PYR_RATIO_UP);
    bao_cuda_multiply_scalar(pFlowPyr[nLayerIdx],pFlowPyr[nLayerIdx],2.0f,arrH[nLayerIdx],arrW[nLayerIdx]);

    //blf cost filter
    baoCudaBLFCostFilterRefine(pFlowPyr[nLayerIdx],pImgPyr1[nLayerIdx],pImgPyr2[nLayerIdx],pCensusPyr1[nLayerIdx],pCensusPyr2[nLayerIdx],arrW[nLayerIdx],arrH[nLayerIdx],arrPitchUchar4[nLayerIdx],arrPitchUchar1[nLayerIdx]);
}

