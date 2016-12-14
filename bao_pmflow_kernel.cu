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


#define  SIG_S_TABLE_LEN  (PATCH_R*2+1)
#define  CENSUS_MAX_DIFF  8

__constant__ float cSpatialGaussian[PATCH_R+1];   //gaussian array in device side
__constant__ float cCensusNegativeGaussian[CENSUS_MAX_DIFF+1];  //for census transform distance
texture<uchar4, 2, cudaReadModeNormalizedFloat> rgbaImg1Tex; //in texture memory
texture<uchar4, 2, cudaReadModeNormalizedFloat> rgbaImg2Tex; //in texture memory
texture<unsigned char, 2, cudaReadModeElementType> census1Tex; //in texture memory
texture<unsigned char, 2, cudaReadModeElementType> census2Tex; //in texture memory


#define BLOCK_DIM_X  16
#define BLOCK_DIM_Y  16


//////////////////////////////////////////////////////////////////////////
// random number generator (RNG)
curandState* g_d_rand_states;

__global__ void d_setup_randgen(curandState* randState, int w, int h)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    //int id = id_y*w + id_x;
    int block_id = blockIdx.y * gridDim.x + blockIdx.x;

    /* Each thread gets same seed, a different sequence 
       number, no offset */ 
    //   __device__ void
    //     curand_init (
    //     unsigned long long seed, unsigned long long sequence,
    //     unsigned long long offset, curandState_t *state)

    if (threadIdx.x == 0 && threadIdx.y == 0) //only one thread in each block works! (to avoid branch divergence inside curand_init())
    {
        curandState localState = randState[block_id];
        curand_init(1234, block_id, 0, &localState);
        randState[block_id] = localState;
    }
}

__global__ void d_gen_rand_field(curandState* randStateArr, 
                                short2* vecField,int w,int h,size_t disp_mem_w)
{
    __shared__ short2 s_res_vec[BLOCK_DIM_Y][BLOCK_DIM_X];
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    //int id = id_y*w + id_x;
    int block_id = blockIdx.y * gridDim.x + blockIdx.x;

    if (threadIdx.x == 0 && threadIdx.y == 0)  //only one thread in each block works! 
    {
        unsigned int rdn1,rdn2;

        /* Copy state to local memory for efficiency */
        curandState localState = randStateArr[block_id];
        for (int i=0; i<BLOCK_DIM_Y; i++) 
        {
//#pragma unroll
            for (int j=0; j<BLOCK_DIM_X; j++)
            {
                rdn1 = curand(&localState);
                rdn2 = curand(&localState);
                /* Store results into shared memory*/
                s_res_vec[i][j].x = short(rdn1%(w+1));
                s_res_vec[i][j].y = short(rdn2%(h+1));
            }
        }

        /* Copy state back to global memory */
        randStateArr[block_id] = localState;
    }
    __syncthreads();

    /* Store results into global memory*/
    vecField[id_y*disp_mem_w + id_x] = s_res_vec[threadIdx.y][threadIdx.x]; //all thread works
}

__global__ void d_gen_rand_field_scaled(curandState* randStateArr, 
                                short2* vecField,float* scaleField,int w,int h,size_t disp_mem_w,size_t scale_mem_w)
{
    __shared__ short2 s_res_vec[BLOCK_DIM_Y][BLOCK_DIM_X];
    __shared__ float s_res_scale_filed[BLOCK_DIM_Y][BLOCK_DIM_X];
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    //int id = id_y*w + id_x;
    int block_id = blockIdx.y * gridDim.x + blockIdx.x;

    if (threadIdx.x == 0 && threadIdx.y == 0)  //only one thread in each block works! 
    {
        unsigned int rdn1,rdn2;

        /* Copy state to local memory for efficiency */
        curandState localState = randStateArr[block_id];
        for (int i=0; i<BLOCK_DIM_Y; i++) 
        {
//#pragma unroll
            for (int j=0; j<BLOCK_DIM_X; j++)
            {
                rdn1 = curand(&localState);
                rdn2 = curand(&localState);
                /* Store results into shared memory*/
                s_res_vec[i][j].x = short(rdn1%(w+1));
                s_res_vec[i][j].y = short(rdn2%(h+1));
                s_res_scale_filed[i][j] = float((10 + ((rdn2%PM_SCALE_RANGE)-PM_SCALE_MIN))/float(10.0f)); //0.9~1.3
            }
        }

        /* Copy state back to global memory */
        randStateArr[block_id] = localState;
    }
    __syncthreads();

    /* Store results into global memory*/
    vecField[id_y*disp_mem_w + id_x] = s_res_vec[threadIdx.y][threadIdx.x]; //all thread works
    scaleField[id_y*disp_mem_w + id_x] = s_res_scale_filed[threadIdx.y][threadIdx.x]; //all thread works
}


extern "C"
void baoGenerateRandomField(short2* d_disp_vec,int w,int h,size_t disp_mem_w)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    //bao_timer_gpu timer;
    //timer.start();
    d_setup_randgen<<<gridSize, blockSize>>>(g_d_rand_states,w,h);
    //timer.time_display("Generate Random Field 1");
    //timer.start();
    d_gen_rand_field<<<gridSize, blockSize>>>(g_d_rand_states,d_disp_vec,w,h,disp_mem_w);
    //timer.time_display("Generate Random Field 2");
}

extern "C"
void baoGenerateRandomField_Scaled(short2* d_disp_vec,float* d_scale,int w,int h,size_t disp_mem_w,size_t scale_mem_w)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    //bao_timer_gpu timer;
    //timer.start();
    d_setup_randgen<<<gridSize, blockSize>>>(g_d_rand_states,w,h);
    //timer.time_display("Generate Random Field 1");
    //timer.start();
    d_gen_rand_field_scaled<<<gridSize, blockSize>>>(g_d_rand_states,d_disp_vec,d_scale,w,h,disp_mem_w,scale_mem_w);
    //timer.time_display("Generate Random Field 2");
}

//////////////////////////////////////////////////////////////////////////
// compute cost 
// __device__ float _d_euclidean_dist(float4 a, float4 b)
// {
//     float mod = (b.x - a.x) * (b.x - a.x) +
//                 (b.y - a.y) * (b.y - a.y) +
//                 (b.z - a.z) * (b.z - a.z);
// 
//     return mod;
// }

// __device__ float _d_compute_patch_dist_L2(int x1,int y1,int x2,int y2)
// {
//     float cost_val = 0.0f;
//     for (int i = -PATCH_R; i <= PATCH_R; i++)
//     {
// #pragma unroll  //unrolling makes 2x faster, when R=8
//         for (int j = -PATCH_R; j <= PATCH_R; j++)
//         {
//             float4 curPix1 = tex2D(rgbaImg1Tex, x1 + j, y1 + i); //data in frame 1
//             float4 curPix2 = tex2D(rgbaImg2Tex, x2 + j, y2 + i); //data in frame 2
//             cost_val += _d_euclidean_dist(curPix1, curPix2);
//         }
//     }
// 
//     return cost_val;
// }


// original code
// __device__ void _d_bilateral_dist(float& cost, float& weight, float4 c1, float4 c2, float4 a, float4 b, unsigned char s1, unsigned char s2, int dx, int dy)
// {
//     float mod = max(max(abs(b.x - a.x), abs(b.y - a.y)), abs(b.z - a.z));
//     unsigned char census_dist = 0;
//     unsigned char census_diff = s1^s2;
//     while (census_diff) { census_dist++; census_diff &= census_diff-1; };
// 
//     float cost_ad = 1 - __expf(-(mod*mod) / (LAMBDA_AD*LAMBDA_AD));
//     float cost_census = cCensusNegativeGaussian[census_dist];
// 
//     float delta_r1 = max(max(abs(c1.x - a.x), abs(c1.y - a.y)), abs(c1.z - a.z));
//     float delta_r2 = max(max(abs(c2.x - b.x), abs(c2.y - b.y)), abs(c2.z - b.z));
//     
//     float coef_r = __expf(-(delta_r1*delta_r1+delta_r2*delta_r2) / (PM_SIG_R*PM_SIG_R));
//     float coef_s = cSpatialGaussian[dx] * cSpatialGaussian[dy];
// 
//     weight = coef_r * coef_s; //output
//     cost = (cost_ad + cost_census)*weight; //output
// 
//     return;
// }


// optimized register use
__device__ inline void _d_bilateral_dist(float& cost, float& weight, float4 c1, float4 c2, float4 a, float4 b, unsigned char s1, unsigned char s2, int dx, int dy)
{
    float temp;
    unsigned char census_dist = 0;
    unsigned char census_diff = s1^s2;
    while (census_diff) { census_dist++; census_diff &= census_diff-1; };
    cost = max(max(abs(b.x - a.x), abs(b.y - a.y)), abs(b.z - a.z));
    cost = 1 - __expf(-(cost*cost) / (LAMBDA_AD*LAMBDA_AD));
    cost += cCensusNegativeGaussian[census_dist];

    weight = max(max(abs(c1.x - a.x), abs(c1.y - a.y)), abs(c1.z - a.z)); //delta_r1
    weight *= weight;
    temp = max(max(abs(c2.x - b.x), abs(c2.y - b.y)), abs(c2.z - b.z)); //delta_r2 
    temp *= temp;
    weight = __expf(-(weight+temp) / (PM_SIG_R*PM_SIG_R));
    weight *= cSpatialGaussian[dx] * cSpatialGaussian[dy];  //output
    cost *= weight; //output
    return;
}

__device__ inline float _d_compute_patch_dist(int x1,int y1,int x2,int y2)
{
    float4 centerPix1 = tex2D(rgbaImg1Tex, x1, y1); //data in frame 1
    float4 centerPix2 = tex2D(rgbaImg2Tex, x2, y2); //data in frame 2
    float4 curPix1;
    float4 curPix2;
    unsigned char curCensus1;
    unsigned char curCensus2;
    float weight = 0.0f;
    float cost = 0.0f;
    float cost_sum = 0.0f;
    float weight_sum = 0.0f;
    float temp;
//#pragma unroll  //NOTE: unrolling will be slower!!
    for (int i = -PATCH_R; i <= PATCH_R; i+=2) //skip pixels
    {
//#pragma unroll  //NOTE: unrolling will be slower (R=9)!! 
        for (int j = -PATCH_R; j <= PATCH_R; j+=2) //skip pixels
        {
            curPix1 = tex2D(rgbaImg1Tex, x1 + j, y1 + i);
            curPix2 = tex2D(rgbaImg2Tex, x2 + j, y2 + i);
            curCensus1 = tex2D(census1Tex, x1 + j, y1 + i);
            curCensus2 = tex2D(census2Tex, x2 + j, y2 + i);
            
            //compute blf distance
            curCensus1 ^= curCensus2; //unsigned char census_diff = s1^s2;
            curCensus2 = 0; //unsigned char census_dist = 0;
            while (curCensus1) { curCensus2++; curCensus1 &= curCensus1-1;}
            cost = max(max(abs(curPix1.x - curPix2.x), abs(curPix1.y - curPix2.y)), abs(curPix1.z - curPix2.z));
            cost = 1 - __expf(-(cost*cost) / (LAMBDA_AD*LAMBDA_AD));
            cost += cCensusNegativeGaussian[curCensus2];
            weight = max(max(abs(centerPix1.x - curPix1.x), abs(centerPix1.y - curPix1.y)), abs(centerPix1.z - curPix1.z)); //delta_r1
            weight *= weight;
            temp = max(max(abs(centerPix2.x - curPix2.x), abs(centerPix2.y - curPix2.y)), abs(centerPix2.z - curPix2.z)); //delta_r2 
            temp *= temp;
            weight = __expf(-(weight+temp) / (PM_SIG_R*PM_SIG_R));
            weight *= cSpatialGaussian[abs(j)] * cSpatialGaussian[abs(i)];
            cost *= weight;
    
            //_d_bilateral_dist(cost, weight, centerPix1, centerPix2, curPix1, curPix2, curCensus1, curCensus2, abs(j), abs(i));
            cost_sum += cost;
            weight_sum += weight;
        }
    }

    return (cost_sum/weight_sum);
}


// #define  COEF_FL_U_X  0.117f
// #define  COEF_FL_U_Y  0.026f
// #define  COEF_FL_V_X  0.0015f
// #define  COEF_FL_V_Y  0.246f
// 
// #define  COEF_LEFT_U_X  0.505f 
// #define  COEF_LEFT_U_Y  -0.047f
// #define  COEF_LEFT_V_X  -0.128f
// #define  COEF_LEFT_V_Y  0.272f
// 
// #define  COEF_RIGHT_U_X  0.482f 
// #define  COEF_RIGHT_U_Y  0.129f
// #define  COEF_RIGHT_V_X  0.076f
// #define  COEF_RIGHT_V_Y  0.290f

#define  COEF_FL_U_X  0.177f
#define  COEF_FL_U_Y  -0.011f
#define  COEF_FL_V_X  -0.003f
#define  COEF_FL_V_Y  0.301f

#define  COEF_LEFT_U_X  0.125f
#define  COEF_LEFT_U_Y  -0.357f
#define  COEF_LEFT_V_X  0.009f
#define  COEF_LEFT_V_Y  0.308f

#define  COEF_RIGHT_U_X  0.205f
#define  COEF_RIGHT_U_Y  0.370f
#define  COEF_RIGHT_V_X  0.011f
#define  COEF_RIGHT_V_Y  0.296f

__device__ inline float _d_compute_patch_dist_planefitting(int x1,int y1,int x2,int y2)
{
    float4 centerPix1 = tex2D(rgbaImg1Tex, x1, y1); //data in frame 1
    float4 centerPix2 = tex2D(rgbaImg2Tex, x2, y2); //data in frame 2
    float4 curPix1;
    float4 curPix2;
    unsigned char curCensus1;
    unsigned char curCensus2;
    float weight = 0.0f;
    float cost1 = 999999;
    float cost2 = 999999;
    float cost3 = 999999;
    float cost4 = 999999;
    float cost_sum = 0.0f;
    float weight_sum = 0.0f;
    float temp;
    float uu = x2 - x1;
    float vv = y2 - y1;
//#pragma unroll  //NOTE: unrolling will be slower!!
    for (int i = -PATCH_R; i <= PATCH_R; i+=2) //skip pixels
    {
//#pragma unroll  //NOTE: unrolling will be slower (R=9)!! 
        for (int j = -PATCH_R; j <= PATCH_R; j+=2) //skip pixels
        {
            float cx1 = x1+j;
            float cy1 = y1+i;
//             float cx2 = cx1 + uu; //x2+j;
//             float cy2 = cy1 + vv; //y2+i;
            float cx2 = cx1 + uu;
            float cy2 = cy1 + vv;
            curPix1 = tex2D(rgbaImg1Tex, cx1, cy1);
            curPix2 = tex2D(rgbaImg2Tex, cx2, cy2);
            curCensus1 = tex2D(census1Tex, cx1, cy1);
            curCensus2 = tex2D(census2Tex, cx2, cy2);
            
            //compute blf distance
            curCensus1 ^= curCensus2; //unsigned char census_diff = s1^s2;
            curCensus2 = 0; //unsigned char census_dist = 0;
            while (curCensus1) { curCensus2++; curCensus1 &= curCensus1-1;}
            cost1 = max(max(abs(curPix1.x - curPix2.x), abs(curPix1.y - curPix2.y)), abs(curPix1.z - curPix2.z));
            cost1 = 1 - __expf(-(cost1*cost1) / (LAMBDA_AD*LAMBDA_AD));
            cost1 += cCensusNegativeGaussian[curCensus2];
            weight = max(max(abs(centerPix1.x - curPix1.x), abs(centerPix1.y - curPix1.y)), abs(centerPix1.z - curPix1.z)); //delta_r1
            weight *= weight;
            temp = max(max(abs(centerPix2.x - curPix2.x), abs(centerPix2.y - curPix2.y)), abs(centerPix2.z - curPix2.z)); //delta_r2 
            temp *= temp;
            weight = __expf(-(weight+temp) / (PM_SIG_R*PM_SIG_R));
            weight *= cSpatialGaussian[abs(j)] * cSpatialGaussian[abs(i)];
            cost1 *= weight;
    
            //_d_bilateral_dist(cost, weight, centerPix1, centerPix2, curPix1, curPix2, curCensus1, curCensus2, abs(j), abs(i));
            cost_sum += cost1;
            weight_sum += weight;
        }
    }
    cost1 = cost_sum/weight_sum;
    cost_sum = 0.0f;
    weight_sum = 0.0f;
//#pragma unroll  //NOTE: unrolling will be slower!!
    for (int i = -PATCH_R; i <= PATCH_R; i+=2) //skip pixels
    {
//#pragma unroll  //NOTE: unrolling will be slower (R=9)!! 
        for (int j = -PATCH_R; j <= PATCH_R; j+=2) //skip pixels
        {
            float cx1 = x1+j;
            float cy1 = y1+i;
//             float cx2 = cx1 + uu; //x2+j;
//             float cy2 = cy1 + vv; //y2+i;
            float cx2 = cx1 + uu + (j)*COEF_FL_U_X + (i)*COEF_FL_U_Y;
            float cy2 = cy1 + vv + (j)*COEF_FL_V_X + (i)*COEF_FL_V_Y;
            curPix1 = tex2D(rgbaImg1Tex, cx1, cy1);
            curPix2 = tex2D(rgbaImg2Tex, cx2, cy2);
            curCensus1 = tex2D(census1Tex, cx1, cy1);
            curCensus2 = tex2D(census2Tex, cx2, cy2);
            
            //compute blf distance
            curCensus1 ^= curCensus2; //unsigned char census_diff = s1^s2;
            curCensus2 = 0; //unsigned char census_dist = 0;
            while (curCensus1) { curCensus2++; curCensus1 &= curCensus1-1;}
            cost2 = max(max(abs(curPix1.x - curPix2.x), abs(curPix1.y - curPix2.y)), abs(curPix1.z - curPix2.z));
            cost2 = 1 - __expf(-(cost2*cost2) / (LAMBDA_AD*LAMBDA_AD));
            cost2 += cCensusNegativeGaussian[curCensus2];
            weight = max(max(abs(centerPix1.x - curPix1.x), abs(centerPix1.y - curPix1.y)), abs(centerPix1.z - curPix1.z)); //delta_r1
            weight *= weight;
            temp = max(max(abs(centerPix2.x - curPix2.x), abs(centerPix2.y - curPix2.y)), abs(centerPix2.z - curPix2.z)); //delta_r2 
            temp *= temp;
            weight = __expf(-(weight+temp) / (PM_SIG_R*PM_SIG_R));
            weight *= cSpatialGaussian[abs(j)] * cSpatialGaussian[abs(i)];
            cost2 *= weight;
    
            //_d_bilateral_dist(cost, weight, centerPix1, centerPix2, curPix1, curPix2, curCensus1, curCensus2, abs(j), abs(i));
            cost_sum += cost2;
            weight_sum += weight;
        }
    }
    cost2 = cost_sum/weight_sum;
    cost_sum = 0.0f;
    weight_sum = 0.0f;
    
//#pragma unroll  //NOTE: unrolling will be slower!!
    for (int i = -PATCH_R; i <= PATCH_R; i+=2) //skip pixels
    {
//#pragma unroll  //NOTE: unrolling will be slower (R=9)!! 
        for (int j = -PATCH_R; j <= PATCH_R; j+=2) //skip pixels
        {
            float cx1 = x1+j;
            float cy1 = y1+i;
//             float cx2 = cx1 + uu; //x2+j;
//             float cy2 = cy1 + vv; //y2+i;
            float cx2 = cx1 + uu + (j)*COEF_LEFT_U_X + (i)*COEF_LEFT_U_Y;
            float cy2 = cy1 + vv + (j)*COEF_LEFT_V_X + (i)*COEF_LEFT_V_Y;
            curPix1 = tex2D(rgbaImg1Tex, cx1, cy1);
            curPix2 = tex2D(rgbaImg2Tex, cx2, cy2);
            curCensus1 = tex2D(census1Tex, cx1, cy1);
            curCensus2 = tex2D(census2Tex, cx2, cy2);
            
            //compute blf distance
            curCensus1 ^= curCensus2; //unsigned char census_diff = s1^s2;
            curCensus2 = 0; //unsigned char census_dist = 0;
            while (curCensus1) { curCensus2++; curCensus1 &= curCensus1-1;}
            cost3 = max(max(abs(curPix1.x - curPix2.x), abs(curPix1.y - curPix2.y)), abs(curPix1.z - curPix2.z));
            cost3 = 1 - __expf(-(cost3*cost3) / (LAMBDA_AD*LAMBDA_AD));
            cost3 += cCensusNegativeGaussian[curCensus2];
            weight = max(max(abs(centerPix1.x - curPix1.x), abs(centerPix1.y - curPix1.y)), abs(centerPix1.z - curPix1.z)); //delta_r1
            weight *= weight;
            temp = max(max(abs(centerPix2.x - curPix2.x), abs(centerPix2.y - curPix2.y)), abs(centerPix2.z - curPix2.z)); //delta_r2 
            temp *= temp;
            weight = __expf(-(weight+temp) / (PM_SIG_R*PM_SIG_R));
            weight *= cSpatialGaussian[abs(j)] * cSpatialGaussian[abs(i)];
            cost3 *= weight;
    
            //_d_bilateral_dist(cost, weight, centerPix1, centerPix2, curPix1, curPix2, curCensus1, curCensus2, abs(j), abs(i));
            cost_sum += cost3;
            weight_sum += weight;
        }
    }
    cost3 = cost_sum/weight_sum;
    cost_sum = 0.0f;
    weight_sum = 0.0f;
//#pragma unroll  //NOTE: unrolling will be slower!!
    for (int i = -PATCH_R; i <= PATCH_R; i+=2) //skip pixels
    {
//#pragma unroll  //NOTE: unrolling will be slower (R=9)!! 
        for (int j = -PATCH_R; j <= PATCH_R; j+=2) //skip pixels
        {
            float cx1 = x1+j;
            float cy1 = y1+i;
//             float cx2 = cx1 + uu; //x2+j;
//             float cy2 = cy1 + vv; //y2+i;
            float cx2 = cx1 + uu + (j)*COEF_RIGHT_U_X + (i)*COEF_RIGHT_U_Y;
            float cy2 = cy1 + vv + (j)*COEF_RIGHT_V_X + (i)*COEF_RIGHT_V_Y;
            curPix1 = tex2D(rgbaImg1Tex, cx1, cy1);
            curPix2 = tex2D(rgbaImg2Tex, cx2, cy2);
            curCensus1 = tex2D(census1Tex, cx1, cy1);
            curCensus2 = tex2D(census2Tex, cx2, cy2);
            
            //compute blf distance
            curCensus1 ^= curCensus2; //unsigned char census_diff = s1^s2;
            curCensus2 = 0; //unsigned char census_dist = 0;
            while (curCensus1) { curCensus2++; curCensus1 &= curCensus1-1;}
            cost4 = max(max(abs(curPix1.x - curPix2.x), abs(curPix1.y - curPix2.y)), abs(curPix1.z - curPix2.z));
            cost4 = 1 - __expf(-(cost4*cost4) / (LAMBDA_AD*LAMBDA_AD));
            cost4 += cCensusNegativeGaussian[curCensus2];
            weight = max(max(abs(centerPix1.x - curPix1.x), abs(centerPix1.y - curPix1.y)), abs(centerPix1.z - curPix1.z)); //delta_r1
            weight *= weight;
            temp = max(max(abs(centerPix2.x - curPix2.x), abs(centerPix2.y - curPix2.y)), abs(centerPix2.z - curPix2.z)); //delta_r2 
            temp *= temp;
            weight = __expf(-(weight+temp) / (PM_SIG_R*PM_SIG_R));
            weight *= cSpatialGaussian[abs(j)] * cSpatialGaussian[abs(i)];
            cost4 *= weight;
    
            //_d_bilateral_dist(cost, weight, centerPix1, centerPix2, curPix1, curPix2, curCensus1, curCensus2, abs(j), abs(i));
            cost_sum += cost4;
            weight_sum += weight;
        }
    }
    cost4 = cost_sum/weight_sum;
  
    return __min(cost1,__min(cost2, __min(cost3, cost4)));
}

__device__ inline float _d_compute_patch_dist_ad(int x1,int y1,int x2,int y2)
{
    float4 centerPix1 = tex2D(rgbaImg1Tex, x1, y1); //data in frame 1
    float4 centerPix2 = tex2D(rgbaImg2Tex, x2, y2); //data in frame 2
    float4 curPix1;
    float4 curPix2;
    float weight = 0.0f;
    float cost = 0.0f;
    float cost_sum = 0.0f;
    float weight_sum = 0.0f;
    float temp;
//#pragma unroll  //NOTE: unrolling will be slower!!
    for (int i = -PATCH_R; i <= PATCH_R; i+=2) //skip pixels
    {
//#pragma unroll  //NOTE: unrolling will be slower (R=9)!! 
        for (int j = -PATCH_R; j <= PATCH_R; j+=2) //skip pixels
        {
            curPix1 = tex2D(rgbaImg1Tex, x1 + j, y1 + i);
            curPix2 = tex2D(rgbaImg2Tex, x2 + j, y2 + i);
            
            //compute blf distance
            cost = max(max(abs(curPix1.x - curPix2.x), abs(curPix1.y - curPix2.y)), abs(curPix1.z - curPix2.z));
            cost = 1 - __expf(-(cost*cost) / (LAMBDA_AD*LAMBDA_AD));
            weight = max(max(abs(centerPix1.x - curPix1.x), abs(centerPix1.y - curPix1.y)), abs(centerPix1.z - curPix1.z)); //delta_r1
            weight *= weight;
            temp = max(max(abs(centerPix2.x - curPix2.x), abs(centerPix2.y - curPix2.y)), abs(centerPix2.z - curPix2.z)); //delta_r2 
            temp *= temp;
            weight = __expf(-(weight+temp) / (PM_SIG_R*PM_SIG_R));
            weight *= cSpatialGaussian[abs(j)] * cSpatialGaussian[abs(i)];
            cost *= weight;
    
            //_d_bilateral_dist(cost, weight, centerPix1, centerPix2, curPix1, curPix2, curCensus1, curCensus2, abs(j), abs(i));
            cost_sum += cost;
            weight_sum += weight;
        }
    }

    return (cost_sum/weight_sum);
}

__device__ inline float _d_compute_patch_dist_ad_L2(int x1,int y1,int x2,int y2)
{
//     float4 centerPix1 = tex2D(rgbaImg1Tex, x1, y1); //data in frame 1
//     float4 centerPix2 = tex2D(rgbaImg2Tex, x2, y2); //data in frame 2
    float4 curPix1;
    float4 curPix2;
    float weight = 0.0f;
    float cost = 0.0f;
    float cost_sum = 0.0f;
    float weight_sum = 0.0f;
//#pragma unroll  //NOTE: unrolling will be slower!!
    for (int i = -PATCH_R; i <= PATCH_R; i+=2) //skip pixels
    {
//#pragma unroll  //NOTE: unrolling will be slower (R=9)!! 
        for (int j = -PATCH_R; j <= PATCH_R; j+=2) //skip pixels
        {
            curPix1 = tex2D(rgbaImg1Tex, x1 + j, y1 + i);
            curPix2 = tex2D(rgbaImg2Tex, x2 + j, y2 + i);
            
            //compute blf distance
            cost = max(max(abs(curPix1.x - curPix2.x), abs(curPix1.y - curPix2.y)), abs(curPix1.z - curPix2.z));
            cost = 1 - __expf(-(cost*cost) / (LAMBDA_AD*LAMBDA_AD));
            weight = 1.0f;
//             cost *= weight;
    
            cost_sum += cost;
            weight_sum += weight;
        }
    }

    return (cost_sum/weight_sum);
}

__device__ inline float _d_compute_patch_dist_scaled(int x1,int y1,int x2,int y2,float scale)
{
    float4 centerPix1 = tex2D(rgbaImg1Tex, x1, y1); //data in frame 1
    float4 centerPix2 = tex2D(rgbaImg2Tex, x2, y2); //data in frame 2
    float4 curPix1;
    float4 curPix2;
    unsigned char curCensus1;
    unsigned char curCensus2;
    float weight = 0.0f;
    float cost = 0.0f;
    float cost_sum = 0.0f;
    float weight_sum = 0.0f;
    float temp;
//#pragma unroll  //NOTE: unrolling will be slower!!
    for (int i = -PATCH_R; i <= PATCH_R; i+=2) //skip pixels
    {
//#pragma unroll  //NOTE: unrolling will be slower (R=9)!! 
        for (int j = -PATCH_R; j <= PATCH_R; j+=2) //skip pixels
        {
            curPix1 = tex2D(rgbaImg1Tex, x1 + j, y1 + i);
            curPix2 = tex2D(rgbaImg2Tex, x2 + float(j)*scale, y2 + float(i)*scale);
//             curCensus1 = tex2D(census1Tex, x1 + j, y1 + i);
//             curCensus2 = tex2D(census2Tex, x2 + float(j)*scale, y2 + float(i)*scale);
            
            //compute blf distance
//             curCensus1 ^= curCensus2; //unsigned char census_diff = s1^s2;
//             curCensus2 = 0; //unsigned char census_dist = 0;
//             while (curCensus1) { curCensus2++; curCensus1 &= curCensus1-1;}
            cost = max(max(abs(curPix1.x - curPix2.x), abs(curPix1.y - curPix2.y)), abs(curPix1.z - curPix2.z));
            cost = 1 - __expf(-(cost*cost) / (LAMBDA_AD*LAMBDA_AD));
//             cost += cCensusNegativeGaussian[curCensus2];
            weight = max(max(abs(centerPix1.x - curPix1.x), abs(centerPix1.y - curPix1.y)), abs(centerPix1.z - curPix1.z)); //delta_r1
            weight *= weight;
            temp = max(max(abs(centerPix2.x - curPix2.x), abs(centerPix2.y - curPix2.y)), abs(centerPix2.z - curPix2.z)); //delta_r2 
            temp *= temp;
            weight = __expf(-(weight+temp) / (PM_SIG_R*PM_SIG_R));
            weight *= cSpatialGaussian[abs(j)] * cSpatialGaussian[abs(i)];
            cost *= weight;
    
            //_d_bilateral_dist(cost, weight, centerPix1, centerPix2, curPix1, curPix2, curCensus1, curCensus2, abs(j), abs(i));
            cost_sum += cost;
            weight_sum += weight;
        }
    }

    return (cost_sum/weight_sum);
}

__global__ void d_compute_cost_field(float* d_cost, short2* d_disp_vec, int w, int h, size_t cost_mem_w, size_t disp_mem_w)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    
    short2 disp_val = d_disp_vec[id_y*disp_mem_w + id_x];
    
    d_cost[id_y*cost_mem_w + id_x] = _d_compute_patch_dist(id_x,id_y,disp_val.x,disp_val.y);
}

__global__ void d_compute_cost_field_scaled(float* d_cost, short2* d_disp_vec, float* d_scale, int w, int h, size_t cost_mem_w, size_t disp_mem_w, size_t scale_mem_w)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    
    short2 disp_val = d_disp_vec[id_y*disp_mem_w + id_x];
    
    d_cost[id_y*cost_mem_w + id_x] = _d_compute_patch_dist_scaled(id_x,id_y,disp_val.x,disp_val.y,d_scale[id_y*scale_mem_w+id_x]);
}

__global__ void d_compute_cost_field_planefitting(float* d_cost, short2* d_disp_vec, int w, int h, size_t cost_mem_w, size_t disp_mem_w)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    
    short2 disp_val = d_disp_vec[id_y*disp_mem_w + id_x];
    
    d_cost[id_y*cost_mem_w + id_x] = _d_compute_patch_dist_planefitting(id_x,id_y,disp_val.x,disp_val.y);
}


extern "C"
void _initGaussianLookupTable()
{
    float  fGaussian[PATCH_R+1];

    for (int i = 0; i <= PATCH_R; i++)
    {
        fGaussian[i] = expf(-(i*i) / (PM_SIG_S*PM_SIG_S));
    }
    checkCudaErrors(cudaMemcpyToSymbol(cSpatialGaussian, fGaussian, sizeof(float)*(PATCH_R+1)));

    float fNegativeGaussian[CENSUS_MAX_DIFF+1];
    for (int i=0; i<=CENSUS_MAX_DIFF; i++)
    {
        fNegativeGaussian[i] = 1-expf(-float(i*i)/(LAMBDA_CENSUS*CENSUS_MAX_DIFF*LAMBDA_CENSUS*CENSUS_MAX_DIFF));
    }
    checkCudaErrors(cudaMemcpyToSymbol(cCensusNegativeGaussian, fNegativeGaussian, sizeof(float)*(CENSUS_MAX_DIFF+1)));
}

extern "C"
void baoComputeCostField(float* d_cost, short2* d_disp_vec, int w, int h, size_t cost_mem_w, size_t disp_mem_w)
{
    _initGaussianLookupTable();
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    d_compute_cost_field<<<gridSize, blockSize>>>(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w);
}

extern "C"
void baoComputeCostField_Scaled(float* d_cost, short2* d_disp_vec, float* d_scale, int w, int h, size_t cost_mem_w, size_t disp_mem_w, size_t scale_mem_w)
{
    _initGaussianLookupTable();
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    d_compute_cost_field_scaled<<<gridSize, blockSize>>>(d_cost,d_disp_vec,d_scale,w,h,cost_mem_w,disp_mem_w,scale_mem_w);
}

extern "C"
void baoComputeCostField_PlaneFitting(float* d_cost, short2* d_disp_vec, int w, int h, size_t cost_mem_w, size_t disp_mem_w)
{
    _initGaussianLookupTable();
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    d_compute_cost_field_planefitting<<<gridSize, blockSize>>>(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w);
}

//////////////////////////////////////////////////////////////////////////
// propagation step in PatchMatch
#define  PROP_RIM  1 

__global__ void d_neighbor_propagate(float* d_cost, short2* d_disp_vec, int w, int h, size_t cost_mem_w, size_t disp_mem_w)
{
    __shared__ short2 s_disp_vec[BLOCK_DIM_Y+PROP_RIM*2][BLOCK_DIM_X+PROP_RIM*2];
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;

    // copy main data into shared memory
    s_disp_vec[threadIdx.y+PROP_RIM][threadIdx.x+PROP_RIM] = d_disp_vec[id_y*disp_mem_w + id_x];

    // copy upper rim
    if (threadIdx.y == 0) 
    {
        int upper_y = id_y-1;
        if (upper_y < 0) upper_y = 0;
        s_disp_vec[0][threadIdx.x+PROP_RIM] = d_disp_vec[upper_y*disp_mem_w + id_x];
    }

    // copy lower rim
    if (threadIdx.y == BLOCK_DIM_Y-1) 
    {
        int lower_y = id_y+1;
        if (lower_y >= h) lower_y = h-1;
        s_disp_vec[BLOCK_DIM_Y+PROP_RIM*2-1][threadIdx.x+PROP_RIM] = d_disp_vec[lower_y*disp_mem_w + id_x];
    }

    // copy left rim
    if (threadIdx.x == 0) 
    {
        int left_x = id_x-1;
        if (left_x < 0) left_x = 0;
        s_disp_vec[threadIdx.y+PROP_RIM][0] = d_disp_vec[id_y*disp_mem_w + left_x];
    }

    // copy right rim
    if (threadIdx.x == BLOCK_DIM_X-1) 
    {
        int right_x = id_x+1;
        if (right_x >= w) right_x = w-1;
        s_disp_vec[threadIdx.y+PROP_RIM][BLOCK_DIM_X+PROP_RIM*2-1] = d_disp_vec[id_y*disp_mem_w + right_x];
    }

    __syncthreads(); //make sure all data are loaded into shared memory
    
    short2 disp_neighbor[4];
    disp_neighbor[0] = s_disp_vec[threadIdx.y+PROP_RIM-1][threadIdx.x+PROP_RIM]; //upper
    disp_neighbor[1] = s_disp_vec[threadIdx.y+PROP_RIM+1][threadIdx.x+PROP_RIM]; //lower
    disp_neighbor[2] = s_disp_vec[threadIdx.y+PROP_RIM][threadIdx.x+PROP_RIM-1]; //left
    disp_neighbor[3] = s_disp_vec[threadIdx.y+PROP_RIM][threadIdx.x+PROP_RIM+1]; //right

    short2 best_disp = s_disp_vec[threadIdx.y+PROP_RIM][threadIdx.x+PROP_RIM];
    float best_cost = d_cost[id_y*cost_mem_w + id_x];
    
//#pragma unroll
    for (int k=0; k<4; k++)
    {
        float cost_val = _d_compute_patch_dist(id_x,id_y,disp_neighbor[k].x,disp_neighbor[k].y);
        if (cost_val < best_cost)
        {
            best_disp = disp_neighbor[k];
            best_cost = cost_val;
        }
    }

    // copy data to global memory
    d_disp_vec[id_y*cost_mem_w + id_x] = best_disp;
    d_cost[id_y*cost_mem_w + id_x] = best_cost;
}

extern "C"
void baoParallelPropagate(float* d_cost, short2* d_disp_vec, int w, int h, size_t cost_mem_w, size_t disp_mem_w)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    d_neighbor_propagate<<<gridSize, blockSize>>>(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w);
}


//////////////////////////////////////////////////////////////////////////
// propagation step in PatchMatch
__global__ void d_jump_propagate(float* d_cost, short2* d_disp_vec, int w, int h, size_t cost_mem_w, size_t disp_mem_w, int step_size)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;

    short2 best_disp = d_disp_vec[id_y*disp_mem_w + id_x];
    float best_cost = d_cost[id_y*cost_mem_w + id_x];

    // collect neighboring displacement
    short2 neigh_disp_arr[4];
    int neigh_x = id_x;
    int neigh_y = id_y;
    if ((neigh_x=id_x-step_size)>=0) {neigh_disp_arr[0] = d_disp_vec[neigh_y*disp_mem_w + neigh_x]; neigh_disp_arr[0].x -= step_size;}
    else neigh_disp_arr[0] = make_short2(-999,-999);
    if ((neigh_x=id_x+step_size)<w) {neigh_disp_arr[1] = d_disp_vec[neigh_y*disp_mem_w + neigh_x]; neigh_disp_arr[1].x += step_size;}
    else neigh_disp_arr[1] = make_short2(-999,-999);

    neigh_x = id_x;
    if ((neigh_y=id_y-step_size)>=0) {neigh_disp_arr[2] = d_disp_vec[neigh_y*disp_mem_w + neigh_x]; neigh_disp_arr[2].y -= step_size;}
    else neigh_disp_arr[2] = make_short2(-999,-999);
    if ((neigh_y=id_y+step_size)<h) {neigh_disp_arr[3] = d_disp_vec[neigh_y*disp_mem_w + neigh_x]; neigh_disp_arr[3].y += step_size;}
    else neigh_disp_arr[3] = make_short2(-999,-999);
    
    // start to try
//#pragma unroll
    for (int k=0; k<4; k++)
    {
        short2 neigh_disp = neigh_disp_arr[k];
        if (neigh_disp.x<0 || neigh_disp.y<0 || neigh_disp.x>=w || neigh_disp.y>=h) continue;
        float cost_val = _d_compute_patch_dist(id_x,id_y,neigh_disp.x,neigh_disp.y);
        if (cost_val < best_cost)
        {
            best_disp = neigh_disp;
            best_cost = cost_val;
        }
    }

    // copy data to global memory
    d_disp_vec[id_y*disp_mem_w + id_x] = best_disp;
    d_cost[id_y*cost_mem_w + id_x] = best_cost;
}

extern "C"
void baoJumpPropagate(float* d_cost, short2* d_disp_vec, int w, int h, size_t cost_mem_w, size_t disp_mem_w)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    for (int p=0; p<1; p++)
    {
        d_jump_propagate<<<gridSize, blockSize>>>(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w,32);
        d_jump_propagate<<<gridSize, blockSize>>>(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w,16);
        d_jump_propagate<<<gridSize, blockSize>>>(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w,8);
        d_jump_propagate<<<gridSize, blockSize>>>(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w,4);
        d_jump_propagate<<<gridSize, blockSize>>>(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w,2);
        d_jump_propagate<<<gridSize, blockSize>>>(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w,1);
    }
}


//////////////////////////////////////////////////////////////////////////
// propagation step in PatchMatch
#define   LINE_PROP_BLOCKDIM   16

__global__ void d_row_propagate(float* d_cost, short2* d_disp_vec, int w, int h, size_t cost_mem_w, size_t disp_mem_w)
{
    int row_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (row_id >= h) return;

    short2 prev_disp = d_disp_vec[row_id*disp_mem_w];
    for (int i=1; i<w; i++)
    {
        int idx_disp = row_id*disp_mem_w + i;
        int idx_cost = row_id*cost_mem_w + i;
        float cur_best_cost = d_cost[idx_cost];
        prev_disp.x = min(prev_disp.x+1,w-1);
        float cost_val = _d_compute_patch_dist(i,row_id,prev_disp.x,prev_disp.y);
        if (cost_val < cur_best_cost)
        {
            d_disp_vec[idx_disp] = prev_disp;
            d_cost[idx_cost] = cost_val; 
        }
        else
        {
            prev_disp = d_disp_vec[idx_disp];
        }
    }
}

__global__ void d_row_propagate_reverse(float* d_cost, short2* d_disp_vec, int w, int h, size_t cost_mem_w, size_t disp_mem_w)
{
    int row_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (row_id >= h) return;
    
    short2 prev_disp = d_disp_vec[row_id*disp_mem_w + w-1];
    for (int i=w-2; i>=0; i--)
    {
        int idx_disp = row_id*disp_mem_w + i;
        int idx_cost = row_id*cost_mem_w + i;
        float cur_best_cost = d_cost[idx_cost];
        prev_disp.x = max(prev_disp.x-1,0);
        float cost_val = _d_compute_patch_dist(i,row_id,prev_disp.x,prev_disp.y);
        if (cost_val < cur_best_cost)
        {
            d_disp_vec[idx_disp] = prev_disp;
            d_cost[idx_cost] = cost_val; 
        }
        else
        {
            prev_disp = d_disp_vec[idx_disp];
        }
    }
}

__global__ void d_column_propagate(float* d_cost, short2* d_disp_vec, int w, int h, size_t cost_mem_w, size_t disp_mem_w)
{
    int col_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (col_id >= w) return;
    
    short2 prev_disp = d_disp_vec[col_id];
    for (int i=1; i<h; i++)
    {
        int idx_disp = i*disp_mem_w + col_id;
        int idx_cost = i*cost_mem_w + col_id;
        float cur_best_cost = d_cost[idx_cost];
        prev_disp.y = min(prev_disp.y+1,h-1);
        float cost_val = _d_compute_patch_dist(col_id,i,prev_disp.x,prev_disp.y);
        if (cost_val < cur_best_cost)
        {
            d_disp_vec[idx_disp] = prev_disp;
            d_cost[idx_cost] = cost_val; 
        }
        else
        {
            prev_disp = d_disp_vec[idx_disp];
        }
    }
}

__global__ void d_column_propagate_reverse(float* d_cost, short2* d_disp_vec, int w, int h, size_t cost_mem_w, size_t disp_mem_w)
{
    int col_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (col_id >= w) return;

    short2 prev_disp = d_disp_vec[(h-1)*disp_mem_w + col_id];
    for (int i=h-2; i>=0; i--)
    {
        int idx_disp = i*disp_mem_w + col_id;
        int idx_cost = i*cost_mem_w + col_id;
        float cur_best_cost = d_cost[idx_cost];
        prev_disp.y = max(prev_disp.y-1,0);
        float cost_val = _d_compute_patch_dist(col_id,i,prev_disp.x,prev_disp.y);
        if (cost_val < cur_best_cost)
        {
            d_disp_vec[idx_disp] = prev_disp;
            d_cost[idx_cost] = cost_val; 
        }
        else
        {
            prev_disp = d_disp_vec[idx_disp];
        }
    }
}

extern "C"
void baoLinePropagate(float* d_cost, short2* d_disp_vec, int w, int h, size_t cost_mem_w, size_t disp_mem_w)
{
    dim3 gridSize_h(bao_div_ceil(h,LINE_PROP_BLOCKDIM));
    dim3 gridSize_w(bao_div_ceil(w,LINE_PROP_BLOCKDIM));
    dim3 blockSize(LINE_PROP_BLOCKDIM);
    d_row_propagate<<<gridSize_h, blockSize>>>(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w);
    d_column_propagate<<<gridSize_w, blockSize>>>(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w);
    d_row_propagate_reverse<<<gridSize_h, blockSize>>>(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w);
    d_column_propagate_reverse<<<gridSize_w, blockSize>>>(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w);
}


//////////////////////////////////////////////////////////////////////////
// propagation step in PatchMatch
#define   PROP_SEG_LENGTH           10  //the smaller, the faster; the larger, more accurate
#define   ROW_PROP_SEG_BLOCK_DIM_X  16
#define   ROW_PROP_SEG_BLOCK_DIM_Y  16
#define   COL_PROP_SEG_BLOCK_DIM_X  16
#define   COL_PROP_SEG_BLOCK_DIM_Y  16

// __global__ void d_row_propagate_seg_shmem(float* d_cost, short2* d_disp_vec, int w, int h, size_t cost_mem_w, size_t disp_mem_w)
// {
//     // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
//     extern __shared__ unsigned char s_buf[];
//     short2* s_disp = (short2*)s_buf;
//     float* s_cost = (float*)(s_buf + (w*ROW_PROP_SEG_BLOCK_DIM_X)*sizeof(short2));
// 
//     int row_id = threadIdx.x + blockIdx.x * blockDim.x;
//     int seg_id = threadIdx.y + blockIdx.y * blockDim.y;
//     if (row_id >= h) return;
// 
//     // load to shared memory (by warp-size)
//     int num_warps = w/ROW_PROP_SEG_BLOCK_DIM_Y;
//     for (int i=0; i<num_warps; i++)
//     {
//         int temp_idx = i*ROW_PROP_SEG_BLOCK_DIM_Y + threadIdx.y;
//         s_disp[threadIdx.x*disp_mem_w + temp_idx] = d_disp_vec[row_id*disp_mem_w + temp_idx];
//         s_cost[threadIdx.x*cost_mem_w + temp_idx] = d_cost[row_id*cost_mem_w + temp_idx];
//     }
//     for (int i=num_warps*ROW_PROP_SEG_BLOCK_DIM_Y; i<w; i++) //the last warp-sized data
//     {
//         s_disp[threadIdx.x*disp_mem_w + i] = d_disp_vec[row_id*disp_mem_w + i];
//         s_cost[threadIdx.x*cost_mem_w + i] = d_cost[row_id*cost_mem_w + i];
//     }
//     __syncthreads();
// 
//     int start_pix;
//     if (seg_id == 0) start_pix = 0;
//     else start_pix = seg_id*PROP_SEG_LENGTH - 1; //start from the pixel before the segment
//     int end_pix = min(w-1, start_pix+PROP_SEG_LENGTH); //NOTE: the first segment propagate one more pixel than other segments
//     short2 prev_disp = s_disp[threadIdx.x*disp_mem_w + start_pix];
//     for (int i=start_pix+1; i<=end_pix; i++)
//     {
//         int idx_disp = threadIdx.x*disp_mem_w + i;
//         int idx_cost = threadIdx.x*cost_mem_w + i;
//         float cur_best_cost = s_cost[idx_cost];
//         prev_disp.x = min(prev_disp.x+1,w-1);
//         float cost_val = _d_compute_patch_dist(i,row_id,prev_disp.x,prev_disp.y);
//         if (cost_val < cur_best_cost)
//         {
//             s_disp[idx_disp] = prev_disp;
//             s_cost[idx_cost] = cost_val; 
//         }
//         else
//         {
//             prev_disp = s_disp[idx_disp];
//         }
//     }
//     __syncthreads();
// 
//     // save to global memory
//     for (int i=0; i<num_warps; i++)
//     {
//         int temp_idx = i*ROW_PROP_SEG_BLOCK_DIM_Y + threadIdx.y;
//         d_disp_vec[row_id*disp_mem_w + temp_idx] = s_disp[threadIdx.x*disp_mem_w + temp_idx];
//         d_cost[row_id*cost_mem_w + temp_idx] = s_cost[threadIdx.x*cost_mem_w + temp_idx];
//     }
//     for (int i=num_warps*ROW_PROP_SEG_BLOCK_DIM_Y; i<w; i++) //the last warp-sized data
//     {
//         d_disp_vec[row_id*disp_mem_w + i] = s_disp[threadIdx.x*disp_mem_w + i];
//         d_cost[row_id*cost_mem_w + i] = s_cost[threadIdx.x*cost_mem_w + i];
//     }
// }

__global__ void d_row_propagate_seg(float* d_cost, short2* d_disp_vec, int w, int h, size_t cost_mem_w, size_t disp_mem_w)
{
    int row_id = threadIdx.x + blockIdx.x * blockDim.x;
    int seg_id = threadIdx.y + blockIdx.y * blockDim.y;
    if (row_id >= h) return;

    int start_pix;
    if (seg_id == 0) start_pix = 0;
    else start_pix = seg_id*PROP_SEG_LENGTH - 1; //start from the pixel before the segment
    int end_pix = min(w-1, start_pix+PROP_SEG_LENGTH); //NOTE: the first segment propagate one more pixel than other segments
    short2 prev_disp = d_disp_vec[row_id*disp_mem_w + start_pix];
    for (int i=start_pix+1; i<=end_pix; i++)
    {
        int idx_disp = row_id*disp_mem_w + i;
        int idx_cost = row_id*cost_mem_w + i;
        float cur_best_cost = d_cost[idx_cost];
        prev_disp.x = min(prev_disp.x+1,w-1);
        float cost_val = _d_compute_patch_dist(i,row_id,prev_disp.x,prev_disp.y);
        if (cost_val < cur_best_cost)
        {
            d_disp_vec[idx_disp] = prev_disp;
            d_cost[idx_cost] = cost_val; 
        }
        else
        {
            prev_disp = d_disp_vec[idx_disp];
        }
    }
}

__global__ void d_row_propagate_reverse_seg(float* d_cost, short2* d_disp_vec, int w, int h, size_t cost_mem_w, size_t disp_mem_w)
{
    int row_id = threadIdx.x + blockIdx.x * blockDim.x;
    int seg_id = threadIdx.y + blockIdx.y * blockDim.y;
    if (row_id >= h) return;

    int start_pix = (seg_id+1)*PROP_SEG_LENGTH;
    if (start_pix >= w) start_pix = w-1;
    int end_pix = seg_id*PROP_SEG_LENGTH;
    short2 prev_disp = d_disp_vec[row_id*disp_mem_w + start_pix];
    for (int i=start_pix-1; i>=end_pix; i--)
    {
        int idx_disp = row_id*disp_mem_w + i;
        int idx_cost = row_id*cost_mem_w + i;
        float cur_best_cost = d_cost[idx_cost];
        prev_disp.x = max(prev_disp.x-1,0);
        float cost_val = _d_compute_patch_dist(i,row_id,prev_disp.x,prev_disp.y);
        if (cost_val < cur_best_cost)
        {
            d_disp_vec[idx_disp] = prev_disp;
            d_cost[idx_cost] = cost_val; 
        }
        else
        {
            prev_disp = d_disp_vec[idx_disp];
        }
    }
}

__global__ void d_column_propagate_seg(float* d_cost, short2* d_disp_vec, int w, int h, size_t cost_mem_w, size_t disp_mem_w)
{
    int col_id = threadIdx.x + blockIdx.x * blockDim.x;
    int seg_id = threadIdx.y + blockIdx.y * blockDim.y;
    if (col_id >= w) return;
    
    int start_row;
    if (seg_id == 0) start_row = 0;
    else start_row = seg_id*PROP_SEG_LENGTH - 1; //start from the pixel before the segment
    int end_row = min(h-1, start_row+PROP_SEG_LENGTH); //NOTE: the first segment propagate one more pixel than other segments
    short2 prev_disp = d_disp_vec[start_row*disp_mem_w + col_id];
    for (int i=start_row+1; i<=end_row; i++)
    {
        int idx_disp = i*disp_mem_w + col_id;
        int idx_cost = i*cost_mem_w + col_id;
        float cur_best_cost = d_cost[idx_cost];
        prev_disp.y = min(prev_disp.y+1,h-1);
        float cost_val = _d_compute_patch_dist(col_id,i,prev_disp.x,prev_disp.y);
        if (cost_val < cur_best_cost)
        {
            d_disp_vec[idx_disp] = prev_disp;
            d_cost[idx_cost] = cost_val; 
        }
        else
        {
            prev_disp = d_disp_vec[idx_disp];
        }
    }
}

__global__ void d_column_propagate_reverse_seg(float* d_cost, short2* d_disp_vec, int w, int h, size_t cost_mem_w, size_t disp_mem_w)
{
    int col_id = threadIdx.x + blockIdx.x * blockDim.x;
    int seg_id = threadIdx.y + blockIdx.y * blockDim.y;
    if (col_id >= w) return;

    int start_row = (seg_id+1)*PROP_SEG_LENGTH;
    if (start_row >= h) start_row = h-1;
    int end_row = seg_id*PROP_SEG_LENGTH;
    short2 prev_disp = d_disp_vec[start_row*disp_mem_w + col_id];
    for (int i=start_row-1; i>=end_row; i--)
    {
        int idx_disp = i*disp_mem_w + col_id;
        int idx_cost = i*cost_mem_w + col_id;
        float cur_best_cost = d_cost[idx_cost];
        prev_disp.y = max(prev_disp.y-1,0);
        float cost_val = _d_compute_patch_dist(col_id,i,prev_disp.x,prev_disp.y);
        if (cost_val < cur_best_cost)
        {
            d_disp_vec[idx_disp] = prev_disp;
            d_cost[idx_cost] = cost_val; 
        }
        else
        {
            prev_disp = d_disp_vec[idx_disp];
        }
    }
}

extern "C"
void baoSegPropagate(float* d_cost, short2* d_disp_vec, int w, int h, size_t cost_mem_w, size_t disp_mem_w, int iter_idx)
{
    int num_row_seg = bao_div_ceil(w,PROP_SEG_LENGTH);
    int num_col_seg = bao_div_ceil(h,PROP_SEG_LENGTH);
    dim3 gridSize_rowprop(bao_div_ceil(h,ROW_PROP_SEG_BLOCK_DIM_X),bao_div_ceil(num_row_seg,ROW_PROP_SEG_BLOCK_DIM_Y));
    dim3 gridSize_colprop(bao_div_ceil(w,COL_PROP_SEG_BLOCK_DIM_X),bao_div_ceil(num_col_seg,COL_PROP_SEG_BLOCK_DIM_Y));
    dim3 blockSize_rowprop(ROW_PROP_SEG_BLOCK_DIM_X,ROW_PROP_SEG_BLOCK_DIM_Y);
    dim3 blockSize_colprop(COL_PROP_SEG_BLOCK_DIM_X,COL_PROP_SEG_BLOCK_DIM_Y);
    //d_row_propagate_seg_shmem<<<gridSize_rowprop, blockSize_rowprop, w*ROW_PROP_SEG_BLOCK_DIM_X*(sizeof(short2)+sizeof(float))>>>(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w);
    d_row_propagate_seg<<<gridSize_rowprop, blockSize_rowprop>>>(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w);
    d_column_propagate_seg<<<gridSize_colprop, blockSize_colprop>>>(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w);
    d_row_propagate_reverse_seg<<<gridSize_rowprop, blockSize_rowprop>>>(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w);
    d_column_propagate_reverse_seg<<<gridSize_colprop, blockSize_colprop>>>(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w);
}

__global__ void d_row_propagate_seg_scaled(float* d_cost, short2* d_disp_vec, float* d_scale, int w, int h, size_t cost_mem_w, size_t disp_mem_w, size_t scale_mem_w)
{
    int row_id = threadIdx.x + blockIdx.x * blockDim.x;
    int seg_id = threadIdx.y + blockIdx.y * blockDim.y;
    if (row_id >= h) return;

    int start_pix;
    if (seg_id == 0) start_pix = 0;
    else start_pix = seg_id*PROP_SEG_LENGTH - 1; //start from the pixel before the segment
    int end_pix = min(w-1, start_pix+PROP_SEG_LENGTH); //NOTE: the first segment propagate one more pixel than other segments
    short2 prev_disp = d_disp_vec[row_id*disp_mem_w + start_pix];
    float prev_scale = d_scale[row_id*scale_mem_w + start_pix];
    for (int i=start_pix+1; i<=end_pix; i++)
    {
        int idx_disp = row_id*disp_mem_w + i;
        int idx_scale = row_id*scale_mem_w + i;
        int idx_cost = row_id*cost_mem_w + i;
        float cur_best_cost = d_cost[idx_cost];
        prev_disp.x = min(prev_disp.x+1,w-1);
        float cost_val = _d_compute_patch_dist_scaled(i,row_id,prev_disp.x,prev_disp.y,prev_scale);
        if (cost_val < cur_best_cost)
        {
            d_disp_vec[idx_disp] = prev_disp;
            d_scale[idx_scale] = prev_scale;
            d_cost[idx_cost] = prev_scale; 
        }
        else
        {
            prev_disp = d_disp_vec[idx_disp];
            prev_scale = d_scale[idx_scale];
        }
    }
}

__global__ void d_row_propagate_reverse_seg_scaled(float* d_cost, short2* d_disp_vec, float* d_scale, int w, int h, size_t cost_mem_w, size_t disp_mem_w, size_t scale_mem_w)
{
    int row_id = threadIdx.x + blockIdx.x * blockDim.x;
    int seg_id = threadIdx.y + blockIdx.y * blockDim.y;
    if (row_id >= h) return;

    int start_pix = (seg_id+1)*PROP_SEG_LENGTH;
    if (start_pix >= w) start_pix = w-1;
    int end_pix = seg_id*PROP_SEG_LENGTH;
    short2 prev_disp = d_disp_vec[row_id*disp_mem_w + start_pix];
    float prev_scale = d_scale[row_id*scale_mem_w + start_pix];
    for (int i=start_pix-1; i>=end_pix; i--)
    {
        int idx_disp = row_id*disp_mem_w + i;
        int idx_scale = row_id*scale_mem_w + i;
        int idx_cost = row_id*cost_mem_w + i;
        float cur_best_cost = d_cost[idx_cost];
        prev_disp.x = max(prev_disp.x-1,0);
        float cost_val = _d_compute_patch_dist_scaled(i,row_id,prev_disp.x,prev_disp.y,prev_scale);
        if (cost_val < cur_best_cost)
        {
            d_disp_vec[idx_disp] = prev_disp;
            d_scale[idx_scale] = prev_scale;
            d_cost[idx_cost] = cost_val; 
        }
        else
        {
            prev_disp = d_disp_vec[idx_disp];
            prev_scale = d_scale[idx_scale];
        }
    }
}

__global__ void d_column_propagate_seg_scaled(float* d_cost, short2* d_disp_vec, float* d_scale, int w, int h, size_t cost_mem_w, size_t disp_mem_w, size_t scale_mem_w)
{
    int col_id = threadIdx.x + blockIdx.x * blockDim.x;
    int seg_id = threadIdx.y + blockIdx.y * blockDim.y;
    if (col_id >= w) return;
    
    int start_row;
    if (seg_id == 0) start_row = 0;
    else start_row = seg_id*PROP_SEG_LENGTH - 1; //start from the pixel before the segment
    int end_row = min(h-1, start_row+PROP_SEG_LENGTH); //NOTE: the first segment propagate one more pixel than other segments
    short2 prev_disp = d_disp_vec[start_row*disp_mem_w + col_id];
    float prev_scale = d_scale[start_row*scale_mem_w + col_id];
    for (int i=start_row+1; i<=end_row; i++)
    {
        int idx_disp = i*disp_mem_w + col_id;
        int idx_scale = i*scale_mem_w + col_id;
        int idx_cost = i*cost_mem_w + col_id;
        float cur_best_cost = d_cost[idx_cost];
        prev_disp.y = min(prev_disp.y+1,h-1);
        float cost_val = _d_compute_patch_dist_scaled(col_id,i,prev_disp.x,prev_disp.y,prev_scale);
        if (cost_val < cur_best_cost)
        {
            d_disp_vec[idx_disp] = prev_disp;
            d_scale[idx_scale] = prev_scale;
            d_cost[idx_cost] = cost_val; 
        }
        else
        {
            prev_disp = d_disp_vec[idx_disp];
            prev_scale = d_scale[idx_scale];
        }
    }
}

__global__ void d_column_propagate_reverse_seg_scaled(float* d_cost, short2* d_disp_vec, float* d_scale, int w, int h, size_t cost_mem_w, size_t disp_mem_w, size_t scale_mem_w)
{
    int col_id = threadIdx.x + blockIdx.x * blockDim.x;
    int seg_id = threadIdx.y + blockIdx.y * blockDim.y;
    if (col_id >= w) return;

    int start_row = (seg_id+1)*PROP_SEG_LENGTH;
    if (start_row >= h) start_row = h-1;
    int end_row = seg_id*PROP_SEG_LENGTH;
    short2 prev_disp = d_disp_vec[start_row*disp_mem_w + col_id];
    float prev_scale = d_scale[start_row*scale_mem_w + col_id];
    for (int i=start_row-1; i>=end_row; i--)
    {
        int idx_disp = i*disp_mem_w + col_id;
        int idx_scale = i*scale_mem_w + col_id;
        int idx_cost = i*cost_mem_w + col_id;
        float cur_best_cost = d_cost[idx_cost];
        prev_disp.y = max(prev_disp.y-1,0);
        float cost_val = _d_compute_patch_dist_scaled(col_id,i,prev_disp.x,prev_disp.y,prev_scale);
        if (cost_val < cur_best_cost)
        {
            d_disp_vec[idx_disp] = prev_disp;
            d_scale[idx_scale] = prev_scale;
            d_cost[idx_cost] = cost_val; 
        }
        else
        {
            prev_disp = d_disp_vec[idx_disp];
            prev_scale = d_scale[idx_scale];
        }
    }
}

extern "C"
void baoSegPropagate_Scaled(float* d_cost, short2* d_disp_vec, float* d_scale, int w, int h, size_t cost_mem_w, size_t disp_mem_w, size_t scale_mem_w, int iter_idx)
{
    int num_row_seg = bao_div_ceil(w,PROP_SEG_LENGTH);
    int num_col_seg = bao_div_ceil(h,PROP_SEG_LENGTH);
    dim3 gridSize_rowprop(bao_div_ceil(h,ROW_PROP_SEG_BLOCK_DIM_X),bao_div_ceil(num_row_seg,ROW_PROP_SEG_BLOCK_DIM_Y));
    dim3 gridSize_colprop(bao_div_ceil(w,COL_PROP_SEG_BLOCK_DIM_X),bao_div_ceil(num_col_seg,COL_PROP_SEG_BLOCK_DIM_Y));
    dim3 blockSize_rowprop(ROW_PROP_SEG_BLOCK_DIM_X,ROW_PROP_SEG_BLOCK_DIM_Y);
    dim3 blockSize_colprop(COL_PROP_SEG_BLOCK_DIM_X,COL_PROP_SEG_BLOCK_DIM_Y);
    //d_row_propagate_seg_shmem<<<gridSize_rowprop, blockSize_rowprop, w*ROW_PROP_SEG_BLOCK_DIM_X*(sizeof(short2)+sizeof(float))>>>(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w);
    d_row_propagate_seg_scaled<<<gridSize_rowprop, blockSize_rowprop>>>(d_cost,d_disp_vec,d_scale,w,h,cost_mem_w,disp_mem_w,scale_mem_w);
    d_column_propagate_seg_scaled<<<gridSize_colprop, blockSize_colprop>>>(d_cost,d_disp_vec,d_scale,w,h,cost_mem_w,disp_mem_w,scale_mem_w);
    d_row_propagate_reverse_seg_scaled<<<gridSize_rowprop, blockSize_rowprop>>>(d_cost,d_disp_vec,d_scale,w,h,cost_mem_w,disp_mem_w,scale_mem_w);
    d_column_propagate_reverse_seg_scaled<<<gridSize_colprop, blockSize_colprop>>>(d_cost,d_disp_vec,d_scale,w,h,cost_mem_w,disp_mem_w,scale_mem_w);
}


__global__ void d_row_propagate_seg_planefitting(float* d_cost, short2* d_disp_vec, int w, int h, size_t cost_mem_w, size_t disp_mem_w)
{
    int row_id = threadIdx.x + blockIdx.x * blockDim.x;
    int seg_id = threadIdx.y + blockIdx.y * blockDim.y;
    if (row_id >= h) return;

    int start_pix;
    if (seg_id == 0) start_pix = 0;
    else start_pix = seg_id*PROP_SEG_LENGTH - 1; //start from the pixel before the segment
    int end_pix = min(w-1, start_pix+PROP_SEG_LENGTH); //NOTE: the first segment propagate one more pixel than other segments
    short2 prev_disp = d_disp_vec[row_id*disp_mem_w + start_pix];
    for (int i=start_pix+1; i<=end_pix; i++)
    {
        int idx_disp = row_id*disp_mem_w + i;
        int idx_cost = row_id*cost_mem_w + i;
        float cur_best_cost = d_cost[idx_cost];
        prev_disp.x = min(prev_disp.x+1,w-1);
        float cost_val = _d_compute_patch_dist_planefitting(i,row_id,prev_disp.x,prev_disp.y);
        if (cost_val < cur_best_cost)
        {
            d_disp_vec[idx_disp] = prev_disp;
            d_cost[idx_cost] = cost_val; 
        }
        else
        {
            prev_disp = d_disp_vec[idx_disp];
        }
    }
}

__global__ void d_row_propagate_reverse_seg_planefitting(float* d_cost, short2* d_disp_vec, int w, int h, size_t cost_mem_w, size_t disp_mem_w)
{
    int row_id = threadIdx.x + blockIdx.x * blockDim.x;
    int seg_id = threadIdx.y + blockIdx.y * blockDim.y;
    if (row_id >= h) return;

    int start_pix = (seg_id+1)*PROP_SEG_LENGTH;
    if (start_pix >= w) start_pix = w-1;
    int end_pix = seg_id*PROP_SEG_LENGTH;
    short2 prev_disp = d_disp_vec[row_id*disp_mem_w + start_pix];
    for (int i=start_pix-1; i>=end_pix; i--)
    {
        int idx_disp = row_id*disp_mem_w + i;
        int idx_cost = row_id*cost_mem_w + i;
        float cur_best_cost = d_cost[idx_cost];
        prev_disp.x = max(prev_disp.x-1,0);
        float cost_val = _d_compute_patch_dist_planefitting(i,row_id,prev_disp.x,prev_disp.y);
        if (cost_val < cur_best_cost)
        {
            d_disp_vec[idx_disp] = prev_disp;
            d_cost[idx_cost] = cost_val; 
        }
        else
        {
            prev_disp = d_disp_vec[idx_disp];
        }
    }
}

__global__ void d_column_propagate_seg_planefitting(float* d_cost, short2* d_disp_vec, int w, int h, size_t cost_mem_w, size_t disp_mem_w)
{
    int col_id = threadIdx.x + blockIdx.x * blockDim.x;
    int seg_id = threadIdx.y + blockIdx.y * blockDim.y;
    if (col_id >= w) return;
    
    int start_row;
    if (seg_id == 0) start_row = 0;
    else start_row = seg_id*PROP_SEG_LENGTH - 1; //start from the pixel before the segment
    int end_row = min(h-1, start_row+PROP_SEG_LENGTH); //NOTE: the first segment propagate one more pixel than other segments
    short2 prev_disp = d_disp_vec[start_row*disp_mem_w + col_id];
    for (int i=start_row+1; i<=end_row; i++)
    {
        int idx_disp = i*disp_mem_w + col_id;
        int idx_cost = i*cost_mem_w + col_id;
        float cur_best_cost = d_cost[idx_cost];
        prev_disp.y = min(prev_disp.y+1,h-1);
        float cost_val = _d_compute_patch_dist_planefitting(col_id,i,prev_disp.x,prev_disp.y);
        if (cost_val < cur_best_cost)
        {
            d_disp_vec[idx_disp] = prev_disp;
            d_cost[idx_cost] = cost_val; 
        }
        else
        {
            prev_disp = d_disp_vec[idx_disp];
        }
    }
}

__global__ void d_column_propagate_reverse_seg_planefitting(float* d_cost, short2* d_disp_vec, int w, int h, size_t cost_mem_w, size_t disp_mem_w)
{
    int col_id = threadIdx.x + blockIdx.x * blockDim.x;
    int seg_id = threadIdx.y + blockIdx.y * blockDim.y;
    if (col_id >= w) return;

    int start_row = (seg_id+1)*PROP_SEG_LENGTH;
    if (start_row >= h) start_row = h-1;
    int end_row = seg_id*PROP_SEG_LENGTH;
    short2 prev_disp = d_disp_vec[start_row*disp_mem_w + col_id];
    for (int i=start_row-1; i>=end_row; i--)
    {
        int idx_disp = i*disp_mem_w + col_id;
        int idx_cost = i*cost_mem_w + col_id;
        float cur_best_cost = d_cost[idx_cost];
        prev_disp.y = max(prev_disp.y-1,0);
        float cost_val = _d_compute_patch_dist_planefitting(col_id,i,prev_disp.x,prev_disp.y);
        if (cost_val < cur_best_cost)
        {
            d_disp_vec[idx_disp] = prev_disp;
            d_cost[idx_cost] = cost_val; 
        }
        else
        {
            prev_disp = d_disp_vec[idx_disp];
        }
    }
}

extern "C"
void baoSegPropagate_PlaneFitting(float* d_cost, short2* d_disp_vec, int w, int h, size_t cost_mem_w, size_t disp_mem_w, int iter_idx)
{
    int num_row_seg = bao_div_ceil(w,PROP_SEG_LENGTH);
    int num_col_seg = bao_div_ceil(h,PROP_SEG_LENGTH);
    dim3 gridSize_rowprop(bao_div_ceil(h,ROW_PROP_SEG_BLOCK_DIM_X),bao_div_ceil(num_row_seg,ROW_PROP_SEG_BLOCK_DIM_Y));
    dim3 gridSize_colprop(bao_div_ceil(w,COL_PROP_SEG_BLOCK_DIM_X),bao_div_ceil(num_col_seg,COL_PROP_SEG_BLOCK_DIM_Y));
    dim3 blockSize_rowprop(ROW_PROP_SEG_BLOCK_DIM_X,ROW_PROP_SEG_BLOCK_DIM_Y);
    dim3 blockSize_colprop(COL_PROP_SEG_BLOCK_DIM_X,COL_PROP_SEG_BLOCK_DIM_Y);
    //d_row_propagate_seg_shmem<<<gridSize_rowprop, blockSize_rowprop, w*ROW_PROP_SEG_BLOCK_DIM_X*(sizeof(short2)+sizeof(float))>>>(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w);
    d_row_propagate_seg_planefitting<<<gridSize_rowprop, blockSize_rowprop>>>(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w);
    d_column_propagate_seg_planefitting<<<gridSize_colprop, blockSize_colprop>>>(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w);
    d_row_propagate_reverse_seg_planefitting<<<gridSize_rowprop, blockSize_rowprop>>>(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w);
    d_column_propagate_reverse_seg_planefitting<<<gridSize_colprop, blockSize_colprop>>>(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w);
}


//////////////////////////////////////////////////////////////////////////
// random guess step in PatchMatch

// __global__ void d_update_random_guess_para(curandState* randStateArr, float* d_cost, short2* d_disp_vec, int w, int h, size_t cost_mem_w, size_t disp_mem_w)
// {
//     int id_x = threadIdx.x + blockIdx.x * blockDim.x;
//     int id_y = threadIdx.y + blockIdx.y * blockDim.y;
//     if (id_x >= w || id_y >= h) return;
//     
//     short2 best_disp = d_disp_vec[id_y*disp_mem_w + id_x];
//     float best_cost = d_cost[id_y*cost_mem_w + id_x];
//     short2 disp_rand_guess[NUM_RAND_GUESS];
// 
//     // produce random location
//     int block_id = blockIdx.y * gridDim.x + blockIdx.x; //for RNG (random number generator)
//     curandState localState = randStateArr[block_id];
//     unsigned int rdn1,rdn2;
//     int mag = max(h,w);
// 
// #pragma unroll
//     for (int k=0; k<NUM_RAND_GUESS; k++)
//     {
//         /* Sampling window */
//         rdn1 = curand(&localState);
//         rdn2 = curand(&localState);
//         int xmin = max(best_disp.x-mag,0), xmax = min(best_disp.x+mag+1,w+1);
//         int ymin = max(best_disp.y-mag,0), ymax = min(best_disp.y+mag+1,h+1);
//         disp_rand_guess[k].x = short(xmin + rdn1%(xmax-xmin));
//         disp_rand_guess[k].y = short(ymin + rdn2%(ymax-ymin));
//         mag /= 2; //shrink the sampling window
//     }
//     randStateArr[block_id] = localState; //write back the rand state for next time use
//     
//     // start to try
// #pragma unroll //outer loop unrolling does not matter
//     for (int k=0; k<NUM_RAND_GUESS; k++)
//     {
//         float cost_val = _d_compute_patch_dist(id_x,id_y,disp_rand_guess[k].x,disp_rand_guess[k].y);
//         if (cost_val < best_cost)
//         {
//             best_disp = disp_rand_guess[k];
//             best_cost = cost_val;
//         }
//     }
// 
//     // copy data to global memory
//     d_disp_vec[id_y*disp_mem_w + id_x] = best_disp;
//     d_cost[id_y*cost_mem_w + id_x] = best_cost;
// }

__global__ void d_update_random_guess(curandState* randStateArr, float* d_cost, short2* d_disp_vec, int w, int h, size_t cost_mem_w, size_t disp_mem_w)
{
    __shared__ short2 s_rand_disp_vec[BLOCK_DIM_Y][BLOCK_DIM_X];
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    
    short2 best_disp = d_disp_vec[id_y*disp_mem_w + id_x];
    float best_cost = d_cost[id_y*cost_mem_w + id_x];
    short2 disp_rand_guess[NUM_RAND_GUESS];

    // produce random location
    int block_id = blockIdx.y * gridDim.x + blockIdx.x; //for RNG (random number generator)
    curandState localState = randStateArr[block_id];
    unsigned int rdn1,rdn2;
    int mag = SEARCH_RANGE; //restrict search range smaller, ori: max(h,w)

//#pragma unroll
    for (int k=0; k<NUM_RAND_GUESS; k++)
    {
        if (threadIdx.x == 0 && threadIdx.y == 0)  //only one thread in each block generate the random numbers! 
        {
            for (int i=0; i<BLOCK_DIM_Y; i++) 
            {
                //#pragma unroll
                for (int j=0; j<BLOCK_DIM_X; j++)
                {
                    rdn1 = curand(&localState);
                    rdn2 = curand(&localState);
                    /* Store results into shared memory*/
                    s_rand_disp_vec[i][j].x = short(rdn1);
                    s_rand_disp_vec[i][j].y = short(rdn2);
                }
            }
        }
        __syncthreads();
        /* Sampling window */
        //rdn1 = curand(&localState);
        //rdn2 = curand(&localState);
        rdn1 = s_rand_disp_vec[threadIdx.y][threadIdx.x].x;
        rdn2 = s_rand_disp_vec[threadIdx.y][threadIdx.x].y;
        short xmin = max(best_disp.x-mag,0), xmax = min(best_disp.x+mag+1,w+1);
        short ymin = max(best_disp.y-mag,0), ymax = min(best_disp.y+mag+1,h+1);
        disp_rand_guess[k].x = short(xmin + rdn1%(xmax-xmin));
        disp_rand_guess[k].y = short(ymin + rdn2%(ymax-ymin));
        if (mag/2 >= SEARCH_RADIUS_MIN) mag /= 2; //shrink the sampling window
    }
    
    if (threadIdx.x == 0 && threadIdx.y == 0) randStateArr[block_id] = localState; //write back the rand state for next time use
    
    // start to try
//#pragma unroll  //NOTE: outer loop unrolling will cause slower!!
    for (int k=0; k<NUM_RAND_GUESS; k++)
    {
        float cost_val = _d_compute_patch_dist(id_x,id_y,disp_rand_guess[k].x,disp_rand_guess[k].y);
        if (cost_val < best_cost)
        {
            best_disp = disp_rand_guess[k];
            best_cost = cost_val;
        }
    }

    // copy data to global memory
    d_disp_vec[id_y*disp_mem_w + id_x] = best_disp;
    d_cost[id_y*cost_mem_w + id_x] = best_cost;

    return;
}

extern "C"
void baoRandomSearch(float* d_cost, short2* d_disp_vec, int w, int h, size_t cost_mem_w, size_t disp_mem_w)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    d_update_random_guess<<<gridSize, blockSize>>>(g_d_rand_states,d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w);
}

__global__ void d_update_random_guess_scaled(curandState* randStateArr, float* d_cost, short2* d_disp_vec, float* d_scale, int w, int h, size_t cost_mem_w, size_t disp_mem_w, size_t scale_mem_w)
{
    __shared__ short2 s_rand_disp_vec[BLOCK_DIM_Y][BLOCK_DIM_X];
    __shared__ float s_rand_scale[BLOCK_DIM_Y][BLOCK_DIM_X];
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    
    short2 best_disp = d_disp_vec[id_y*disp_mem_w + id_x];
    float best_scale = d_scale[id_y*scale_mem_w + id_x];
    float best_cost = d_cost[id_y*cost_mem_w + id_x];
    short2 disp_rand_guess[NUM_RAND_GUESS];
    float  scale_rand_guess[NUM_RAND_GUESS];

    // produce random location
    int block_id = blockIdx.y * gridDim.x + blockIdx.x; //for RNG (random number generator)
    curandState localState = randStateArr[block_id];
    unsigned int rdn1,rdn2;
    int mag = SEARCH_RANGE; //restrict search range smaller, ori: max(h,w)

//#pragma unroll
    for (int k=0; k<NUM_RAND_GUESS; k++)
    {
        if (threadIdx.x == 0 && threadIdx.y == 0)  //only one thread in each block generate the random numbers! 
        {
            for (int i=0; i<BLOCK_DIM_Y; i++) 
            {
                //#pragma unroll
                for (int j=0; j<BLOCK_DIM_X; j++)
                {
                    rdn1 = curand(&localState);
                    rdn2 = curand(&localState);
                    /* Store results into shared memory*/
                    s_rand_disp_vec[i][j].x = short(rdn1);
                    s_rand_disp_vec[i][j].y = short(rdn2);
                    s_rand_scale[i][j] = float((10 + ((rdn2%PM_SCALE_RANGE)-PM_SCALE_MIN))/float(10.0f)); //0.9~1.3
                }
            }
        }
        __syncthreads();
        /* Sampling window */
        //rdn1 = curand(&localState);
        //rdn2 = curand(&localState);
        rdn1 = s_rand_disp_vec[threadIdx.y][threadIdx.x].x;
        rdn2 = s_rand_disp_vec[threadIdx.y][threadIdx.x].y;
        short xmin = max(best_disp.x-mag,0), xmax = min(best_disp.x+mag+1,w+1);
        short ymin = max(best_disp.y-mag,0), ymax = min(best_disp.y+mag+1,h+1);
        disp_rand_guess[k].x = short(xmin + rdn1%(xmax-xmin));
        disp_rand_guess[k].y = short(ymin + rdn2%(ymax-ymin));
        scale_rand_guess[k] = s_rand_scale[threadIdx.y][threadIdx.x];
        if (mag/2 >= SEARCH_RADIUS_MIN) mag /= 2; //shrink the sampling window
    }
    
    if (threadIdx.x == 0 && threadIdx.y == 0) randStateArr[block_id] = localState; //write back the rand state for next time use
    
    // start to try
//#pragma unroll  //NOTE: outer loop unrolling will cause slower!!
    for (int k=0; k<NUM_RAND_GUESS; k++)
    {
        float cost_val = _d_compute_patch_dist_scaled(id_x,id_y,disp_rand_guess[k].x,disp_rand_guess[k].y,scale_rand_guess[k]);
        if (cost_val < best_cost)
        {
            best_disp = disp_rand_guess[k];
            best_scale = scale_rand_guess[k];
            best_cost = cost_val;
        }
    }

    // copy data to global memory
    d_disp_vec[id_y*disp_mem_w + id_x] = best_disp;
    d_scale[id_y*scale_mem_w + id_x] = best_scale;
    d_cost[id_y*cost_mem_w + id_x] = best_cost;

    return;
}

extern "C"
void baoRandomSearch_Scaled(float* d_cost, short2* d_disp_vec, float* d_scale, int w, int h, size_t cost_mem_w, size_t disp_mem_w, size_t scale_mem_w)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    d_update_random_guess_scaled<<<gridSize, blockSize>>>(g_d_rand_states,d_cost,d_disp_vec,d_scale,w,h,cost_mem_w,disp_mem_w,scale_mem_w);
}


__global__ void d_update_random_guess_planefitting(curandState* randStateArr, float* d_cost, short2* d_disp_vec, int w, int h, size_t cost_mem_w, size_t disp_mem_w)
{
    __shared__ short2 s_rand_disp_vec[BLOCK_DIM_Y][BLOCK_DIM_X];
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    
    short2 best_disp = d_disp_vec[id_y*disp_mem_w + id_x];
    float best_cost = d_cost[id_y*cost_mem_w + id_x];
    short2 disp_rand_guess[NUM_RAND_GUESS];

    // produce random location
    int block_id = blockIdx.y * gridDim.x + blockIdx.x; //for RNG (random number generator)
    curandState localState = randStateArr[block_id];
    unsigned int rdn1,rdn2;
    int mag = SEARCH_RANGE; //restrict search range smaller, ori: max(h,w)

//#pragma unroll
    for (int k=0; k<NUM_RAND_GUESS; k++)
    {
        if (threadIdx.x == 0 && threadIdx.y == 0)  //only one thread in each block generate the random numbers! 
        {
            for (int i=0; i<BLOCK_DIM_Y; i++) 
            {
                //#pragma unroll
                for (int j=0; j<BLOCK_DIM_X; j++)
                {
                    rdn1 = curand(&localState);
                    rdn2 = curand(&localState);
                    /* Store results into shared memory*/
                    s_rand_disp_vec[i][j].x = short(rdn1);
                    s_rand_disp_vec[i][j].y = short(rdn2);
                }
            }
        }
        __syncthreads();
        /* Sampling window */
        //rdn1 = curand(&localState);
        //rdn2 = curand(&localState);
        rdn1 = s_rand_disp_vec[threadIdx.y][threadIdx.x].x;
        rdn2 = s_rand_disp_vec[threadIdx.y][threadIdx.x].y;
        short xmin = max(best_disp.x-mag,0), xmax = min(best_disp.x+mag+1,w+1);
        short ymin = max(best_disp.y-mag,0), ymax = min(best_disp.y+mag+1,h+1);
        disp_rand_guess[k].x = short(xmin + rdn1%(xmax-xmin));
        disp_rand_guess[k].y = short(ymin + rdn2%(ymax-ymin));
        if (mag/2 >= SEARCH_RADIUS_MIN) mag /= 2; //shrink the sampling window
    }
    
    if (threadIdx.x == 0 && threadIdx.y == 0) randStateArr[block_id] = localState; //write back the rand state for next time use
    
    // start to try
//#pragma unroll  //NOTE: outer loop unrolling will cause slower!!
    for (int k=0; k<NUM_RAND_GUESS; k++)
    {
        float cost_val = _d_compute_patch_dist_planefitting(id_x,id_y,disp_rand_guess[k].x,disp_rand_guess[k].y);
        if (cost_val < best_cost)
        {
            best_disp = disp_rand_guess[k];
            best_cost = cost_val;
        }
    }

    // copy data to global memory
    d_disp_vec[id_y*disp_mem_w + id_x] = best_disp;
    d_cost[id_y*cost_mem_w + id_x] = best_cost;

    return;
}

extern "C"
void baoRandomSearch_PlaneFitting(float* d_cost, short2* d_disp_vec, int w, int h, size_t cost_mem_w, size_t disp_mem_w)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    d_update_random_guess_planefitting<<<gridSize, blockSize>>>(g_d_rand_states,d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w);
}

//////////////////////////////////////////////////////////////////////////
// interface of the whole PatchMatch algorithm
extern "C"
void baoCudaPatchMatch(short2* d_disp_vec, float* d_cost, uchar4* d_img1, uchar4* d_img2, unsigned char* d_census1, unsigned char* d_census2, int w, int h, size_t img_pitch, size_t cost_pitch, size_t disp_pitch, size_t census_pitch)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);

    // allocate memory for RNG
    checkCudaErrors(cudaMalloc(&g_d_rand_states, gridSize.x*gridSize.y*sizeof(curandState))); //one RNG per block, (allocating memory takes 1ms+)

    // bind the array to the texture
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    checkCudaErrors(cudaBindTexture2D(0, rgbaImg1Tex, d_img1, desc, w, h, img_pitch));
    checkCudaErrors(cudaBindTexture2D(0, rgbaImg2Tex, d_img2, desc, w, h, img_pitch));
    
    //bind the census with texture memory
    census1Tex.filterMode = cudaFilterModePoint;
    census1Tex.normalized = false;
    census2Tex.filterMode = cudaFilterModePoint;
    census2Tex.normalized = false;
    cudaChannelFormatDesc desc_census = cudaCreateChannelDesc<unsigned char>();
    checkCudaErrors(cudaBindTexture2D(0, census1Tex, d_census1, desc_census, w, h, census_pitch));
    checkCudaErrors(cudaBindTexture2D(0, census2Tex, d_census2, desc_census, w, h, census_pitch));
//     getLastCudaError("Census Binding FAILED");

    size_t cost_mem_w = cost_pitch/sizeof(float);
    size_t disp_mem_w = disp_pitch/sizeof(short2);

    // init random NNF
//     bao_timer_gpu timer;
//     timer.start();
    baoGenerateRandomField(d_disp_vec,w,h,disp_mem_w); //5ms, R=9
//     timer.time_display(">>>PM: Init Random Field");
//     getLastCudaError("PM: Init Random Field FAILED");
    
    // calc init cost
//     timer.start();
    baoComputeCostField(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w); //5ms, R=9
//     timer.time_display(">>>PM: Init Cost Field");
//     getLastCudaError("PM: Init Cost Field FAILED");

    // iterated updating: random search and propagation
//     timer.start();
    for (int i=0; i<NUM_ITER; i++)
    {
//         for (int p=0; p<10; p++) 
//         {
//             //if (i==4 && p==4) timer.start();
//             baoParallelPropagate(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w); //10ms, after several iterations
//             //if (i==4 && p==4) timer.time_display("PM: One Propagate Step");
//         }
//         if (i==4) timer.start();
        //baoLinePropagate(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w); //105ms, R=9
        baoSegPropagate(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w,i);
        //baoJumpPropagate(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w);
//         if (i==4) timer.time_display(">>>>>>PM: Propagate Step");
//         getLastCudaError("PM: Propagate Step FAILED");

//         if (i==4) timer.start();
        baoRandomSearch(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w); //56ms, after several iterations, R=9
//         if (i==4) timer.time_display(">>>>>>PM: Random Search Step");
//         getLastCudaError("PM: Random Search Step FAILED");
    }
//     timer.time_display(">>>PM: Main Body");

    // free memory of RNG
    checkCudaErrors(cudaFree(g_d_rand_states));
}

extern "C"
void baoCudaPatchMatch_Scaled(short2* d_disp_vec, float* d_scale, float* d_cost, uchar4* d_img1, uchar4* d_img2, unsigned char* d_census1, unsigned char* d_census2, int w, int h, size_t img_pitch, size_t cost_pitch, size_t disp_pitch, size_t scale_pitch, size_t census_pitch)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);

    // allocate memory for RNG
    checkCudaErrors(cudaMalloc(&g_d_rand_states, gridSize.x*gridSize.y*sizeof(curandState))); //one RNG per block, (allocating memory takes 1ms+)

    // bind the array to the texture
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    checkCudaErrors(cudaBindTexture2D(0, rgbaImg1Tex, d_img1, desc, w, h, img_pitch));
    checkCudaErrors(cudaBindTexture2D(0, rgbaImg2Tex, d_img2, desc, w, h, img_pitch));
    
//     //bind the census with texture memory
//     census1Tex.filterMode = cudaFilterModePoint;
//     census1Tex.normalized = false;
//     census2Tex.filterMode = cudaFilterModePoint;
//     census2Tex.normalized = false;
//     cudaChannelFormatDesc desc_census = cudaCreateChannelDesc<unsigned char>();
//     checkCudaErrors(cudaBindTexture2D(0, census1Tex, d_census1, desc_census, w, h, census_pitch));
//     checkCudaErrors(cudaBindTexture2D(0, census2Tex, d_census2, desc_census, w, h, census_pitch));
//     getLastCudaError("Census Binding FAILED");

    size_t cost_mem_w = cost_pitch/sizeof(float);
    size_t disp_mem_w = disp_pitch/sizeof(short2);
    size_t scale_mem_w = scale_pitch/sizeof(float);

    // init random NNF
//     bao_timer_gpu timer;
//     timer.start();
    baoGenerateRandomField_Scaled(d_disp_vec,d_scale,w,h,disp_mem_w,scale_mem_w); //5ms, R=9
//     timer.time_display("PM: Init Random Field");
//     getLastCudaError("PM: Init Random Field FAILED");
    
    // calc init cost
//     timer.start();
    baoComputeCostField_Scaled(d_cost,d_disp_vec,d_scale,w,h,cost_mem_w,disp_mem_w,scale_mem_w); //5ms, R=9
//     timer.time_display("PM: Init Cost Field");
//     getLastCudaError("PM: Init Cost Field FAILED");

    // iterated updating: random search and propagation
//     timer.start();
    for (int i=0; i<NUM_ITER; i++)
    {
//         for (int p=0; p<10; p++) 
//         {
//             //if (i==4 && p==4) timer.start();
//             baoParallelPropagate(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w); //10ms, after several iterations
//             //if (i==4 && p==4) timer.time_display("PM: One Propagate Step");
//         }
        //if (i==4) timer.start();
        //baoLinePropagate(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w); //105ms, R=9
        baoSegPropagate_Scaled(d_cost,d_disp_vec,d_scale,w,h,cost_mem_w,disp_mem_w,scale_mem_w,i);
        //baoJumpPropagate(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w);
        //if (i==4) timer.time_display("PM: Propagate Step");
//         getLastCudaError("PM: Propagate Step FAILED");

        //if (i==4) timer.start();
        baoRandomSearch_Scaled(d_cost,d_disp_vec,d_scale,w,h,cost_mem_w,disp_mem_w,scale_mem_w); //56ms, after several iterations, R=9
        //if (i==4) timer.time_display("PM: Random Search Step");
//         getLastCudaError("PM: Random Search Step FAILED");
    }
//     timer.time_display("PM: Main Body");

    // free memory of RNG
    checkCudaErrors(cudaFree(g_d_rand_states));
}

extern "C"
void baoCudaPatchMatch_PlaneFitting(short2* d_disp_vec, float* d_cost, uchar4* d_img1, uchar4* d_img2, unsigned char* d_census1, unsigned char* d_census2, int w, int h, size_t img_pitch, size_t cost_pitch, size_t disp_pitch, size_t census_pitch)
{
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);

    // allocate memory for RNG
    checkCudaErrors(cudaMalloc(&g_d_rand_states, gridSize.x*gridSize.y*sizeof(curandState))); //one RNG per block, (allocating memory takes 1ms+)

    // bind the array to the texture
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    checkCudaErrors(cudaBindTexture2D(0, rgbaImg1Tex, d_img1, desc, w, h, img_pitch));
    checkCudaErrors(cudaBindTexture2D(0, rgbaImg2Tex, d_img2, desc, w, h, img_pitch));
    
    //bind the census with texture memory
    census1Tex.filterMode = cudaFilterModePoint;
    census1Tex.normalized = false;
    census2Tex.filterMode = cudaFilterModePoint;
    census2Tex.normalized = false;
    cudaChannelFormatDesc desc_census = cudaCreateChannelDesc<unsigned char>();
    checkCudaErrors(cudaBindTexture2D(0, census1Tex, d_census1, desc_census, w, h, census_pitch));
    checkCudaErrors(cudaBindTexture2D(0, census2Tex, d_census2, desc_census, w, h, census_pitch));
//     getLastCudaError("Census Binding FAILED");

    size_t cost_mem_w = cost_pitch/sizeof(float);
    size_t disp_mem_w = disp_pitch/sizeof(short2);

    // init random NNF
//     bao_timer_gpu timer;
//     timer.start();
    baoGenerateRandomField(d_disp_vec,w,h,disp_mem_w); //5ms, R=9
//     timer.time_display(">>>PM: Init Random Field");
//     getLastCudaError("PM: Init Random Field FAILED");
    
    // calc init cost
//     timer.start();
    baoComputeCostField_PlaneFitting(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w); //5ms, R=9
//     timer.time_display(">>>PM: Init Cost Field");
//     getLastCudaError("PM: Init Cost Field FAILED");

    // iterated updating: random search and propagation
//     timer.start();
    for (int i=0; i<NUM_ITER; i++)
    {
//         for (int p=0; p<10; p++) 
//         {
//             //if (i==4 && p==4) timer.start();
//             baoParallelPropagate(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w); //10ms, after several iterations
//             //if (i==4 && p==4) timer.time_display("PM: One Propagate Step");
//         }
//         if (i==4) timer.start();
        //baoLinePropagate(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w); //105ms, R=9
        baoSegPropagate_PlaneFitting(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w,i);
        //baoJumpPropagate(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w);
//         if (i==4) timer.time_display(">>>>>>PM: Propagate Step");
//         getLastCudaError("PM: Propagate Step FAILED");

//         if (i==4) timer.start();
        baoRandomSearch_PlaneFitting(d_cost,d_disp_vec,w,h,cost_mem_w,disp_mem_w); //56ms, after several iterations, R=9
//         if (i==4) timer.time_display(">>>>>>PM: Random Search Step");
//         getLastCudaError("PM: Random Search Step FAILED");
    }
//     timer.time_display(">>>PM: Main Body");

    // free memory of RNG
    checkCudaErrors(cudaFree(g_d_rand_states));
}


//////////////////////////////////////////////////////////////////////////
// bilateral filter refinement (cost volume)
__global__ void d_bilateral_refine_flow(float2* d_flow_vec, int w, int h)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    float2 flow_val = d_flow_vec[id_y*w+id_x];
    if (flow_val.x > UNKNOWN_FLOW_THRESH || flow_val.y > UNKNOWN_FLOW_THRESH) {d_flow_vec[id_y*w+id_x]=make_float2(0); return;}
    short candidates_x[3];
    short candidates_y[3];
    candidates_x[1] = short(flow_val.x)+id_x;
    candidates_y[1] = short(flow_val.y)+id_y;
    candidates_x[0] = candidates_x[1]-1;
    candidates_y[0] = candidates_y[1]-1;
    candidates_x[2] = candidates_x[1]+1;
    candidates_y[2] = candidates_y[1]+1;
    short2 best_disp_vec;
    best_disp_vec.x = candidates_x[1];
    best_disp_vec.y = candidates_y[1];
    float min_cost_val = 999999;
    int radius = 5;

    //#pragma unroll
    for (int m=0; m<3; m++) for (int n=0; n<3; n++)
    {
        if (candidates_x[m]<0 || candidates_y[n]<0 || candidates_x[m]>=w || candidates_y[n]>=h) continue;
        float cost_val = _d_compute_patch_dist(id_x,id_y,candidates_x[m],candidates_y[n]);
        if (cost_val < min_cost_val)
        {
            min_cost_val = cost_val;
            best_disp_vec.x = candidates_x[m];
            best_disp_vec.y = candidates_y[n];
        }
    }
    flow_val.x = best_disp_vec.x - id_x;
    flow_val.y = best_disp_vec.y - id_y;
    d_flow_vec[id_y*w+id_x]=flow_val;
}
__global__ void d_bilateral_refine_flow_planefitting(float2* d_flow_vec, int w, int h)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;
    float2 flow_val = d_flow_vec[id_y*w+id_x];
    if (flow_val.x > UNKNOWN_FLOW_THRESH || flow_val.y > UNKNOWN_FLOW_THRESH) {d_flow_vec[id_y*w+id_x]=make_float2(0); return;}
    short candidates_x[3];
    short candidates_y[3];
    candidates_x[1] = short(flow_val.x)+id_x;
    candidates_y[1] = short(flow_val.y)+id_y;
    candidates_x[0] = candidates_x[1]-1;
    candidates_y[0] = candidates_y[1]-1;
    candidates_x[2] = candidates_x[1]+1;
    candidates_y[2] = candidates_y[1]+1;
    short2 best_disp_vec;
    best_disp_vec.x = candidates_x[1];
    best_disp_vec.y = candidates_y[1];
    float min_cost_val = 999999;
    int radius = 5;

    //#pragma unroll
    for (int m=0; m<3; m++) for (int n=0; n<3; n++)
    {
        if (candidates_x[m]<0 || candidates_y[n]<0 || candidates_x[m]>=w || candidates_y[n]>=h) continue;
        float cost_val = _d_compute_patch_dist_planefitting(id_x,id_y,candidates_x[m],candidates_y[n]);
        if (cost_val < min_cost_val)
        {
            min_cost_val = cost_val;
            best_disp_vec.x = candidates_x[m];
            best_disp_vec.y = candidates_y[n];
        }
    }
    flow_val.x = best_disp_vec.x - id_x;
    flow_val.y = best_disp_vec.y - id_y;
    d_flow_vec[id_y*w+id_x]=flow_val;
}
extern "C" 
void baoCudaBLFCostFilterRefine(float2* d_flow_vec, uchar4* d_img1, uchar4* d_img2, unsigned char* d_census1, unsigned char* d_census2, int w, int h, size_t img_pitch, size_t census_pitch)
{
    // bind the array to the texture
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    checkCudaErrors(cudaBindTexture2D(0, rgbaImg1Tex, d_img1, desc, w, h, img_pitch));
    checkCudaErrors(cudaBindTexture2D(0, rgbaImg2Tex, d_img2, desc, w, h, img_pitch));
    
    //bind the census with texture memory
    census1Tex.filterMode = cudaFilterModePoint;
    census1Tex.normalized = false;
    census2Tex.filterMode = cudaFilterModePoint;
    census2Tex.normalized = false;
    cudaChannelFormatDesc desc_census = cudaCreateChannelDesc<unsigned char>();
    checkCudaErrors(cudaBindTexture2D(0, census1Tex, d_census1, desc_census, w, h, census_pitch));
    checkCudaErrors(cudaBindTexture2D(0, census2Tex, d_census2, desc_census, w, h, census_pitch));
//     getLastCudaError("Census Binding FAILED");

    //start algorithm
//     bao_timer_gpu timer;
//     timer.start();
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    //d_bilateral_refine_flow<<<gridSize, blockSize>>>(d_flow_vec,w,h);
    d_bilateral_refine_flow_planefitting<<<gridSize, blockSize>>>(d_flow_vec,w,h);
//     timer.time_display("BLF Refinement");
//     getLastCudaError("BLF Refinement FAILED");
}


#define SIMILAR_MIN_COST  0.1 //for sintel 0.01, for webcam 0.1
__global__ void d_eliminate_still_region_flow(float2* d_flow, int w, int h, size_t flow_mem_w)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= w || id_y >= h) return;

    float cost = _d_compute_patch_dist_ad_L2(id_x, id_y, id_x, id_y);
    if (cost <= SIMILAR_MIN_COST) d_flow[id_y*flow_mem_w + id_x] = make_float2(0);
}
extern "C" 
void baoEliminateStillRegionFlow(float2* d_flow, uchar4* d_img1, uchar4* d_img2, int w, int h, size_t img_pitch)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    checkCudaErrors(cudaBindTexture2D(0, rgbaImg1Tex, d_img1, desc, w, h, img_pitch));
    checkCudaErrors(cudaBindTexture2D(0, rgbaImg2Tex, d_img2, desc, w, h, img_pitch));

    //launch kernel
    size_t flow_mem_w = w;
    dim3 gridSize(bao_div_ceil(w,BLOCK_DIM_X),bao_div_ceil(h,BLOCK_DIM_Y));
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);

    d_eliminate_still_region_flow<<<gridSize, blockSize>>>(d_flow, w, h, flow_mem_w);
}

