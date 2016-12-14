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


#include "bao_flow_patchmatch_multiscale_cuda.h"
#include "bao_basic.h"
#include "bao_basic_cuda.h"
#include "bao_flow_tools.h"
#include "defs.h"


// extern float*** g_imgshow; //for debug
// extern float4*** g_imgshow2; //for debug
// extern int g_nLevel; //for debug

//SEE bao_pmflow_kernel.cu and bao_pmflow_refine_kernel.cu
extern "C" void baoCudaPatchMatch(short2* d_disp_vec, float* d_cost, uchar4* d_img1, uchar4* d_img2, unsigned char* d_census1, unsigned char* d_census2, int w, int h, size_t img_pitch, size_t cost_pitch, size_t disp_pitch, size_t census_pitch);
extern "C" void baoCudaLeftRightCheck(short2* d_disp_vec, float* d_cost, short2* d_disp_vec2, float* d_cost2, int w, int h, size_t cost_pitch, size_t disp_pitch);
extern "C" void baoCudaLeftRightCheck_Buffered(short2* d_disp_vec, float* d_cost, short2* d_disp_vec2, float* d_cost2, short2* d_disp_vec_temp, float* d_cost_temp, int w, int h, size_t cost_pitch, size_t disp_pitch);
extern "C" void baoCudaOutlierRemoval(short2* d_disp_vec, float* d_cost, int w, int h, size_t cost_pitch, size_t disp_pitch);
extern "C" void baoCudaWeightedMedianFilter(short2* d_disp_vec, float* d_cost, uchar4* d_img, int w, int h, size_t img_pitch, size_t cost_pitch, size_t disp_pitch, int num_iter, bool is_only_occlusion);
extern "C" void baoCudaFillHole(short2* d_disp_vec, float* d_cost, uchar4* d_img, int w, int h, size_t img_pitch, size_t cost_pitch, size_t disp_pitch);
extern "C" void baoCudaNNF2Flow(float2* d_flow, short2* d_disp_vec, int w, int h, size_t disp_pitch, size_t flow_pitch);
extern "C" void baoCudaFlow2NNF(short2* d_disp_vec, float2* d_flow, int w, int h, size_t disp_pitch, size_t flow_pitch);
extern "C" void baoCudaImageSmoothing(uchar4* d_img_smoothed, uchar4* d_img, int w, int h, size_t img_pitch);
extern "C" void baoCudaFlowSmoothing(float2* d_flow, uchar4* d_img, int w, int h, size_t img_pitch, size_t flow_pitch);
extern "C" void baoCudaFlowCutoff(float2* d_flow, int w, int h, size_t flow_pitch, float max_flow_val);
extern "C" void baoCudaPatchMatch_Scaled(short2* d_disp_vec, float* d_scale, float* d_cost, uchar4* d_img1, uchar4* d_img2, unsigned char* d_census1, unsigned char* d_census2, int w, int h, size_t img_pitch, size_t cost_pitch, size_t disp_pitch, size_t scale_pitch, size_t census_pitch);
extern "C" void baoCudaPatchMatch_PlaneFitting(short2* d_disp_vec, float* d_cost, uchar4* d_img1, uchar4* d_img2, unsigned char* d_census1, unsigned char* d_census2, int w, int h, size_t img_pitch, size_t cost_pitch, size_t disp_pitch, size_t census_pitch);
extern "C" void baoCudaSubpixRefine(float2* d_flow, short2* d_disp_vec, uchar4* d_img1, uchar4* d_img2, unsigned char* d_census1_up, unsigned char* d_census2_up, int w, int h, size_t img_pitch, size_t census_pitch_up, size_t disp_pitch, size_t flow_pitch);
extern "C" void baoCudaCensusTransform_Bicubic(unsigned char* d_census1, unsigned char* d_census2, int w_up, int h_up, size_t census_pitch, uchar4* d_img1, uchar4* d_img2, int w, int h, size_t img_pitch);

extern "C" void baoEliminateStillRegionFlow(float2* d_flow, uchar4* d_img1, uchar4* d_img2, int w, int h, size_t img_pitch);

//SEE bao_flow_c2f_classic_kernel.cu
extern "C" void baoCudaPatchMatchMultiscalePrepare(uchar4** pImgPyr1, uchar4** pImgPyr2, unsigned char** pCensusPyr1, unsigned char** pCensusPyr2, uchar4** pTempPyr1, uchar4** pTempPyr2, int* arrH, int* arrW, size_t*arrPitchUchar4, size_t*arrPitchUchar1, int nLevels, uchar4*d_img1, uchar4*d_img2, int h, int w);
extern "C" void baoCudaBLF_C2F(float2** pFlowPyr, uchar4** pImgPyr1, uchar4** pImgPyr2, unsigned char** pCensusPyr1, unsigned char** pCensusPyr2, float2** pTempPyr1, float2** pTempPyr2, int* arrH, int* arrW, size_t* arrPitchUchar4, size_t* arrPitchUchar1, int nLayerIdx);

extern "C" void baoCudaFlowClassicRefine(float2* flow_uv, uchar4* d_img1, uchar4* d_img2, int h, int w, size_t pitchUchar4);

void bao_cuda_convert_flow_to_colorshow(uchar4* rgbflow, float2* flow_vec, int h, int w, float max_disp_x=100, float max_disp_y=100);


bao_flow_patchmatch_multiscale_cuda::bao_flow_patchmatch_multiscale_cuda():m_h_arr(NULL),m_w_arr(NULL),m_nLevels(0),m_h(0),m_w(0),
h_img1(NULL),
h_img2(NULL),
h_colorflow(NULL),
h_flow(NULL),
d_img1(NULL),
d_img2(NULL),
d_colorflow(NULL),
m_img1_pyr(NULL),
m_img2_pyr(NULL),
m_img1_census_pyramid(NULL),
m_img2_census_pyramid(NULL),
m_pmcost1_pyramid(NULL),
m_pmcost2_pyramid(NULL),
m_pmcost_temp_pyramid(NULL),
m_disp_vec1_pyramid(NULL),
m_disp_vec2_pyramid(NULL),
m_disp_vec_temp_pyramid(NULL),
m_scale1_pyramid(NULL),
m_scale2_pyramid(NULL),
m_flow1_pyramid(NULL),
m_tempf1_pyr(NULL),
m_tempf2_pyr(NULL),
m_tempu1_pyr(NULL),
m_tempu2_pyr(NULL),
m_arrPitchUchar4(NULL),
m_arrPitchUchar1(NULL),
m_pitchUchar4(0),
h_flow_pyr(NULL),h_temp1_pyr(NULL),h_temp2_pyr(NULL),
d_census1_up(NULL),d_census2_up(NULL)
{

}

bao_flow_patchmatch_multiscale_cuda::~bao_flow_patchmatch_multiscale_cuda()
{
    _destroy();
}

void bao_flow_patchmatch_multiscale_cuda::init( unsigned char***img1,unsigned char***img2,int h,int w )
{
    init(h,w);
    set_data(img1,img2);
}

void bao_flow_patchmatch_multiscale_cuda::init( int h,int w )
{
    m_h = h;
    m_w = w;
    m_nLevels = bao_pyr_init_dim(m_h_arr,m_w_arr,h,w,PYR_MAX_DEPTH,PYR_RATIO);
    m_arrPitchUchar4 = bao_alloc<size_t>(m_nLevels);
    m_arrPitchUchar1 = bao_alloc<size_t>(m_nLevels);
    h_img1 = bao_alloc<uchar4>(h,w);
    h_img2 = bao_alloc<uchar4>(h,w);
    h_colorflow = bao_alloc<uchar4>(h,w);
    h_flow = bao_alloc<float2>(h,w);

    h_flow_pyr = bao_pyr_alloc<float2>(m_nLevels,m_h_arr,m_w_arr);
    h_temp1_pyr = bao_pyr_alloc<float>(m_nLevels,m_h_arr,m_w_arr);
    h_temp2_pyr = bao_pyr_alloc<float>(m_nLevels,m_h_arr,m_w_arr);
//     h_imgshow = bao_pyr_alloc<float>(m_nLevels,m_h_arr,m_w_arr); //for debug
//     g_imgshow = h_imgshow; //for debug
//     h_imgshow2 = bao_pyr_alloc<float4>(m_nLevels,m_h_arr,m_w_arr); //for debug
//     g_imgshow2 = h_imgshow2; //for debug

    //bao_cuda_init();
    d_img1 = bao_cuda_alloc_pitched<uchar4>(m_pitchUchar4,h,w);
    d_img2 = bao_cuda_alloc_pitched<uchar4>(m_pitchUchar4,h,w);
    d_census1_up = bao_cuda_alloc_pitched<unsigned char>(m_census_pitch_up,h*SUBPIX_UP_FACTOR,w*SUBPIX_UP_FACTOR);
    d_census2_up = bao_cuda_alloc_pitched<unsigned char>(m_census_pitch_up,h*SUBPIX_UP_FACTOR,w*SUBPIX_UP_FACTOR);
    d_colorflow = bao_cuda_alloc<uchar4>(h,w);

    m_img1_pyr=bao_cuda_pyr_alloc_pitched<uchar4>(m_arrPitchUchar4,m_nLevels,m_h_arr,m_w_arr);
    m_img2_pyr=bao_cuda_pyr_alloc_pitched<uchar4>(m_arrPitchUchar4,m_nLevels,m_h_arr,m_w_arr);
    m_img1_census_pyramid=bao_cuda_pyr_alloc_pitched<unsigned char>(m_arrPitchUchar1,m_nLevels,m_h_arr,m_w_arr);
    m_img2_census_pyramid=bao_cuda_pyr_alloc_pitched<unsigned char>(m_arrPitchUchar1,m_nLevels,m_h_arr,m_w_arr);
    m_pmcost1_pyramid=bao_cuda_pyr_alloc<float>(m_nLevels,m_h_arr,m_w_arr);
    m_pmcost2_pyramid=bao_cuda_pyr_alloc<float>(m_nLevels,m_h_arr,m_w_arr);
    m_pmcost_temp_pyramid=bao_cuda_pyr_alloc<float>(m_nLevels,m_h_arr,m_w_arr);
    m_disp_vec1_pyramid=bao_cuda_pyr_alloc<short2>(m_nLevels,m_h_arr,m_w_arr);
    m_disp_vec2_pyramid=bao_cuda_pyr_alloc<short2>(m_nLevels,m_h_arr,m_w_arr);
    m_disp_vec_temp_pyramid=bao_cuda_pyr_alloc<short2>(m_nLevels,m_h_arr,m_w_arr);
    m_scale1_pyramid=bao_cuda_pyr_alloc<float>(m_nLevels,m_h_arr,m_w_arr);
    m_scale2_pyramid=bao_cuda_pyr_alloc<float>(m_nLevels,m_h_arr,m_w_arr);
    m_flow1_pyramid=bao_cuda_pyr_alloc<float2>(m_nLevels,m_h_arr,m_w_arr);

    m_tempf1_pyr=bao_cuda_pyr_alloc<float2>(m_nLevels,m_h_arr,m_w_arr);
    m_tempf2_pyr=bao_cuda_pyr_alloc<float2>(m_nLevels,m_h_arr,m_w_arr);
    m_tempu1_pyr=bao_cuda_pyr_alloc<uchar4>(m_nLevels,m_h_arr,m_w_arr);
    m_tempu2_pyr=bao_cuda_pyr_alloc<uchar4>(m_nLevels,m_h_arr,m_w_arr);
}

bool bao_flow_patchmatch_multiscale_cuda::set_data( unsigned char***img1,unsigned char***img2 )
{
    bao_rgb2rgba(h_img1,img1,m_h,m_w);
    bao_rgb2rgba(h_img2,img2,m_h,m_w);
    bao_cuda_copy_h2d_pitched(d_img1,m_pitchUchar4,h_img1[0],m_h,m_w);
    bao_cuda_copy_h2d_pitched(d_img2,m_pitchUchar4,h_img2[0],m_h,m_w);
    //prepare data
    _prepare_data();
    return true;
}

void bao_flow_patchmatch_multiscale_cuda::_destroy()
{
    bao_cuda_free_pitched(d_img1);
    bao_cuda_free_pitched(d_img2);
    bao_cuda_free_pitched(d_census1_up);
    bao_cuda_free_pitched(d_census2_up);
    bao_cuda_free(d_colorflow);
    bao_cuda_pyr_free_pitched(m_img1_pyr,m_nLevels);
    bao_cuda_pyr_free_pitched(m_img2_pyr,m_nLevels);
    bao_cuda_pyr_free_pitched(m_img1_census_pyramid,m_nLevels);
    bao_cuda_pyr_free_pitched(m_img2_census_pyramid,m_nLevels);
    bao_cuda_pyr_free(m_pmcost1_pyramid,m_nLevels);
    bao_cuda_pyr_free(m_pmcost2_pyramid,m_nLevels);
    bao_cuda_pyr_free(m_pmcost_temp_pyramid,m_nLevels);
    bao_cuda_pyr_free(m_disp_vec1_pyramid,m_nLevels);
    bao_cuda_pyr_free(m_disp_vec2_pyramid,m_nLevels);
    bao_cuda_pyr_free(m_disp_vec_temp_pyramid,m_nLevels);
    bao_cuda_pyr_free(m_scale1_pyramid,m_nLevels);
    bao_cuda_pyr_free(m_scale2_pyramid,m_nLevels);
    bao_cuda_pyr_free(m_flow1_pyramid,m_nLevels);
    bao_cuda_pyr_free(m_tempf1_pyr,m_nLevels);
    bao_cuda_pyr_free(m_tempf2_pyr,m_nLevels);
    bao_cuda_pyr_free(m_tempu1_pyr,m_nLevels);
    bao_cuda_pyr_free(m_tempu2_pyr,m_nLevels);

//     bao_pyr_free(h_imgshow,m_nLevels);
//     bao_pyr_free(h_imgshow2,m_nLevels);
    bao_pyr_free(h_flow_pyr,m_nLevels);
    bao_pyr_free(h_temp1_pyr,m_nLevels);
    bao_pyr_free(h_temp2_pyr,m_nLevels);

    bao_free(h_img1);
    bao_free(h_img2);
    bao_free(h_colorflow);
    bao_free(h_flow);
    bao_free(m_arrPitchUchar4);
    bao_free(m_arrPitchUchar1);
    bao_pyr_destroy_dim(m_h_arr,m_w_arr);

}


void bao_flow_patchmatch_multiscale_cuda::_prepare_data()
{
    baoCudaPatchMatchMultiscalePrepare(m_img1_pyr,m_img2_pyr,m_img1_census_pyramid,m_img2_census_pyramid,m_tempu1_pyr,m_tempu2_pyr,m_h_arr,m_w_arr,m_arrPitchUchar4,m_arrPitchUchar1,m_nLevels,d_img1,d_img2,m_h,m_w);
}

void bao_flow_patchmatch_multiscale_cuda::compute_flow( float**disp1_x,float**disp1_y,unsigned char***color_flow )
{
    int pm_layer = PYR_MAX_DEPTH-1;
    bao_timer_gpu_cpu timer;
//     timer.start();
    // patch match
     baoCudaPatchMatch(m_disp_vec1_pyramid[pm_layer],m_pmcost1_pyramid[pm_layer],m_img1_pyr[pm_layer],m_img2_pyr[pm_layer],m_img1_census_pyramid[pm_layer],m_img2_census_pyramid[pm_layer],m_w_arr[pm_layer],m_h_arr[pm_layer],m_arrPitchUchar4[pm_layer],m_w_arr[pm_layer]*sizeof(float),m_w_arr[pm_layer]*sizeof(short2),m_arrPitchUchar1[pm_layer]);
     baoCudaPatchMatch(m_disp_vec2_pyramid[pm_layer],m_pmcost2_pyramid[pm_layer],m_img2_pyr[pm_layer],m_img1_pyr[pm_layer],m_img2_census_pyramid[pm_layer],m_img1_census_pyramid[pm_layer],m_w_arr[pm_layer],m_h_arr[pm_layer],m_arrPitchUchar4[pm_layer],m_w_arr[pm_layer]*sizeof(float),m_w_arr[pm_layer]*sizeof(short2),m_arrPitchUchar1[pm_layer]);
//     baoCudaPatchMatch_Scaled(m_disp_vec1_pyramid[pm_layer],m_scale1_pyramid[pm_layer],m_pmcost1_pyramid[pm_layer],m_img1_pyr[pm_layer],m_img2_pyr[pm_layer],m_img1_census_pyramid[pm_layer],m_img2_census_pyramid[pm_layer],m_w_arr[pm_layer],m_h_arr[pm_layer],m_arrPitchUchar4[pm_layer],m_w_arr[pm_layer]*sizeof(float),m_w_arr[pm_layer]*sizeof(short2),m_w_arr[pm_layer]*sizeof(float),m_arrPitchUchar1[pm_layer]);
//     baoCudaPatchMatch_Scaled(m_disp_vec2_pyramid[pm_layer],m_scale2_pyramid[pm_layer],m_pmcost2_pyramid[pm_layer],m_img2_pyr[pm_layer],m_img1_pyr[pm_layer],m_img2_census_pyramid[pm_layer],m_img1_census_pyramid[pm_layer],m_w_arr[pm_layer],m_h_arr[pm_layer],m_arrPitchUchar4[pm_layer],m_w_arr[pm_layer]*sizeof(float),m_w_arr[pm_layer]*sizeof(short2),m_w_arr[pm_layer]*sizeof(float),m_arrPitchUchar1[pm_layer]);
//    baoCudaPatchMatch_PlaneFitting(m_disp_vec1_pyramid[pm_layer],m_pmcost1_pyramid[pm_layer],m_img1_pyr[pm_layer],m_img2_pyr[pm_layer],m_img1_census_pyramid[pm_layer],m_img2_census_pyramid[pm_layer],m_w_arr[pm_layer],m_h_arr[pm_layer],m_arrPitchUchar4[pm_layer],m_w_arr[pm_layer]*sizeof(float),m_w_arr[pm_layer]*sizeof(short2),m_arrPitchUchar1[pm_layer]);
//     baoCudaPatchMatch_PlaneFitting(m_disp_vec2_pyramid[pm_layer],m_pmcost2_pyramid[pm_layer],m_img2_pyr[pm_layer],m_img1_pyr[pm_layer],m_img2_census_pyramid[pm_layer],m_img1_census_pyramid[pm_layer],m_w_arr[pm_layer],m_h_arr[pm_layer],m_arrPitchUchar4[pm_layer],m_w_arr[pm_layer]*sizeof(float),m_w_arr[pm_layer]*sizeof(short2),m_arrPitchUchar1[pm_layer]);
//     timer.time_display("PM Total");

    // left-right check
//     timer.start();
     baoCudaLeftRightCheck(m_disp_vec1_pyramid[pm_layer],m_pmcost1_pyramid[pm_layer],m_disp_vec2_pyramid[pm_layer],m_pmcost2_pyramid[pm_layer],m_w_arr[pm_layer],m_h_arr[pm_layer],m_w_arr[pm_layer]*sizeof(float),m_w_arr[pm_layer]*sizeof(short2));
//     baoCudaLeftRightCheck_Buffered(m_disp_vec1_pyramid[pm_layer],m_pmcost1_pyramid[pm_layer],m_disp_vec2_pyramid[pm_layer],m_pmcost2_pyramid[pm_layer],m_disp_vec_temp_pyramid[pm_layer],m_pmcost_temp_pyramid[pm_layer],m_w_arr[pm_layer],m_h_arr[pm_layer],m_w_arr[pm_layer]*sizeof(float),m_w_arr[pm_layer]*sizeof(short2));

    // fill holes
    baoCudaOutlierRemoval(m_disp_vec1_pyramid[pm_layer],m_pmcost1_pyramid[pm_layer],m_w_arr[pm_layer],m_h_arr[pm_layer],m_w_arr[pm_layer]*sizeof(float),m_w_arr[pm_layer]*sizeof(short2)); //remove isolated outliers
//     baoCudaWeightedMedianFilter(m_disp_vec1_pyramid[pm_layer],m_pmcost1_pyramid[pm_layer],m_img1_pyr[pm_layer],m_w_arr[pm_layer],m_h_arr[pm_layer],m_arrPitchUchar4[pm_layer],m_w_arr[pm_layer]*sizeof(float),m_w_arr[pm_layer]*sizeof(short2),1,false);
    baoCudaWeightedMedianFilter(m_disp_vec1_pyramid[pm_layer],m_pmcost1_pyramid[pm_layer],m_img1_pyr[pm_layer],m_w_arr[pm_layer],m_h_arr[pm_layer],m_arrPitchUchar4[pm_layer],m_w_arr[pm_layer]*sizeof(float),m_w_arr[pm_layer]*sizeof(short2),20,true);
    baoCudaFillHole(m_disp_vec1_pyramid[pm_layer],m_pmcost1_pyramid[pm_layer],m_img1_pyr[pm_layer],m_w_arr[pm_layer],m_h_arr[pm_layer],m_arrPitchUchar4[pm_layer],m_w_arr[pm_layer]*sizeof(float),m_w_arr[pm_layer]*sizeof(short2)); //fill holes
    
    
//     baoCudaOutlierRemoval(m_disp_vec2_pyramid[pm_layer],m_pmcost2_pyramid[pm_layer],m_w_arr[pm_layer],m_h_arr[pm_layer],m_w_arr[pm_layer]*sizeof(float),m_w_arr[pm_layer]*sizeof(short2)); //remove isolated outliers
//     baoCudaWeightedMedianFilter(m_disp_vec2_pyramid[pm_layer],m_pmcost2_pyramid[pm_layer],m_img2_pyr[pm_layer],m_w_arr[pm_layer],m_h_arr[pm_layer],m_arrPitchUchar4[pm_layer],m_w_arr[pm_layer]*sizeof(float),m_w_arr[pm_layer]*sizeof(short2),20,true);
//     baoCudaFillHole(m_disp_vec2_pyramid[pm_layer],m_pmcost2_pyramid[pm_layer],m_img2_pyr[pm_layer],m_w_arr[pm_layer],m_h_arr[pm_layer],m_arrPitchUchar4[pm_layer],m_w_arr[pm_layer]*sizeof(float),m_w_arr[pm_layer]*sizeof(short2)); //fill holes
// 
//     // wmf
//     baoCudaWeightedMedianFilter(m_disp_vec1_pyramid[pm_layer],m_pmcost1_pyramid[pm_layer],m_img1_pyr[pm_layer],m_w_arr[pm_layer],m_h_arr[pm_layer],m_arrPitchUchar4[pm_layer],m_w_arr[pm_layer]*sizeof(float),m_w_arr[pm_layer]*sizeof(short2),1,false);
//     baoCudaWeightedMedianFilter(m_disp_vec2_pyramid[pm_layer],m_pmcost2_pyramid[pm_layer],m_img2_pyr[pm_layer],m_w_arr[pm_layer],m_h_arr[pm_layer],m_arrPitchUchar4[pm_layer],m_w_arr[pm_layer]*sizeof(float),m_w_arr[pm_layer]*sizeof(short2),1,false);
// 
//     // second round left-right check and filling holes
//     baoCudaLeftRightCheck(m_disp_vec1_pyramid[pm_layer],m_pmcost1_pyramid[pm_layer],m_disp_vec2_pyramid[pm_layer],m_pmcost2_pyramid[pm_layer],m_w_arr[pm_layer],m_h_arr[pm_layer],m_w_arr[pm_layer]*sizeof(float),m_w_arr[pm_layer]*sizeof(short2));
//     baoCudaOutlierRemoval(m_disp_vec1_pyramid[pm_layer],m_pmcost1_pyramid[pm_layer],m_w_arr[pm_layer],m_h_arr[pm_layer],m_w_arr[pm_layer]*sizeof(float),m_w_arr[pm_layer]*sizeof(short2)); //remove isolated outliers
//     baoCudaWeightedMedianFilter(m_disp_vec1_pyramid[pm_layer],m_pmcost1_pyramid[pm_layer],m_img1_pyr[pm_layer],m_w_arr[pm_layer],m_h_arr[pm_layer],m_arrPitchUchar4[pm_layer],m_w_arr[pm_layer]*sizeof(float),m_w_arr[pm_layer]*sizeof(short2),20,true);
//     baoCudaFillHole(m_disp_vec1_pyramid[pm_layer],m_pmcost1_pyramid[pm_layer],m_img1_pyr[pm_layer],m_w_arr[pm_layer],m_h_arr[pm_layer],m_arrPitchUchar4[pm_layer],m_w_arr[pm_layer]*sizeof(float),m_w_arr[pm_layer]*sizeof(short2)); //fill holes
    
    // convert to flow
    baoCudaNNF2Flow(m_flow1_pyramid[pm_layer],m_disp_vec1_pyramid[pm_layer],m_w_arr[pm_layer],m_h_arr[pm_layer],m_w_arr[pm_layer]*sizeof(short2),m_w_arr[pm_layer]*sizeof(float2));
//     baoEliminateStillRegionFlow(m_flow1_pyramid[pm_layer],m_img1_pyr[pm_layer],m_img2_pyr[pm_layer],m_w_arr[pm_layer],m_h_arr[pm_layer],m_arrPitchUchar4[pm_layer]);
//     timer.time_display("Refine");

    //for debug show
    //bao_cuda_copy_d2h(h_temp1_pyr[pm_layer][0],m_scale1_pyramid[pm_layer],m_h_arr[pm_layer],m_w_arr[pm_layer]);
    //imshow(h_temp1_pyr[pm_layer],m_h_arr[pm_layer],m_w_arr[pm_layer]);
    bao_cuda_copy_d2h(h_flow_pyr[pm_layer][0],m_flow1_pyramid[pm_layer],m_h_arr[pm_layer],m_w_arr[pm_layer]);
    for (int i=0;i<m_h_arr[pm_layer];i++) for(int j=0;j<m_w_arr[pm_layer];j++) 
    {
        h_temp1_pyr[pm_layer][i][j]=h_flow_pyr[pm_layer][i][j].x;
        h_temp2_pyr[pm_layer][i][j]=h_flow_pyr[pm_layer][i][j].y;
    }
    //bao_save_flo_file("test.flo",h_temp1_pyr[pm_layer],h_temp2_pyr[pm_layer],m_h_arr[pm_layer],m_w_arr[pm_layer]);
    
    //upsample and blf refine
//     timer.start();
    for (int nlayer=pm_layer-1; nlayer>=0; nlayer--)
    {
//         baoCudaFlowSmoothing(m_flow1_pyramid[nlayer+1],m_img1_pyr[nlayer+1],m_w_arr[nlayer+1],m_h_arr[nlayer+1],m_arrPitchUchar4[nlayer+1],m_w_arr[nlayer+1]*sizeof(float2));
        baoCudaBLF_C2F(m_flow1_pyramid,m_img1_pyr,m_img2_pyr,m_img1_census_pyramid,m_img2_census_pyramid,m_tempf1_pyr,m_tempf2_pyr,m_h_arr,m_w_arr,m_arrPitchUchar4,m_arrPitchUchar1,nlayer);

        baoCudaFlowSmoothing(m_flow1_pyramid[nlayer],m_img1_pyr[nlayer],m_w_arr[nlayer],m_h_arr[nlayer],m_arrPitchUchar4[nlayer],m_w_arr[nlayer]*sizeof(float2));
        baoCudaWeightedMedianFilter(m_disp_vec1_pyramid[nlayer],m_pmcost1_pyramid[nlayer],m_img1_pyr[nlayer],m_w_arr[nlayer],m_h_arr[nlayer],m_arrPitchUchar4[nlayer],m_w_arr[nlayer]*sizeof(float),m_w_arr[nlayer]*sizeof(short2),1,false);
    }
//     timer.time_display("Upsample");

    //final flow smoothing
//     timer.start();
//     baoEliminateStillRegionFlow(m_flow1_pyramid[0],m_img1_pyr[0],m_img2_pyr[0],m_w_arr[0],m_h_arr[0],m_arrPitchUchar4[0]);
//     baoCudaFlowCutoff(m_flow1_pyramid[0],m_w_arr[0],m_h_arr[0],m_w_arr[0]*sizeof(float2),MAX_FLOW_VAL);
    baoCudaFlowSmoothing(m_flow1_pyramid[0],m_img1_pyr[0],m_w_arr[0],m_h_arr[0],m_arrPitchUchar4[0],m_w_arr[0]*sizeof(float2));
//     timer.time_display("Final step");

    //L1 refinement
//     timer.start();
//     baoCudaFlowClassicRefine(m_flow1_pyramid[0],m_img1_pyr[0],m_img2_pyr[0],m_h_arr[0],m_w_arr[0],m_arrPitchUchar4[0]);
//     timer.time_display("L1 ref");

    // copy result (d_disp_vec) to host
//     checkCudaErrors(cudaDeviceSynchronize());
    bao_cuda_copy_d2h(h_flow[0],m_flow1_pyramid[0],m_h,m_w);

    // output
    for (int i=0;i<m_h;i++) for(int j=0;j<m_w;j++) 
    {
        disp1_x[i][j] = h_flow[i][j].x; 
        disp1_y[i][j] = h_flow[i][j].y; 
    }

    if (color_flow != NULL)
    {
        //convert to color show
        bao_cuda_convert_flow_to_colorshow(d_colorflow,m_flow1_pyramid[0],m_h,m_w,20,20);
        bao_cuda_copy_d2h(h_colorflow[0],d_colorflow,m_h,m_w);
        bao_rgba2rgb(color_flow,h_colorflow,m_h,m_w);
    }
}




