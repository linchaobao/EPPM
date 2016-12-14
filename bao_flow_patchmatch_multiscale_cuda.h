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


#ifndef _BAO_FLOW_PATCHMATCH_MULTISCALE_CUDA_H_
#define _BAO_FLOW_PATCHMATCH_MULTISCALE_CUDA_H_

#include "bao_basic_cuda.h"

class bao_flow_patchmatch_multiscale_cuda
{
public:
    bao_flow_patchmatch_multiscale_cuda();
    ~bao_flow_patchmatch_multiscale_cuda();

public:
    //interface    
    void init(int h,int w);
    void init(unsigned char***img1,unsigned char***img2,int h,int w);
    bool set_data(unsigned char***img1,unsigned char***img2); //if two image is equal (contents), return false
    void compute_flow(float**disp1_x,float**disp1_y,unsigned char***color_flow=NULL);

private:
    void _destroy();
    void _prepare_data();

private:
    int m_h;
    int m_w;
    int* m_h_arr;
    int* m_w_arr;
    int  m_nLevels;
    size_t* m_arrPitchUchar4;
    size_t* m_arrPitchUchar1;

    //host ptr
    uchar4**          h_img1;
    uchar4**          h_img2;
    uchar4**          h_colorflow;
    float2**          h_flow;
    float2***         h_flow_pyr; //for debug
    float***          h_temp1_pyr; //for debug
    float***          h_temp2_pyr; //for debug

//     float***           h_imgshow; //for debug
//     float4***          h_imgshow2; //for debug

    //the following are device ptr
    uchar4*           d_img1;
    uchar4*           d_img2;
    uchar4*           d_colorflow;
    size_t            m_pitchUchar4;
    uchar4**          m_img1_pyr;
    uchar4**          m_img2_pyr;
    unsigned char**   m_img1_census_pyramid;
    unsigned char**   m_img2_census_pyramid;
    float**           m_pmcost1_pyramid;
    float**           m_pmcost2_pyramid;
    float**           m_pmcost_temp_pyramid;
    short2**          m_disp_vec1_pyramid;
    short2**          m_disp_vec2_pyramid;
    short2**          m_disp_vec_temp_pyramid;
    float**           m_scale1_pyramid;
    float**           m_scale2_pyramid;
    float2**          m_flow1_pyramid;
    float2**          m_tempf1_pyr;
    float2**          m_tempf2_pyr;
    uchar4**          m_tempu1_pyr;
    uchar4**          m_tempu2_pyr;
    
    //subpixel
    unsigned char*  d_census1_up;
    unsigned char*  d_census2_up;
    size_t m_census_pitch_up;
};



#endif


