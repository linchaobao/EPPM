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
 * Tools like reading and writing optical flow file (*.flo). Just some interfaces to Middlebury code. 
 */
#ifndef _BAO_FLOW_TOOLS_H_
#define _BAO_FLOW_TOOLS_H_

#include "bao_basic.h"

// the "official" threshold - if the absolute value of either 
// flow component is greater, it's considered unknown
#ifndef UNKNOWN_FLOW_THRESH
#define UNKNOWN_FLOW_THRESH 1e9
#endif

// value to use to represent unknown flow
#ifndef UNKNOWN_FLOW
#define UNKNOWN_FLOW 1e10
#endif

void bao_read_flo_file_size(const char*filename,int& h,int& w);
void bao_load_flo_file(const char*filename,float**disp_x,float**disp_y,int h,int w);
void bao_save_flo_file(const char*filename,float**disp_x,float**disp_y,int h,int w);
void bao_flow_cutoff(float**disp_x_out,float**disp_y_out,float**disp_x,float**disp_y,int h,int w,int cutoff_val,bool is_cut_invalid_flow_value=false);
void bao_convert_flow_to_colorshow(unsigned char***disp_color,float**disp_x,float**disp_y,int h,int w);
void bao_display_flow_error(float**disp_x,float**disp_y,float**disp_gt_x,float**disp_gt_y,int h,int w);
void bao_calc_flow_error(float**disp_x,float**disp_y,float**disp_gt_x,float**disp_gt_y,int h,int w,float& epe,float& aae,int border=0,bool is_show_error=false);
float bao_calc_flow_error_percentage(float**disp_x,float**disp_y,float**disp_gt_x,float**disp_gt_y,int h,int w,int error_thresh,unsigned char**error_map=0);
void bao_display_flow_vec_gray(float**disp_x,float**disp_y,int h,int w);
void bao_display_flow_vec_color(float**disp_x,float**disp_y,int h,int w,char*savecolorimage=0,bool is_show=true);
void bao_display_flow_vec_color(unsigned char***img1,float**disp_x,float**disp_y,int h,int w,char*savecolorimage=0,char*savecombimage=0,bool is_show=true);


#endif
