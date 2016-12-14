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


#include "bao_flow_tools.h"
#include "../3rdparty/middlebury/imageLib/Image.h"
#include "../3rdparty/middlebury/flowIO.h"
#include "../3rdparty/middlebury/colorcode.h"

void bao_read_flo_file_size(const char*filename,int& h,int& w)
{
    ReadFlowFileSize(h,w,filename);
}

void bao_load_flo_file(const char*filename,float**disp_x,float**disp_y,int h,int w)
{
    CFloatImage floData;
    ReadFlowFile(floData, filename);
    for (int y=0; y<h; y++) for (int x=0; x<w; x++)
    {
        disp_x[y][x] = floData.Pixel(x,y,0);
        disp_y[y][x] = floData.Pixel(x,y,1);
    }
}

void bao_save_flo_file(const char*filename,float**disp_x,float**disp_y,int h,int w)
{
    CFloatImage floData(w,h,2);
    for (int y=0; y<h; y++) 
    {
        float*dataLine = &floData.Pixel(0,y,0);
        for (int x=0; x<w; x++)
        {
            dataLine[2*x] = disp_x[y][x];
            dataLine[2*x+1] = disp_y[y][x];
        }
    }
    WriteFlowFile(floData, filename);
}

void bao_calc_flow_error(float**disp_x,float**disp_y,float**disp_gt_x,float**disp_gt_y,int h,int w,float& epe_out,float& aae_out,int border,bool is_show_error)
{
    float**disp_error=bao_alloc<float>(h,w);
    float**disp_angle_error=bao_alloc<float>(h,w);
    int num_valid_pix = 0;
    float uu,vv,gtuu,gtvv;
    float cos_val,angle_val,total_angle_val=0,mean_angle_val;
    float epe_val,total_epe_val=0,mean_epe_val;
    for (int y=border; y<h-border; y++) for (int x=border; x<w-border; x++)
    {
        gtuu = disp_gt_x[y][x];
        gtvv = disp_gt_y[y][x];
        if ((fabs(gtuu) > 0 && fabs(gtuu) <= UNKNOWN_FLOW_THRESH) || (fabs(gtvv) >0 && fabs(gtvv) <= UNKNOWN_FLOW_THRESH))
        {
            num_valid_pix++;
            uu = disp_x[y][x];
            vv = disp_y[y][x];
            cos_val = (uu*gtuu+vv*gtvv+1.0f)/(sqrt(uu*uu+vv*vv+1.0f)*sqrt(gtuu*gtuu+gtvv*gtvv+1.0f));
            angle_val = acos(cos_val);
            total_angle_val += angle_val;
            epe_val = sqrt((uu-gtuu)*(uu-gtuu)+(vv-gtvv)*(vv-gtvv));
            total_epe_val += epe_val;

            disp_error[y][x]=epe_val;
            disp_angle_error[y][x]=angle_val * 180.0f / 3.14159f;
        }
        else
        {
            disp_error[y][x]=0;
            disp_angle_error[y][x]=0;
        }
    }

    if (num_valid_pix>0)
    {
        mean_angle_val = total_angle_val / num_valid_pix;
        aae_out = mean_angle_val * 180.0f / 3.14159f; //turn radian to degree
        mean_epe_val = total_epe_val / num_valid_pix;
        epe_out = mean_epe_val;
    }
    if (is_show_error)
    {
        //imshow(disp_error,h,w);
        //imshow(disp_angle_error,h,w);
    }
    bao_free(disp_error);
    bao_free(disp_angle_error);
}


float bao_calc_flow_error_percentage(float**disp_x,float**disp_y,float**disp_gt_x,float**disp_gt_y,int h,int w,int error_thresh,unsigned char**error_map)
{
    if (error_map) bao_memset(error_map,h,w);
    int num_valid_pix = 0;
    int num_correct_pix = 0;
    float uu,vv,gtuu,gtvv;
    float epe_val;
    for (int y=0; y<h; y++) for (int x=0; x<w; x++)
    {
        gtuu = disp_gt_x[y][x];
        gtvv = disp_gt_y[y][x];
        if (fabs(gtuu) <= UNKNOWN_FLOW_THRESH || fabs(gtvv) <= UNKNOWN_FLOW_THRESH)
        {
            num_valid_pix++;
            uu = disp_x[y][x];
            vv = disp_y[y][x];
            epe_val = sqrt((uu-gtuu)*(uu-gtuu)+(vv-gtvv)*(vv-gtvv));
            if (epe_val <= error_thresh) num_correct_pix++;
            else if (error_map) error_map[y][x] = 255;
        }
    }

    if (num_valid_pix>0)
    {
        return 1.0f - float(num_correct_pix)/float(num_valid_pix);
    }
    return 0;
}


void bao_display_flow_error(float**disp_x,float**disp_y,float**disp_gt_x,float**disp_gt_y,int h,int w)
{
    float**disp_error=bao_alloc<float>(h,w);
    float temp_x_error,temp_y_error;
    for (int y=0; y<h; y++) for (int x=0; x<w; x++) 
    {
        temp_x_error=0;
        temp_y_error=0;
        if (fabs(disp_gt_x[y][x])<=UNKNOWN_FLOW_THRESH)
        {
            temp_x_error=fabs(disp_gt_x[y][x]-disp_x[y][x]);
        }        
        if (fabs(disp_gt_y[y][x])<=UNKNOWN_FLOW_THRESH)
        {
            temp_y_error=fabs(disp_gt_y[y][x]-disp_y[y][x]);
        }
        disp_error[y][x]=sqrt(temp_x_error*temp_x_error+temp_y_error*temp_y_error);
    }
    //imshow(disp_error,h,w);
    bao_free(disp_error);
}

void bao_flow_cutoff(float**disp_x_out,float**disp_y_out,float**disp_x,float**disp_y,int h,int w,int cutoff_val,bool is_cut_invalid_flow_value)
{
    cutoff_val = fabs(cutoff_val);
    if (is_cut_invalid_flow_value)
    {
        for (int y=0; y<h; y++) for (int x=0; x<w; x++) 
        {
            float val_x = disp_x[y][x];
            float val_y = disp_y[y][x];
            disp_x_out[y][x] = __max(__min(val_x,cutoff_val),-cutoff_val);
            disp_y_out[y][x] = __max(__min(val_y,cutoff_val),-cutoff_val);
        }
    }
    else
    {
        for (int y=0; y<h; y++) for (int x=0; x<w; x++) 
        {
            float val_x = disp_x[y][x];
            float val_y = disp_y[y][x];
            if (unknown_flow(val_x, val_y))
            {
                disp_x_out[y][x] = disp_x[y][x];
                disp_y_out[y][x] = disp_y[y][x];
            }
            else
            {
                disp_x_out[y][x] = __max(__min(val_x,cutoff_val),-cutoff_val);
                disp_y_out[y][x] = __max(__min(val_y,cutoff_val),-cutoff_val);
            }
        }
    }
}


void bao_convert_flow_to_colorshow(unsigned char***disp_color,float**disp_x,float**disp_y,int h,int w)
{
    // determine motion range:
    float maxx = -999999, maxy = -999999;
    float minx =  999999, miny =  999999;
    float maxrad = -1;
    float fx,fy,rad;
    for (int y=0; y<h; y++) for (int x=0; x<w; x++) 
    {
        fx = disp_x[y][x];
        fy = disp_y[y][x];
        if (unknown_flow(fx, fy))
            continue;
        maxx = __max(maxx, fx);
        maxy = __max(maxy, fy);
        minx = __min(minx, fx);
        miny = __min(miny, fy);
        rad = sqrt(fx * fx + fy * fy);
        maxrad = __max(maxrad, rad);
    }
    uchar pixel_val[3];
    for (int y=0; y<h; y++) for (int x=0; x<w; x++)
    {
        fx = disp_x[y][x];
        fy = disp_y[y][x];
        if (unknown_flow(fx, fy)) pixel_val[0] = pixel_val[1] = pixel_val[2] = 0;
        else computeColor(fx/maxrad, fy/maxrad, pixel_val);
        disp_color[y][x][0] = pixel_val[2];
        disp_color[y][x][1] = pixel_val[1];
        disp_color[y][x][2] = pixel_val[0];
    }
}

void bao_display_flow_vec_gray(float**disp_x,float**disp_y,int h,int w)
{
    float**disp_x_temp=bao_alloc<float>(h,w);
    float**disp_y_temp=bao_alloc<float>(h,w);
    for (int y=0; y<h; y++) for (int x=0; x<w; x++) 
    {
        disp_x_temp[y][x] = disp_x[y][x];
        disp_y_temp[y][x] = disp_y[y][x];
        if (disp_x_temp[y][x]>UNKNOWN_FLOW_THRESH)
            disp_x_temp[y][x] = 0;
        if (disp_y_temp[y][x]>UNKNOWN_FLOW_THRESH)
            disp_y_temp[y][x] = 0;
    }
    //imshow(disp_x_temp,h,w);
    //imshow(disp_y_temp,h,w);
    bao_free(disp_x_temp);
    bao_free(disp_y_temp);
}

void bao_display_flow_vec_color(float**disp_x,float**disp_y,int h,int w,char*savecolorimage,bool is_show)
{
    unsigned char***disp_color = bao_alloc<unsigned char>(h,w,3);
    bao_convert_flow_to_colorshow(disp_color,disp_x,disp_y,h,w);
    bao_free(disp_color);
}

void bao_display_flow_vec_color(unsigned char***img1,float**disp_x,float**disp_y,int h,int w,char*savecolorimage,char*savecombimage,bool is_show)
{
    unsigned char***disp_color = bao_alloc<unsigned char>(h,w,3);
    bao_convert_flow_to_colorshow(disp_color,disp_x,disp_y,h,w);
    if (savecombimage)
    {
        if (w>=h*2)
        {
            unsigned char*** comb_img = bao_alloc<unsigned char>(h*2,w,3);
            for (int y=0; y<h; y++) for (int x=0; x<w; x++) for (int c=0; c<3; c++)
            {
                comb_img[y][x][c] = disp_color[y][x][c];
                comb_img[h+y][x][c] = img1[y][x][c];
            }
            bao_free(comb_img);
        }
        else
        {
            unsigned char*** comb_img = bao_alloc<unsigned char>(h,w*2,3);
            for (int y=0; y<h; y++) for (int x=0; x<w; x++) for (int c=0; c<3; c++)
            {
                comb_img[y][x][c] = img1[y][x][c];
                comb_img[y][w+x][c] = disp_color[y][x][c];
            }
            bao_free(comb_img);
        }
    }
    bao_free(disp_color);
}
