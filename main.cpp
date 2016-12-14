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
 
#include "bao_basic.h"
#include "bao_flow_patchmatch_multiscale_cuda.h"
#include "bao_flow_tools.h"
#include <fstream>
#include <string>

using namespace std;


int main(int argc,char*argv[])
{
  int h = 480;
  int w = 640;
	
  //alloc memory
  unsigned char***img1=bao_alloc<unsigned char>(h,w,3);
  unsigned char***img2=bao_alloc<unsigned char>(h,w,3);
  float**disp1_x=bao_alloc<float>(h,w);
  float**disp1_y=bao_alloc<float>(h,w);
  float**disp2_x=bao_alloc<float>(h,w);
  float**disp2_y=bao_alloc<float>(h,w);
  memset(&(disp1_x[0][0]),0,sizeof(float)*h*w);
  memset(&(disp1_y[0][0]),0,sizeof(float)*h*w);
  memset(&(disp2_x[0][0]),0,sizeof(float)*h*w);
  memset(&(disp2_y[0][0]),0,sizeof(float)*h*w);

  //read images 
  cout << "loading image ... " << endl;
  int nchannels = 0;
  bao_loadimage_ppm("frame10.ppm",img1[0][0],h,w,&nchannels);//load image
  bao_loadimage_ppm("frame11.ppm",img2[0][0],h,w,&nchannels);//load image

  cout << "Processing (image size " << w << " * " << h << " * " << nchannels << ")...\n";
  bao_timer_gpu_cpu timer; 
  bao_flow_patchmatch_multiscale_cuda  eppm;
  
  timer.start();
  eppm.init(img1,img2,h,w);
  eppm.compute_flow(disp1_x,disp1_y);
  timer.time_display("GPU");
  
  cout << "Saving flo file..." << h << "*" << w << endl;
  bao_save_flo_file("flow.flo",disp1_x,disp1_y,h,w);

  bao_free(img1);
  bao_free(img2);
  bao_free(disp1_x);
  bao_free(disp1_y);
  bao_free(disp2_x);
  bao_free(disp2_y);

    return 0;
}


