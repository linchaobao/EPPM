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


//////////////////////////////////////////////////////////////////////////
// CPU timer
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

void bao_timer_cpu::start()
{
#ifdef _WIN32
    LARGE_INTEGER li;
    if(!QueryPerformanceFrequency(&li)) 
    {
        printf("QueryPerformanceFrequency failed!\n");
    }
    m_pc_frequency = double(li.QuadPart); ///1000.0;
    QueryPerformanceCounter(&li);
    m_counter_start = li.QuadPart;
#else
    gettimeofday(&timerStart, NULL);
#endif
}

double bao_timer_cpu::stop()
{
#ifdef _WIN32
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return double(li.QuadPart-m_counter_start)/m_pc_frequency;
#else
    struct timeval timerStop, timerElapsed;
    gettimeofday(&timerStop, NULL);
    timersub(&timerStop, &timerStart, &timerElapsed);
    return timerElapsed.tv_sec*1000.0+timerElapsed.tv_usec/1000.0;
#endif
}

double bao_timer_cpu::time_display(char *disp,int nr_frame)
{ 
    double sec = stop()/nr_frame;
    printf("Running time (%s) is: %5.5f Seconds.\n",disp,sec);
    return sec;
}

double bao_timer_cpu::fps_display(char *disp,int nr_frame)
{ 
    double fps = (double)nr_frame/stop();
    printf("Running time (%s) is: %5.5f frame per second.\n",disp,fps);
    return fps;
}


//////////////////////////////////////////////////////////////////////////
// Other tools
BAO_FLOAT bao_inv_3x3(BAO_FLOAT h_inv[9],BAO_FLOAT h_in[9],BAO_FLOAT threshold)
{
    BAO_FLOAT det;
    BAO_FLOAT h5h7, h4h8, h3h8,h5h6,h4h6,h3h7;
    BAO_FLOAT h1h8, h2h7, h0h8, h2h6, h0h7, h1h6, h1h5, h2h4, h0h5, h2h3, h0h4, h1h3;
    h4h8= h_in[4]*h_in[8]; h5h7=h_in[5]*h_in[7]; h3h8= h_in[3]*h_in[8]; 
    h5h6= h_in[5]*h_in[6]; h3h7= h_in[3]*h_in[7]; h4h6= h_in[4]*h_in[6]; 
    det= h_in[0]*(h4h8-h5h7)-h_in[1]*(h3h8-h5h6)+h_in[2]*(h3h7-h4h6);
    if (abs(det)<threshold) 
    {
        printf("[det<%e] ",threshold);
        memset(h_inv,0,sizeof(BAO_FLOAT)*9);
    }
    else
    {
        h1h8= h_in[1]*h_in[8];
        h2h7= h_in[2]*h_in[7];
        h0h8= h_in[0]*h_in[8]; 
        h2h6= h_in[2]*h_in[6]; 
        h0h7= h_in[0]*h_in[7];
        h1h6= h_in[1]*h_in[6]; 
        h1h5= h_in[1]*h_in[5];
        h2h4= h_in[2]*h_in[4];
        h0h5= h_in[0]*h_in[5];
        h2h3= h_in[2]*h_in[3];
        h0h4= h_in[0]*h_in[4];
        h1h3= h_in[1]*h_in[3];
        //inv_det= 1.0/det;
        //h_inv[0]=  inv_det*(h4h8-h5h7);
        //h_inv[3]= -inv_det*(h3h8-h5h6);
        //h_inv[6]=  inv_det*(h3h7-h4h6);
        //h_inv[1]= -inv_det*(h1h8-h2h7);
        //h_inv[4]=  inv_det*(h0h8-h2h6);
        //h_inv[7]= -inv_det*(h0h7-h1h6);
        //h_inv[2]=  inv_det*(h1h5-h2h4);
        //h_inv[5]= -inv_det*(h0h5-h2h3);
        //h_inv[8]=  inv_det*(h0h4-h1h3);
        h_inv[0]=  (h4h8-h5h7)/det;
        h_inv[3]= -(h3h8-h5h6)/det;
        h_inv[6]=  (h3h7-h4h6)/det;
        h_inv[1]= -(h1h8-h2h7)/det;
        h_inv[4]=  (h0h8-h2h6)/det;
        h_inv[7]= -(h0h7-h1h6)/det;
        h_inv[2]=  (h1h5-h2h4)/det;
        h_inv[5]= -(h0h5-h2h3)/det;
        h_inv[8]=  (h0h4-h1h3)/det;
    }
    return(det);
    //for (int i=0; i<9; i++) { cout<<h_inv[i]<<" "; } cout<<endl;
}


int bao_loadimage_ppm(char* filename,unsigned char *image,int h,int w,int *nr_channel)
{
	FILE * file_in; int nrc;
	char line[2048];
	int	i; int imax,hc,wc;	
	unsigned char *image_=image;
	file_in = fopen(filename,"rb");
	if(!file_in)
	{
		printf("Please check input filename: %s\n",filename);
		exit(0);
	}
	if(fgetc(file_in)=='P') 
		fscanf(file_in,"%d\n",&i);
	else
	{
		printf("Bad	header in ppm file.\n");
		exit(1);
	}
	while(fgets(line,2048,file_in)!=NULL)
	{
		if(line[0]=='#') continue;
		else
		{	
			sscanf(line, "%d %d\n",&wc,&hc);
			break;
		}
	}
	char str_tmp[100];
	switch (i)
	{
		case 5:
			fgets(str_tmp,100,file_in);
			imax=atoi(str_tmp);
			if(nr_channel!=NULL) (*nr_channel)=1;
			memset(image,0,sizeof(unsigned char)*h*w);
			fread(image,sizeof(unsigned char),h*w,file_in);
			break;
		case 6:
			fgets(str_tmp,100,file_in);
			imax=atoi(str_tmp);
			if(nr_channel!=NULL) (*nr_channel)=3;
			memset(image,0,sizeof(unsigned char)*h*w*3);
			fread(image,sizeof(unsigned char),h*w*3,file_in);
			break;
		case 2:
			fgets(str_tmp,100,file_in);
			imax=atoi(str_tmp);
			for(int y=0;y<h;y++) for(int x=0;x<w;x++)
			{
				//if(fscanf_s(file_in,"%d",&imax)!=1){printf("error in reading file.\n");getchar();exit(0);}
				fscanf(file_in,"%d",&imax);
				*image_++=imax;
			}
			break;
		case 3:
			fgets(str_tmp,100,file_in);
			imax=atoi(str_tmp);
			int cr,cg,cb;
			for(int y=0;y<h;y++) for(int x=0;x<w;x++)
			{
				//if(fscanf_s(file_in,"%d%d%d",&cr,&cg,&cb)!=3){printf("error in reading file.\n");getchar();exit(0);}
				fscanf(file_in,"%d%d%d",&cr,&cg,&cb);
				*image_++=cr; *image_++=cg; *image_++=cb; 
			}
			break;
		case 9:
			fgets(str_tmp,100,file_in);
			nrc=atoi(str_tmp);
			fscanf(file_in,"%d\n",&nrc);
			if(nr_channel!=NULL) (*nr_channel)=nrc;
			fgets(str_tmp,100,file_in);
			imax=atoi(str_tmp);
			fread(image,sizeof(unsigned char),h*w*nrc,file_in);
			break;
		default:
			printf("Can not open image [%s]!!\n",filename);
			break;					
	}
	fclose(file_in);
	return (0);
}
