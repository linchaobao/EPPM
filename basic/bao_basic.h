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

#ifndef _BAO_BASIC_H_
#define _BAO_BASIC_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>
//#include <type_traits> //for template, see if T is a specific type: (std::is_integral<unsigned char>::value)

#include <iostream>
using std::cout;
using std::endl;
using std::cin;

#include <algorithm>
using std::max;
using std::min;

#include <limits> //for type traits
using std::numeric_limits;

// min max macro
#ifndef __max
#define __max(a,b)  (((a) > (b)) ? (a) : (b))
#endif
#ifndef __min
#define __min(a,b)  (((a) < (b)) ? (a) : (b))
#endif

typedef double BAO_FLOAT;
#define BAO_ZERO  1e-16

template<typename T>
class BAO_DISP
{
public:
    T x;
    T y;
    BAO_DISP():x(0),y(0){}
    BAO_DISP(T xp,T yp):x(xp),y(yp){}
    //void operator=(BAO_DISP& another){x=another.x;y=another.y;}
    bool operator==(BAO_DISP& another){return (another.x==x && another.y==y);}
};

//#define BAO_NO_IMDBG   //uncomment this if not want to link imdebug


//////////////////////////////////////////////////////////////////////////
// Basic macro define
#define NOMINMAX  //remove the macro of max and min, use std library (<algorithm>)

//#include <WinDef.h> //for min max
// #ifndef NOMINMAX
// #ifndef max
// #define max(a,b)            (((a) > (b)) ? (a) : (b))
// #endif
// 
// #ifndef min
// #define min(a,b)            (((a) < (b)) ? (a) : (b))
// #endif
// #endif  /* NOMINMAX */

#define BAO_PI                     3.141592653589793
#define BAO_FLOAT_MAX              1.175494351e+38F
#define BAO_DOUBLE_MAX             1.7E+308
#define BAO_INT_MAX                2147483647
#define BAO_USHORT_MAX             65535
#define BAO_UCHAR_MAX              255
#define BAO_FLT_RELATIVE_ACCURACY  2.2204e-016
#define BAO_FLOAT_THRESH_ZERO      1e-6
#define BAO_DOUBLE_THRESH_ZERO     1e-16
#define BAO_MAX_STRLEN             300


//////////////////////////////////////////////////////////////////////////
// Memory allocate and free
#define BAO_MEM_PADDING  0  

template<typename T>
inline T* bao_alloc(int n)
{
    T* p;
    p=(T*) malloc(sizeof(T)*(n+BAO_MEM_PADDING));
    if (p==NULL) {printf("bao_alloc_1(): memory allocation (%d MB) failed.\n",sizeof(T)*(n+BAO_MEM_PADDING)/(1024*1024)); getchar(); exit(0); }
    return (p);
}

template<typename T>
inline void bao_free(T* &p)
{
    if (p!=NULL)
    {
        free(p);
        p=NULL;
    }
}

template<typename T>
inline T** bao_alloc(int r,int c)
{
    T *a,**p;
    a=(T*) malloc(sizeof(T)*(r*c+BAO_MEM_PADDING));
    if(a==NULL) {printf("bao_alloc_2(): memory allocation (%d MB) failed.\n",sizeof(T)*(r*c+BAO_MEM_PADDING)/(1024*1024)); getchar(); exit(0); }
    p=(T**) malloc(sizeof(T*)*r);
    for(int i=0;i<r;i++) p[i]= &a[i*c];
    return(p);
}

template<typename T>
inline void bao_free(T** &p)
{
    if(p!=NULL)
    {
        free(p[0]);
        free(p);
        p=NULL;
    }
}

template<typename T>
inline T*** bao_alloc(int n,int r,int c) //NOTE: if parameters are (r,c,n), it is also OK
{
    T *a,**p,***pp;
    int rc=r*c;
    int i,j;
    a=(T*) malloc(sizeof(T)*(n*rc+BAO_MEM_PADDING));
    if(a==NULL) {printf("bao_alloc_3(): memory allocation (%d MB) failed.\n",sizeof(T)*(n*rc+BAO_MEM_PADDING)/(1024*1024)); getchar(); exit(0); }
    p=(T**) malloc(sizeof(T*)*n*r);
    pp=(T***) malloc(sizeof(T**)*n);
    for(i=0;i<n;i++) 
        for(j=0;j<r;j++) 
            p[i*r+j]=&a[i*rc+j*c];
    for(i=0;i<n;i++) 
        pp[i]=&p[i*r];
    return(pp);
}

template<typename T>
inline void bao_free(T*** &p)
{
    if(p!=NULL)
    {
        free(p[0][0]);
        free(p[0]);
        free(p);
        p=NULL;
    }
}


//////////////////////////////////////////////////////////////////////////
// Pyramid memory operation
inline int bao_pyr_init_dim(int* &arrH, int* &arrW, int h, int w, BAO_FLOAT ratio=0.5f, int minW=20) 
{
    if (minW == 0) minW = 1;
    int nLevels = (int)(log(BAO_FLOAT(minW)/w)/log(ratio));
    if (nLevels<=0) nLevels = 1; //ensure at least 1 layers (original dim)
    arrH = bao_alloc<int>(nLevels);
    arrW = bao_alloc<int>(nLevels);
    arrH[0] = h;
    arrW[0] = w;
    for (int i=1; i<nLevels; i++)
    {
        arrH[i] = int(BAO_FLOAT(h)*pow(ratio,i));
        arrW[i] = int(BAO_FLOAT(w)*pow(ratio,i));
    }
    return nLevels;
}

inline int bao_pyr_init_dim(int* &arrH, int* &arrW, int h, int w, int maxDepth, BAO_FLOAT ratio=0.5f) 
{
    if (maxDepth == 0) maxDepth = 1;
    int nLevels = maxDepth;
    if (nLevels<=0) nLevels = 1; //ensure at least 1 layers (original dim)
    arrH = bao_alloc<int>(nLevels);
    arrW = bao_alloc<int>(nLevels);
    arrH[0] = h;
    arrW[0] = w;
    for (int i=1; i<nLevels; i++)
    {
        arrH[i] = int(BAO_FLOAT(h)*pow(ratio,i));
        arrW[i] = int(BAO_FLOAT(w)*pow(ratio,i));
    }
    return nLevels;
}

inline void bao_pyr_destroy_dim(int* &arrH, int* &arrW)
{
    bao_free(arrH);
    bao_free(arrW);
}

template<typename T>
inline T*** bao_pyr_alloc(int nLevels, int* arrH, int* arrW)
{
    T*** pPyr = (T***)malloc(nLevels*sizeof(T**));
    for (int i=0; i<nLevels; i++)
    {
        pPyr[i] = bao_alloc<T>(arrH[i],arrW[i]);
    }
    return pPyr;
}

template<typename T>
inline T**** bao_pyr_alloc(int nLevels, int* arrH, int* arrW, int nChannels)
{
    T**** pPyr = (T****)malloc(nLevels*sizeof(T***));
    for (int i=0; i<nLevels; i++)
    {
        pPyr[i] = bao_alloc<T>(arrH[i],arrW[i],nChannels);
    }
    return pPyr;
}

template<typename T>
inline void bao_pyr_free(T*** &p, int nLevels)
{
    if (p == NULL) return;
    for (int i=0; i<nLevels; i++)
    {
        bao_free(p[i]);
    }
    free(p); //DO NOT use bao_free!
    p = NULL;
}

template<typename T>
inline void bao_pyr_free(T**** &p, int nLevels)
{
    if (p == NULL) return;
    for (int i=0; i<nLevels; i++)
    {
        bao_free(p[i]);
    }
    free(p); //DO NOT use bao_free!
    p = NULL;
}


//////////////////////////////////////////////////////////////////////////
// Common tools
inline int bao_round(BAO_FLOAT in_x){if(in_x<0) return (int)(in_x-0.5); else return (int)(in_x+0.5);} //because VC does not have a round()

template<typename T1, typename T2>
inline T1 bao_type_cast(T2 val)
{
    BAO_FLOAT max_val = BAO_FLOAT(numeric_limits<T1>::max());
    BAO_FLOAT min_val = BAO_FLOAT(numeric_limits<T1>::min()); //NOTE: for float type, it is the minimum positive number!
    if (min_val>0) min_val = -max_val;
    return (T1)__max(min_val, __min(max_val, val));
}

template<typename T>
inline void bao_copy(T* img_out, T* img_in, int len){memcpy(img_out,img_in,sizeof(T)*len);}

template<typename T>
inline void bao_copy(T** img_out, T** img_in, int h, int w){memcpy(img_out[0],img_in[0],sizeof(T)*h*w);}

template<typename T>
inline void bao_copy(T*** img_out, T*** img_in, int h, int w, int d=3){memcpy(img_out[0][0],img_in[0][0],sizeof(T)*h*w*d);}

template<typename T1, typename T2>
inline void bao_copy(T1* img_out, T2* img_in, int len)
{
    if (sizeof(T1) < sizeof(T2)) //truncation
    {
        for (int i=0;i<len;i++) img_out[i]=bao_type_cast<T1>(img_in[i]);
    }
    else
    {
        for (int i=0;i<len;i++) img_out[i]=T1(img_in[i]);
    }
}

template<typename T1, typename T2>
inline void bao_copy(T1** img_out, T2** img_in, int h, int w)
{
    if (sizeof(T1) < sizeof(T2)) //truncation
    {
        for (int i=0;i<h;i++) for(int j=0;j<w;j++) 
            img_out[i][j]=bao_type_cast<T1>(img_in[i][j]);
    }
    else
    {
        for (int i=0;i<h;i++) for(int j=0;j<w;j++) img_out[i][j]=T1(img_in[i][j]);
    }
}

template<typename T1, typename T2>
inline void bao_copy(T1*** img_out, T2*** img_in, int h, int w, int d=3)
{
    if (sizeof(T1) < sizeof(T2)) //truncation
    {
        for (int i=0;i<h;i++) for(int j=0;j<w;j++) for (int k=0;k<d;k++)
            img_out[i][j][k]=bao_type_cast<T1>(img_in[i][j][k]);
    }
    else
    {
        for (int i=0;i<h;i++) for(int j=0;j<w;j++) for (int k=0;k<d;k++) img_out[i][j][k]=T1(img_in[i][j][k]);
    }
}

template<typename T>
inline void bao_memset(T** img_in, int h, int w){memset(img_in[0],0,sizeof(T)*h*w);}

template<typename T>
inline void bao_memset(T*** img_in, int h, int w, int d=3){memset(img_in[0][0],0,sizeof(T)*h*w*d);}

template<typename T1, typename T2>
inline void bao_add(T1**img_out, T2**img1, T2**img2, int h, int w){for(int y=0; y<h; y++) for (int x=0; x<w; x++) img_out[y][x] = T1(img1[y][x] + img2[y][x]);}

template<typename T1, typename T2>
inline void bao_add(T1***img_out, T2***img1, T2***img2, int h, int w, int c=3){for(int y=0; y<h; y++) for (int x=0; x<w; x++) for (int k=0; k<c; k++) img_out[y][x][k] = T1(img1[y][x][k] + img2[y][x][k]);}

template<typename T1, typename T2>
inline void bao_blending(T1**img_out, T2**img1, T2**img2, int h, int w, BAO_FLOAT weight1=.5f, BAO_FLOAT weight2=.5f){for(int y=0; y<h; y++) for (int x=0; x<w; x++) img_out[y][x] = T1(img1[y][x]*weight1 + img2[y][x]*weight2);}

template<typename T1, typename T2>
inline void bao_blending(T1***img_out, T2***img1, T2***img2, int h, int w, int c=3, BAO_FLOAT weight1=.5f, BAO_FLOAT weight2=.5f){for(int y=0; y<h; y++) for (int x=0; x<w; x++) for (int k=0; k<c; k++) img_out[y][x][k] = T1(img1[y][x][k]*weight1 + img2[y][x][k]*weight2);}

template<typename T1, typename T2>
inline void bao_minus(T1**img_out, T2**img1, T2**img2, int h, int w){for(int y=0; y<h; y++) for (int x=0; x<w; x++) img_out[y][x] = T1(img1[y][x] - img2[y][x]);}

template<typename T1, typename T2>
inline void bao_minus(T1***img_out, T2***img1, T2***img2, int h, int w, int c=3){for(int y=0; y<h; y++) for (int x=0; x<w; x++) for (int k=0; k<c; k++) img_out[y][x][k] = T1(img1[y][x][k] - img2[y][x][k]);}

template<typename T1, typename T2>
inline void bao_multiply_bypixel(T1**img_out, T2**img1, T2**img2, int h, int w){for(int y=0; y<h; y++) for (int x=0; x<w; x++) img_out[y][x] = T1(img1[y][x] * img2[y][x]);} //same as bao_vec_product_bypixel()

template<typename T1, typename T2>
inline void bao_multiply_bypixel(T1***img_out, T2***img1, T2***img2, int h, int w, int c=3){for(int y=0; y<h; y++) for (int x=0; x<w; x++) for (int k=0; k<c; k++) img_out[y][x][k] = T1(img1[y][x][k] * img2[y][x][k]);}

template<typename T1, typename T2>
inline void bao_multiply_scalar(T1**img_out, T2**img1, BAO_FLOAT scale, int h, int w){for(int y=0; y<h; y++) for (int x=0; x<w; x++) img_out[y][x] = T1(img1[y][x] * scale);} //same as bao_vec_product_bypixel()

template<typename T1, typename T2>
inline void bao_multiply_scalar(T1***img_out, T2***img1, BAO_FLOAT scale, int h, int w, int c=3){for(int y=0; y<h; y++) for (int x=0; x<w; x++) for (int k=0; k<c; k++) img_out[y][x][k] = T1(img1[y][x][k] * scale);}

template<typename T1, typename T2>
inline BAO_FLOAT bao_div(T1 x,T2 y){return (x/BAO_FLOAT(y+BAO_FLT_RELATIVE_ACCURACY));}

template<typename T1, typename T2>
double bao_psnr(T1 **a,T2 **b,int h,int w)
{
    T1 *pa;
    T2 *pb; 
    double ssn=0;
    pa=a[0];
    pb=b[0];
    for(int y=0;y<h;y++)
    {
        for(int x=0;x<w;x++)
        {
            double ab=double((*pa++)-(*pb++))/255;
            ssn+=ab*ab;
        }
    }
    if(ssn<BAO_FLOAT_THRESH_ZERO) return 999.0;
    ssn=log(double(h*w)/ssn)*10.0/log(10.0);
    return (ssn);
}

BAO_FLOAT bao_inv_3x3(BAO_FLOAT out[9],BAO_FLOAT in[9],BAO_FLOAT threshold=BAO_ZERO);

template<typename T1, typename T2>
inline void bao_vec_product_bypixel(T1* out, T2* a, T2* b, int len){for(int i=0;i<len;i++)*out++=T1(*a++)*T1(*b++);} //same as bao_multiply()

template<typename T1, typename T2>
inline T1 bao_vec_inner_product(T2* in1, T2* in2, int len){BAO_FLOAT res=0; for(int i=0;i<len;i++) res+=in1[i]*in2[i]; return T1(res);}

template<typename T1, typename T2>
inline T1 bao_vec_inner_product(T2** in1, T2** in2, int h, int w){BAO_FLOAT res=0; for (int i=0;i<h;i++) for(int j=0;j<w;j++) res+=in1[i][j]*in2[i][j]; return T1(res);} 

template<typename T1, typename T2>
inline T1 bao_vec_L2_norm(T2* in, int len){BAO_FLOAT res=0; for(int i=0;i<len;i++) res+=in[i]*in[i]; return T1(res);}

template<typename T1, typename T2>
inline T1 bao_vec_L2_norm(T2** in, int h, int w){BAO_FLOAT res=0; for (int i=0;i<h;i++) for(int j=0;j<w;j++) res+=in[i][j]*in[i][j]; return T1(res);}

template<typename T>
inline void bao_vec_minmax(T& min_val, T& max_val, T* vec, int len)
{
    max_val = vec[0];
    min_val = vec[0];
    for(int i=1; i<len; i++)
    {
        T cur_val = vec[i];
        if (cur_val>max_val) max_val = cur_val;
        if (cur_val<min_val) min_val = cur_val;
    }
}

template<typename T1, typename T2>
inline void bao_vec_rescale(T1* img_out, T2* img_in, int len, T1 new_max_val=1, T1 new_min_val=0, T2 ori_max_val=0, T2 ori_min_val=0)
{
    if (ori_max_val==ori_min_val) bao_vec_minmax(ori_min_val,ori_max_val,img_in,len);
    BAO_FLOAT ori_range_val = BAO_FLOAT(ori_max_val - ori_min_val);
    T1 new_range_val = new_max_val - new_min_val;
    if (ori_range_val == 0) 
    {
        bao_copy(img_out,img_in,len);
        return;
    }
    BAO_FLOAT factor = new_range_val/ori_range_val;
    for(int i=0; i<len; i++)
    {
        img_out[i] = bao_type_cast<T1>((img_in[i]-(BAO_FLOAT)ori_min_val)*factor + new_min_val);
    }
}

template<typename T2>
inline void bao_vec_rescale(BAO_FLOAT* img_out, T2* img_in, int len, BAO_FLOAT new_max_val=1, BAO_FLOAT new_min_val=0, T2 ori_max_val=0, T2 ori_min_val=0)
{
    if (ori_max_val==ori_min_val) bao_vec_minmax(ori_min_val,ori_max_val,img_in,len);
    BAO_FLOAT ori_range_val = BAO_FLOAT(ori_max_val - ori_min_val);
    BAO_FLOAT new_range_val = new_max_val - new_min_val;
    if (ori_range_val == 0) 
    {
        bao_copy(img_out,img_in,len);
        return;
    }
    BAO_FLOAT factor = new_range_val/ori_range_val;
    for(int i=0; i<len; i++)
    {
        img_out[i] = (BAO_FLOAT)((img_in[i]-(BAO_FLOAT)ori_min_val)*factor + new_min_val);
    }
}

template<typename T>
inline T bao_dist_rgb_max(T* a, T* b) {T x,y,z; x=abs(a[0]-b[0]); y=abs(a[1]-b[1]); z=abs(a[2]-b[2]); return(max(max(x,y),z));}

template<typename T1, typename T2>
inline void bao_multichannels_average(T1** img_out, T2*** img_in, int h, int w, int c=3, BAO_FLOAT* weights=NULL)
{
    if (weights == NULL)
    {
        for (int i=0;i<h;i++) for(int j=0;j<w;j++) 
        {
            BAO_FLOAT val = 0;
            for (int k=0; k<c; k++) val += img_in[i][j][k];
            img_out[i][j]=bao_type_cast<T1>(val/BAO_FLOAT(c));
        }
    }
    else
    {
        for (int i=0;i<h;i++) for(int j=0;j<w;j++) 
        {
            BAO_FLOAT val = 0;
            BAO_FLOAT sum_weight = BAO_ZERO;
            for (int k=0; k<c; k++) {val += img_in[i][j][k]*weights[k]; sum_weight += weights[k];}
            img_out[i][j]=bao_type_cast<T1>(val/sum_weight);
        }
    }
}

template<typename T>
inline void bao_swap_ptr(T*& p1, T*& p2)
{
    T* temp = p1;
    p1 = p2;
    p2 = temp;
}

template<typename T>
inline void bao_swap_ptr(T**& p1, T**& p2)
{
    T** temp = p1;
    p1 = p2;
    p2 = temp;
}

template<typename T>
inline void bao_swap_ptr(T***& p1, T***& p2)
{
    T*** temp = p1;
    p1 = p2;
    p2 = temp;
}

template<typename T1, typename T2>
inline void bao_deriv_x(T1**img_out, T2**img_in, int h, int w, bool use_five_points=false)
{
    if (use_five_points)
    {
        BAO_FLOAT xFilter[5]={1.f/12.f, -8.f/12.f, 0, 8.f/12.f, -1.f/12.f};
        BAO_FLOAT val;
        for(int y=0; y<h; y++) for (int x=0; x<w; x++)
        {
            val = 0.f;
            for (int dx=-2; dx<=2; dx++)
            {
                int cx = __max(0,__min(w-1,x+dx));
                val += img_in[y][cx] * xFilter[2+dx];
            }
            img_out[y][x] = T1(val);
        }
    }
    else
    {
        for(int y=0; y<h; y++) for (int x=0; x<w-1; x++)
        {
            img_out[y][x] = T1(img_in[y][x+1] - img_in[y][x]);
        }
    }
}

template<typename T1, typename T2>
inline void bao_deriv_x(T1***img_out, T2***img_in, int h, int w, int c=3, bool use_five_points=false)
{
    if (use_five_points)
    {
        BAO_FLOAT xFilter[5]={1.f/12.f, -8.f/12.f, 0, 8.f/12.f, -1.f/12.f};
        BAO_FLOAT val;
        for(int y=0; y<h; y++) for (int x=0; x<w; x++) for (int k=0; k<c; k++)
        {
            val = 0.f;
            for (int dx=-2; dx<=2; dx++)
            {
                int cx = __max(0,__min(w-1,x+dx));
                val += img_in[y][cx][k] * xFilter[2+dx];
            }
            img_out[y][x][k] = T1(val);
        }
    }
    else
    {
        for(int y=0; y<h; y++) for (int x=0; x<w-1; x++) for (int k=0; k<c; k++)
        {
            img_out[y][x][k] = T1(img_in[y][x+1][k] - img_in[y][x][k]);
        }
    }
}

template<typename T1, typename T2>
inline void bao_deriv_y(T1**img_out, T2**img_in, int h, int w, bool use_five_points=false)
{
    if (use_five_points)
    {
        double yFilter[5]={1.f/12.f, -8.f/12.f, 0, 8.f/12.f, -1.f/12.f};
        BAO_FLOAT val;
        for(int y=0; y<h; y++) for (int x=0; x<w; x++)
        {
            val = 0.f;
            for (int dy=-2; dy<=2; dy++)
            {
                int cy = __max(0,__min(h-1,y+dy));
                val += img_in[cy][x] * yFilter[2+dy];
            }
            img_out[y][x] = T1(val);
        }
    }
    else
    {
        for(int y=0; y<h-1; y++) for (int x=0; x<w; x++)
        {
            img_out[y][x] = T1(img_in[y+1][x] - img_in[y][x]);
        }
    }
}

template<typename T1, typename T2>
inline void bao_deriv_y(T1***img_out, T2***img_in, int h, int w, int c=3, bool use_five_points=false)
{
    if (use_five_points)
    {
        double yFilter[5]={1.f/12.f, -8.f/12.f, 0, 8.f/12.f, -1.f/12.f};
        BAO_FLOAT val;
        for(int y=0; y<h; y++) for (int x=0; x<w; x++) for (int k=0; k<c; k++)
        {
            val = 0.f;
            for (int dy=-2; dy<=2; dy++)
            {
                int cy = __max(0,__min(h-1,y+dy));
                val += img_in[cy][x][k] * yFilter[2+dy];
            }
            img_out[y][x][k] = T1(val);
        }
    }
    else
    {
        for(int y=0; y<h-1; y++) for (int x=0; x<w; x++) for (int k=0; k<c; k++)
        {
            img_out[y][x][k] = T1(img_in[y+1][x][k] - img_in[y][x][k]);
        }
    }
}


//////////////////////////////////////////////////////////////////////////
// Color space conversion
template<typename T>
inline T bao_rgb2gray_pixel(T*pixval)
{
    return bao_type_cast<T>(pixval[0]*0.299+pixval[1]*0.587+pixval[2]*0.114);
}

template<typename T1, typename T2>
inline void bao_rgb2gray(T1** img_out, T2*** img_in, int h, int w)
{
    for (int i=0;i<h;i++) for(int j=0;j<w;j++) img_out[i][j]=bao_type_cast<T1>(img_in[i][j][0]*0.299+img_in[i][j][1]*0.587+img_in[i][j][2]*0.114);
}

template<typename T1, typename T2>
inline void bao_rgb2luv_pixel(T1* luv, T2* rgb)
{
//     const BAO_FLOAT _rgb_2_cie_xn=0.95050;
    const BAO_FLOAT _rgb_2_cie_yn=1.00000;
//     const BAO_FLOAT _rgb_2_cie_zn=1.08870;
    const BAO_FLOAT _rgb_2_cie_un_prime=0.19784977571475;
    const BAO_FLOAT _rgb_2_cie_vn_prime=0.46834507665248;
    const BAO_FLOAT _rgb_2_cie_lt=0.008856;
    const BAO_FLOAT _rgb_2_cie_xyz[3][3]={{0.4125,0.3576,0.1804},{0.2125,0.7154,0.0721},{0.0193,0.1192,0.9502}};
//     const BAO_FLOAT _rgb_2_cie_rgb[3][3]={{3.2405,-1.5371,-0.4985},{-0.9693,1.8760,0.0416},{0.0556, -0.2040,1.0573}};

    BAO_FLOAT luv_f[3];
    BAO_FLOAT x,y,z,l0,u_prime,v_prime,constant;
    x=_rgb_2_cie_xyz[0][0]*rgb[0]+_rgb_2_cie_xyz[0][1]*rgb[1]+_rgb_2_cie_xyz[0][2]*rgb[2];
    y=_rgb_2_cie_xyz[1][0]*rgb[0]+_rgb_2_cie_xyz[1][1]*rgb[1]+_rgb_2_cie_xyz[1][2]*rgb[2];
    z=_rgb_2_cie_xyz[2][0]*rgb[0]+_rgb_2_cie_xyz[2][1]*rgb[1]+_rgb_2_cie_xyz[2][2]*rgb[2];
    /*compute L**/
    l0=y/(255.0*_rgb_2_cie_yn);
    if(l0>_rgb_2_cie_lt) luv_f[0]=(BAO_FLOAT)(116.0*(pow(l0,1.0/3.0))-16.0);
    else luv_f[0]=(BAO_FLOAT)(903.3*l0);
    /*compute u_prime and v_prime*/
    constant=x+15*y+3*z;
    if(constant!=0)
    {
        u_prime=(4 * x)/constant;
        v_prime=(9 * y)/constant;
    }
    else
    {
        u_prime=4.0;
        v_prime=9.0/15.0;
    }
    /*compute u* and v**/
    luv_f[1]=(BAO_FLOAT)(13*luv_f[0]*(u_prime-_rgb_2_cie_un_prime));
    luv_f[2]=(BAO_FLOAT)(13*luv_f[0]*(v_prime-_rgb_2_cie_vn_prime));

    luv[0]=T1(luv_f[0]*(255.f/100.f));
    luv[1]=T1(__min(BAO_FLOAT((luv_f[1]+83.138f)*255.0f/258.343f),255.f));
    luv[2]=T1(__min((luv_f[2]+134.104f)*255.0f/241.518f,255.f));
}

template<typename T1, typename T2>
inline void bao_rgb2luv(T1*** luv, T2*** rgb, int h, int w) //assume 3 channels
{
    int i; T1 *luv_; T2 *rgb_;
    luv_=&luv[0][0][0]; rgb_=&rgb[0][0][0];
    for(i=0;i<h*w;i++) {bao_rgb2luv_pixel(luv_++,rgb_++); luv_++; luv_++; rgb_++; rgb_++;}
}

template<typename T1, typename T2>
inline void bao_luv2rgb_pixel(T1 *rgb, T2 *luv)
{
//     const BAO_FLOAT _rgb_2_cie_xn=0.95050;
    const BAO_FLOAT _rgb_2_cie_yn=1.00000;
//     const BAO_FLOAT _rgb_2_cie_zn=1.08870;
    const BAO_FLOAT _rgb_2_cie_un_prime=0.19784977571475;
    const BAO_FLOAT _rgb_2_cie_vn_prime=0.46834507665248;
//     const BAO_FLOAT _rgb_2_cie_lt=0.008856;
//     const BAO_FLOAT _rgb_2_cie_xyz[3][3]={{0.4125,0.3576,0.1804},{0.2125,0.7154,0.0721},{0.0193,0.1192,0.9502}};
    const BAO_FLOAT _rgb_2_cie_rgb[3][3]={{3.2405,-1.5371,-0.4985},{-0.9693,1.8760,0.0416},{0.0556, -0.2040,1.0573}};

    BAO_FLOAT luvf[3];
    luvf[0]=BAO_FLOAT(luv[0])*100.0f/255.f; 
    luvf[1]=BAO_FLOAT(luv[1])*354.0f/255.f-134; 
    luvf[2]=BAO_FLOAT(luv[2])*262.0f/255.f-140; 
    //declare variables...
    int r,g,b; BAO_FLOAT x,y,z,u_prime,v_prime;
    //perform conversion
    if(luvf[0]<0.1) r=g=b=0;
    else
    {
        //convert luv to xyz...
        if(luvf[0]<8.0) y=_rgb_2_cie_yn*luvf[0]/903.3;
        else
        {
            y=(luvf[0]+16.0)/116.0;
            y*=_rgb_2_cie_yn*y*y;
        }
        u_prime=luvf[1]/(13*luvf[0])+_rgb_2_cie_un_prime;
        v_prime=luvf[2]/(13*luvf[0])+_rgb_2_cie_vn_prime;
        x=9*u_prime*y/(4*v_prime);
        z=(12-3*u_prime-20*v_prime)*y/(4*v_prime);
        //convert xyz to rgb...
        r=bao_round((_rgb_2_cie_rgb[0][0]*x+_rgb_2_cie_rgb[0][1]*y+_rgb_2_cie_rgb[0][2]*z)*255.0);
        g=bao_round((_rgb_2_cie_rgb[1][0]*x+_rgb_2_cie_rgb[1][1]*y+_rgb_2_cie_rgb[1][2]*z)*255.0);
        b=bao_round((_rgb_2_cie_rgb[2][0]*x+_rgb_2_cie_rgb[2][1]*y+_rgb_2_cie_rgb[2][2]*z)*255.0);
        //check bounds...
        if(r<0)	r=0; if(r>255)	r=255;
        if(g<0)	g=0; if(g>255)	g=255;
        if(b<0)	b=0; if(b>255)	b=255;
    }
    //assign rgb values to rgb vector rgb
    rgb[0]=r;
    rgb[1]=g;
    rgb[2]=b;
}

template<typename T1, typename T2>
inline void bao_luv2rgb(T1*** rgb, T2*** luv, int h, int w) //assume 3 channels
{
    int i; T1 *rgb_; T2 *luv_; 
    rgb_=&rgb[0][0][0]; luv_=&luv[0][0][0]; 
    for(i=0;i<h*w;i++) {bao_luv2rgb_pixel(rgb_++,luv_++); luv_++; luv_++; rgb_++; rgb_++;}
}

template<typename T1, typename T2>
inline void bao_rgb2yuv_pixel(T1 *out, T2 *in_x)
{
    BAO_FLOAT wr,wg,wb,wru,wgu,wbu,wrv,wgv,wbv,uvmax;
    wr=0.299f; wg=0.587f; wb=0.114f;
    wru=-0.169f; wgu=-0.331f; wbu=0.5f;
    wrv=0.5f; wgv=-0.419f; wbv=-0.081f;
    uvmax=128;
    out[0] = T1(__min(255.f,__max(0.f,wr*in_x[0]+wg*in_x[1]+wb*in_x[2]+0.5f)));
    out[1] = T1(__min(255.f,__max(0.f,wru*in_x[0]+wgu*in_x[1]+wbu*in_x[2]+uvmax+0.5f)));
    out[2] = T1(__min(255.f,__max(0.f,wrv*in_x[0]+wgv*in_x[1]+wbv*in_x[2]+uvmax+0.5f)));
}

template<typename T1, typename T2>
inline void bao_rgb2yuv(T1 ***out, T2 ***in, int h, int w)
{
    T1 *out_x=in[0][0];
    T2 *in_x=out[0][0];
    for(int yi=0;yi<h;yi++) for(int x=0;x<w;x++){bao_rgb2yuv_pixel(out_x,in_x); out_x++; out_x++; out_x++; in_x++; in_x++; in_x++;}
}

template<typename T1, typename T2>
inline void bao_yuv2rgb_pixel(T1 *out, T2 *in_x)
{
    BAO_FLOAT r,g,b,wb,wgu,wbu,wgv,uvmax;
    wb=1.4f; 
    wgu=-0.3437f; wbu=-0.71417f;
    wgv=1.77f;
    uvmax=128;
    BAO_FLOAT uu = BAO_FLOAT(in_x[1])-uvmax;
    BAO_FLOAT vv = BAO_FLOAT(in_x[2])-uvmax;
    r=in_x[0]+wb*vv;
    g=in_x[0]+wgu*uu+wbu*vv;
    b=in_x[0]+wgv*uu;
    out[0]=T1(__min(255.f,__max(0.f,r+0.5f)));
    out[1]=T1(__min(255.f,__max(0.f,g+0.5f)));
    out[2]=T1(__min(255.f,__max(0.f,b+0.5f)));
}

template<typename T1, typename T2>
inline void bao_yuv2rgb(T1 ***out, T2 ***in, int h, int w)
{
    T1 *out_x=out[0][0];
    T2 *in_x=in[0][0];
    for(int yi=0;yi<h;yi++) for(int x=0;x<w;x++){bao_yuv2rgb_pixel(out_x,in_x); out_x++; out_x++; out_x++; in_x++; in_x++; in_x++;}
}


//////////////////////////////////////////////////////////////////////////
// Image resize
inline BAO_FLOAT _bao_w0(BAO_FLOAT a){return (1.0f/6.0f)*(a*(a*(-a + 3.0f) - 3.0f) + 1.0f);}
inline BAO_FLOAT _bao_w1(BAO_FLOAT a){return (1.0f/6.0f)*(a*a*(3.0f*a - 6.0f) + 4.0f);}
inline BAO_FLOAT _bao_w2(BAO_FLOAT a){return (1.0f/6.0f)*(a*(a*(-3.0f*a + 3.0f) + 3.0f) + 1.0f);}
inline BAO_FLOAT _bao_w3(BAO_FLOAT a){return (1.0f/6.0f)*(a*a*a);}

inline BAO_FLOAT bao_cubic_filter(BAO_FLOAT x, BAO_FLOAT c0, BAO_FLOAT c1, BAO_FLOAT c2, BAO_FLOAT c3)
{
    BAO_FLOAT r;
    r = c0 * _bao_w0(x);
    r += c1 * _bao_w1(x);
    r += c2 * _bao_w2(x);
    r += c3 * _bao_w3(x);
    return r;
}

template<typename T1, typename T2>
T1 bao_bicubic_interp_pixel(T2**img,BAO_FLOAT x,BAO_FLOAT y,int h,int w)
{
    int px = (int)(x);
    int py = (int)(y);
    float fx = x - px;
    float fy = y - py;

    if (fx == 0 && fy == 0) return img[py][px];
    else return bao_type_cast<T1>(bao_cubic_filter(fy,
        bao_cubic_filter(fx, (BAO_FLOAT)img[__max(0,__min(h-1,py-1))][__max(0,__min(w-1,px-1))], 
        (BAO_FLOAT)img[__max(0,__min(h-1,py-1))][__max(0,__min(w-1,px))], 
        (BAO_FLOAT)img[__max(0,__min(h-1,py-1))][__max(0,__min(w-1,px+1))], 
        (BAO_FLOAT)img[__max(0,__min(h-1,py-1))][__max(0,__min(w-1,px+2))]),
        bao_cubic_filter(fx, (BAO_FLOAT)img[__max(0,__min(h-1,py))][__max(0,__min(w-1,px-1))],   
        (BAO_FLOAT)img[__max(0,__min(h-1,py))][__max(0,__min(w-1,px))],   
        (BAO_FLOAT)img[__max(0,__min(h-1,py))][__max(0,__min(w-1,px+1))],   
        (BAO_FLOAT)img[__max(0,__min(h-1,py))][__max(0,__min(w-1,px+2))]),
        bao_cubic_filter(fx, (BAO_FLOAT)img[__max(0,__min(h-1,py+1))][__max(0,__min(w-1,px-1))], 
        (BAO_FLOAT)img[__max(0,__min(h-1,py+1))][__max(0,__min(w-1,px))], 
        (BAO_FLOAT)img[__max(0,__min(h-1,py+1))][__max(0,__min(w-1,px+1))], 
        (BAO_FLOAT)img[__max(0,__min(h-1,py+1))][__max(0,__min(w-1,px+2))]),
        bao_cubic_filter(fx, (BAO_FLOAT)img[__max(0,__min(h-1,py+2))][__max(0,__min(w-1,px-1))], 
        (BAO_FLOAT)img[__max(0,__min(h-1,py+2))][__max(0,__min(w-1,px))], 
        (BAO_FLOAT)img[__max(0,__min(h-1,py+2))][__max(0,__min(w-1,px+1))], 
        (BAO_FLOAT)img[__max(0,__min(h-1,py+2))][__max(0,__min(w-1,px+2))])
        ));
}

template<typename T1, typename T2>
T1 bao_bicubic_interp_pixel(T2***img,BAO_FLOAT x,BAO_FLOAT y,int z,int h,int w,int c)
{
    int px = (int)(x);
    int py = (int)(y);
    float fx = x - px;
    float fy = y - py;

    if (fx == 0 && fy == 0) return img[py][px][z];
    else return bao_type_cast<T1>(bao_cubic_filter(fy,
        bao_cubic_filter(fx, (BAO_FLOAT)img[__max(0,__min(h-1,py-1))][__max(0,__min(w-1,px-1))][z], 
        (BAO_FLOAT)img[__max(0,__min(h-1,py-1))][__max(0,__min(w-1,px))][z], 
        (BAO_FLOAT)img[__max(0,__min(h-1,py-1))][__max(0,__min(w-1,px+1))][z], 
        (BAO_FLOAT)img[__max(0,__min(h-1,py-1))][__max(0,__min(w-1,px+2))][z]),
        bao_cubic_filter(fx, (BAO_FLOAT)img[__max(0,__min(h-1,py))][__max(0,__min(w-1,px-1))][z],   
        (BAO_FLOAT)img[__max(0,__min(h-1,py))][__max(0,__min(w-1,px))][z],   
        (BAO_FLOAT)img[__max(0,__min(h-1,py))][__max(0,__min(w-1,px+1))][z],   
        (BAO_FLOAT)img[__max(0,__min(h-1,py))][__max(0,__min(w-1,px+2))][z]),
        bao_cubic_filter(fx, (BAO_FLOAT)img[__max(0,__min(h-1,py+1))][__max(0,__min(w-1,px-1))][z], 
        (BAO_FLOAT)img[__max(0,__min(h-1,py+1))][__max(0,__min(w-1,px))][z], 
        (BAO_FLOAT)img[__max(0,__min(h-1,py+1))][__max(0,__min(w-1,px+1))][z], 
        (BAO_FLOAT)img[__max(0,__min(h-1,py+1))][__max(0,__min(w-1,px+2))][z]),
        bao_cubic_filter(fx, (BAO_FLOAT)img[__max(0,__min(h-1,py+2))][__max(0,__min(w-1,px-1))][z], 
        (BAO_FLOAT)img[__max(0,__min(h-1,py+2))][__max(0,__min(w-1,px))][z], 
        (BAO_FLOAT)img[__max(0,__min(h-1,py+2))][__max(0,__min(w-1,px+1))][z], 
        (BAO_FLOAT)img[__max(0,__min(h-1,py+2))][__max(0,__min(w-1,px+2))][z])
        ));
}

template<typename T1, typename T2>
void bao_bicubic_resize(T1 **out,T2 **in,int h,int w,BAO_FLOAT scale) //h and w are target dimensions
{
    int hv,wu;
    BAO_FLOAT div_scale=1.f/scale;
    hv=int(h*div_scale);
    wu=int(w*div_scale);
    for(int y=0;y<h;y++) for(int x=0;x<w;x++)
    {
//         float u=x*div_scale;
//         float v=y*div_scale;
        BAO_FLOAT u=(BAO_FLOAT)(x+1)*div_scale-1;
        BAO_FLOAT v=(BAO_FLOAT)(y+1)*div_scale-1;
        out[y][x]=bao_bicubic_interp_pixel<T1,T2>(in,u,v,hv,wu);
    }
}

template<typename T1, typename T2>
void bao_bicubic_resize(T1 **out,T2 **in,int h,int w,int inH,int inW) //h and w are target dimensions
{
    BAO_FLOAT div_scale=BAO_FLOAT(inW)/w;
    for(int y=0;y<h;y++) for(int x=0;x<w;x++)
    {
//         float u=x*div_scale;
//         float v=y*div_scale;
        BAO_FLOAT u=(BAO_FLOAT)(x+1)*div_scale-1;
        BAO_FLOAT v=(BAO_FLOAT)(y+1)*div_scale-1;
        out[y][x]=bao_bicubic_interp_pixel<T1,T2>(in,u,v,inH,inW);
    }
}

template<typename T1, typename T2>
void bao_bicubic_resize(T1 ***out,T2 ***in,int h,int w,int c,BAO_FLOAT scale) //h and w are target dimensions
{
    int hv,wu;
    BAO_FLOAT div_scale=1.f/scale;
    hv=int(h*div_scale);
    wu=int(w*div_scale);
    for(int y=0;y<h;y++) for(int x=0;x<w;x++)
    {
//         float u=x*div_scale;
//         float v=y*div_scale;
        BAO_FLOAT u=(BAO_FLOAT)(x+1)*div_scale-1;
        BAO_FLOAT v=(BAO_FLOAT)(y+1)*div_scale-1;
        for(int k=0;k<c;k++) out[y][x][k]=bao_bicubic_interp_pixel<T1,T2>(in,u,v,k,hv,wu,c);
    }
}

template<typename T1, typename T2>
void bao_bicubic_resize(T1 ***out,T2 ***in,int h,int w,int inH,int inW,int c) //h and w are target dimensions
{
    BAO_FLOAT div_scale=BAO_FLOAT(inW)/w;
    for(int y=0;y<h;y++) for(int x=0;x<w;x++)
    {
//         float u=x*div_scale;
//         float v=y*div_scale;
        BAO_FLOAT u=(BAO_FLOAT)(x+1)*div_scale-1;
        BAO_FLOAT v=(BAO_FLOAT)(y+1)*div_scale-1;
        for(int k=0;k<c;k++) out[y][x][k]=bao_bicubic_interp_pixel<T1,T2>(in,u,v,k,inH,inW,c);
    }
}

template<typename T1, typename T2>
T1 bao_bilinear_interp_pixel(T2**img,BAO_FLOAT x,BAO_FLOAT y,int h,int w)
{
    int xx,yy,m,n,u,v;
    xx=x;
    yy=y;
    BAO_FLOAT dx,dy,s;
    dx=__max(__min(x-xx,1),0);
    dy=__max(__min(y-yy,1),0);

    BAO_FLOAT res = 0;
    for(m=0;m<=1;m++) for(n=0;n<=1;n++)
    {
        u=__max(0,__min(w-1,xx+m));
        v=__max(0,__min(h-1,yy+n));
        s=fabs(1-m-dx)*fabs(1-n-dy);
        res += (img[v][u]*s);
    }
    return res;
}

template<typename T1, typename T2>
T1 bao_bilinear_interp_pixel(T2***img,BAO_FLOAT x,BAO_FLOAT y,int z,int h,int w,int c)
{
    int xx,yy,m,n,u,v;
    xx=x;
    yy=y;
    BAO_FLOAT dx,dy,s;
    dx=__max(__min(x-xx,1),0);
    dy=__max(__min(y-yy,1),0);

    BAO_FLOAT res = 0;
    for(m=0;m<=1;m++) for(n=0;n<=1;n++)
    {
        u=__max(0,__min(w-1,xx+m));
        v=__max(0,__min(h-1,yy+n));
        s=fabs(1-m-dx)*fabs(1-n-dy);
        res += (img[v][u][z]*s);
    }
    return res;
}

template<typename T1, typename T2>
void bao_bilinear_resize(T1 **out,T2 **in,int h,int w,BAO_FLOAT scale) //h and w are target dimensions
{
    int hv,wu;
    BAO_FLOAT div_scale=1.f/scale;
    hv=int(h*div_scale);
    wu=int(w*div_scale);
    for(int y=0;y<h;y++) for(int x=0;x<w;x++)
    {
//         float u=x*div_scale;
//         float v=y*div_scale;
        BAO_FLOAT u=(BAO_FLOAT)(x+1)*div_scale-1;
        BAO_FLOAT v=(BAO_FLOAT)(y+1)*div_scale-1;
        out[y][x]=bao_bilinear_interp_pixel<T1,T2>(in,u,v,hv,wu);
    }
}

template<typename T1, typename T2>
void bao_bilinear_resize(T1 **out,T2 **in,int h,int w,int inH,int inW) //h and w are target dimensions
{
    BAO_FLOAT div_scale=BAO_FLOAT(inW)/w;
    for(int y=0;y<h;y++) for(int x=0;x<w;x++)
    {
//         float u=x*div_scale;
//         float v=y*div_scale;
        BAO_FLOAT u=(BAO_FLOAT)(x+1)*div_scale-1;
        BAO_FLOAT v=(BAO_FLOAT)(y+1)*div_scale-1;
        out[y][x]=bao_bilinear_interp_pixel<T1,T2>(in,u,v,inH,inW);
    }
}

template<typename T1, typename T2>
void bao_bilinear_resize(T1 ***out,T2 ***in,int h,int w,int c,BAO_FLOAT scale) //h and w are target dimensions
{
    int hv,wu;
    BAO_FLOAT div_scale=1.f/scale;
    hv=int(h*div_scale);
    wu=int(w*div_scale);
    for(int y=0;y<h;y++) for(int x=0;x<w;x++)
    {
//         float u=x*div_scale;
//         float v=y*div_scale;
        BAO_FLOAT u=(BAO_FLOAT)(x+1)*div_scale-1;
        BAO_FLOAT v=(BAO_FLOAT)(y+1)*div_scale-1;
        for(int k=0;k<c;k++) out[y][x][k]=bao_bilinear_interp_pixel<T1,T2>(in,u,v,k,hv,wu,c);
    }
}

template<typename T1, typename T2>
void bao_bilinear_resize(T1 ***out,T2 ***in,int h,int w,int inH,int inW,int c) //h and w are target dimensions
{
    BAO_FLOAT div_scale=BAO_FLOAT(inW)/w;
    for(int y=0;y<h;y++) for(int x=0;x<w;x++)
    {
//         float u=x*div_scale;
//         float v=y*div_scale;
        BAO_FLOAT u=(BAO_FLOAT)(x+1)*div_scale-1;
        BAO_FLOAT v=(BAO_FLOAT)(y+1)*div_scale-1;
        for(int k=0;k<c;k++) out[y][x][k]=bao_bilinear_interp_pixel<T1,T2>(in,u,v,k,inH,inW,c);
    }
}


//////////////////////////////////////////////////////////////////////////
// CPU timer
#ifndef _WIN32
#include <sys/time.h>
#endif

class bao_timer_cpu
{
public: 
    void start(); 
    double stop(); 
    double time_display(char* disp="", int nr_frame=1); 
    double fps_display(char* disp="", int nr_frame=1); 
private: 
#ifdef _WIN32
    double m_pc_frequency; 
    __int64 m_counter_start;
#else
    struct timeval timerStart;
#endif
}; 



//////////////////////////////////////////////////////////////////////////
// Gaussian pyramid operation
template<typename T1, typename T2>
inline void _bao_downsample_5tap(T1** out_img, T2** in_img, int h, int w) //only for 0.5 downsample ratio, out_img can only be float point!
{
    BAO_FLOAT filter_kernel[5][5] = {
        {0.0025, 0.0125, 0.0200, 0.0125, 0.0025},
        {0.0125, 0.0625, 0.1000, 0.0625, 0.0125},
        {0.0200, 0.1000, 0.1600, 0.1000, 0.0200},
        {0.0125, 0.0625, 0.1000, 0.0625, 0.0125},
        {0.0025, 0.0125, 0.0200, 0.0125, 0.0025}}; //see <Burt and Adelson 1983>

        bao_memset(out_img,h,w);

        //main body
        for (int y=1; y<h-1; y++) for (int x=1; x<w-1; x++)
        {
            //convolution and downsampling
            for (int m=-2; m<=2; m++) for (int n=-2; n<=2; n++)
            {
                out_img[y][x] += (BAO_FLOAT(in_img[2*y+m][2*x+n]) * filter_kernel[m+2][n+2]);
            }
        }

        //first and last column
        for (int y=1; y<h-1; y++)
        {
            for (int m=-2; m<=2; m++) for (int n=0; n<=2; n++)
            {
                out_img[y][0] += BAO_FLOAT(in_img[2*y+m][n]) * filter_kernel[m+2][n+2];
            }
            for (int m=-2; m<=2; m++) for (int n=-2; n<=0; n++)
            {
                out_img[y][w-1] += BAO_FLOAT(in_img[2*y+m][2*(w-1)+n]) * filter_kernel[m+2][n+2];
            }
            out_img[y][0] /= 0.7; //normalization
            out_img[y][w-1] /= 0.7; //normalization
        }

        //first and last row
        for (int x=1; x<w-1; x++)
        {
            for (int m=0; m<=2; m++) for (int n=-2; n<=2; n++)
            {
                out_img[0][x] += BAO_FLOAT(in_img[m][2*x+n]) * filter_kernel[m+2][n+2];
            }
            for (int m=-2; m<=0; m++) for (int n=-2; n<=2; n++)
            {
                out_img[h-1][x] += BAO_FLOAT(in_img[2*(h-1)+m][2*x+n]) * filter_kernel[m+2][n+2];
            }
            out_img[0][x] /= 0.7; //normalization
            out_img[h-1][x] /= 0.7; //normalization
        }

        //four corners
        for (int m=0; m<=2; m++) for (int n=0; n<=2; n++)
        {
            out_img[0][0] += BAO_FLOAT(in_img[m][n]) * filter_kernel[m+2][n+2];
        }
        out_img[0][0] /= 0.49;

        for (int m=-2; m<=0; m++) for (int n=0; n<=2; n++)
        {
            out_img[h-1][0] += BAO_FLOAT(in_img[2*(h-1)+m][n]) * filter_kernel[m+2][n+2];
        }
        out_img[h-1][0] /= 0.49;

        for (int m=0; m<=2; m++) for (int n=-2; n<=0; n++)
        {
            out_img[0][w-1] += BAO_FLOAT(in_img[m][2*(w-1)+n]) * filter_kernel[m+2][n+2];            
        }
        out_img[0][w-1] /= 0.49;

        for (int m=-2; m<=0; m++) for (int n=-2; n<=0; n++)
        {
            out_img[h-1][w-1] += BAO_FLOAT(in_img[2*(h-1)+m][2*(w-1)+n]) * filter_kernel[m+2][n+2];
        }
        out_img[h-1][w-1] /= 0.49;
}


template<typename T1, typename T2>
inline void _bao_upsample_5tap(T1** out_img, T2** in_img, int h, int w) //only for 0.5 downsample ratio, out_img can only be float point!
{
    BAO_FLOAT filter_kernel[5][5] = {
        {0.0025, 0.0125, 0.0200, 0.0125, 0.0025},
        {0.0125, 0.0625, 0.1000, 0.0625, 0.0125},
        {0.0200, 0.1000, 0.1600, 0.1000, 0.0200},
        {0.0125, 0.0625, 0.1000, 0.0625, 0.0125},
        {0.0025, 0.0125, 0.0200, 0.0125, 0.0025}}; //see <Burt and Adelson 1983>

        bao_memset(out_img,h,w);

        //filtering main body (interpolation)
        for (int y=2; y<h-2; y+=2) for (int x=2; x<w-2; x+=2)
        {
            //upsampling and convolution
            for (int m=-2; m<=2; m+=2) for (int n=-2; n<=2; n+=2)
            {
                out_img[y][x] += (in_img[(y-m)/2][(x-n)/2] * filter_kernel[m+2][n+2]);
            }
            out_img[y][x] /= 0.25; //normalization (the sum of weights is 0.25)
        }
        for (int y=3; y<h-2; y+=2) for (int x=2; x<w-2; x+=2)
        {
            //upsampling and convolution
            for (int m=-1; m<=2; m+=2) for (int n=-2; n<=2; n+=2)
            {
                out_img[y][x] += (in_img[(y-m)/2][(x-n)/2] * filter_kernel[m+2][n+2]);
            }
            out_img[y][x] /= 0.25; //normalization (the sum of weights is 0.25)
        }
        for (int y=2; y<h-2; y+=2) for (int x=3; x<w-2; x+=2)
        {
            //upsampling and convolution
            for (int m=-2; m<=2; m+=2) for (int n=-1; n<=2; n+=2)
            {
                out_img[y][x] += (in_img[(y-m)/2][(x-n)/2] * filter_kernel[m+2][n+2]);
            }
            out_img[y][x] /= 0.25; //normalization (the sum of weights is 0.25)
        }
        for (int y=3; y<h-2; y+=2) for (int x=3; x<w-2; x+=2)
        {
            //upsampling and convolution
            for (int m=-1; m<=2; m+=2) for (int n=-1; n<=2; n+=2)
            {
                out_img[y][x] += (in_img[(y-m)/2][(x-n)/2] * filter_kernel[m+2][n+2]);
            }
            out_img[y][x] /= 0.25; //normalization (the sum of weights is 0.25)
        }

        //dealing with edges
        for (int y=2; y<h-2; y+=2)
        {
            for (int m=-2; m<=2; m+=2) for (int n=-2; n<=0; n+=2)
            {
                out_img[y][0] += (in_img[(y-m)/2][(-n)/2] * filter_kernel[m+2][n+2]);
            }
            for (int m=-2; m<=2; m+=2) for (int n=0; n<=2; n+=2)
            {
                out_img[y][w-1] += (in_img[(y-m)/2][(w-1-n)/2] * filter_kernel[m+2][n+2]);
            }
            for (int m=-2; m<=2; m+=2) for (int n=-1; n<=1; n+=2)
            {
                out_img[y][1] += (in_img[(y-m)/2][(1-n)/2] * filter_kernel[m+2][n+2]);
                out_img[y][w-2] += (in_img[(y-m)/2][(w-2-n)/2] * filter_kernel[m+2][n+2]);
            }
            out_img[y][0] /= 0.225;//normalization (the sum of weights is 0.225)
            out_img[y][w-1] /= 0.225;//normalization (the sum of weights is 0.225)
            out_img[y][1] /= 0.25; 
            out_img[y][w-2] /= 0.25;
        }
        for (int y=3; y<h-2; y+=2)
        {
            for (int m=-1; m<=2; m+=2) for (int n=-2; n<=0; n+=2)
            {
                out_img[y][0] += (in_img[(y-m)/2][(-n)/2] * filter_kernel[m+2][n+2]);
            }
            for (int m=-1; m<=2; m+=2) for (int n=0; n<=2; n+=2)
            {
                out_img[y][w-1] += (in_img[(y-m)/2][(w-1-n)/2] * filter_kernel[m+2][n+2]);
            }
            for (int m=-1; m<=1; m+=2) for (int n=-1; n<=1; n+=2)
            {
                out_img[y][1] += (in_img[(y-m)/2][(1-n)/2] * filter_kernel[m+2][n+2]);
                out_img[y][w-2] += (in_img[(y-m)/2][(w-2-n)/2] * filter_kernel[m+2][n+2]);
            }
            out_img[y][0] /= 0.225;//normalization (the sum of weights is 0.225)
            out_img[y][w-1] /= 0.225;//normalization (the sum of weights is 0.225)
            out_img[y][1] /= 0.25; 
            out_img[y][w-2] /= 0.25;
        }

        for (int x=2; x<w-2; x+=2)
        {
            for (int m=-2; m<=0; m+=2) for (int n=-2; n<=2; n+=2)
            {
                out_img[0][x] += (in_img[(-m)/2][(x-n)/2] * filter_kernel[m+2][n+2]);
            }
            for (int m=0; m<=2; m+=2) for (int n=-2; n<=2; n+=2)
            {
                out_img[h-1][x] += (in_img[(h-1-m)/2][(x-n)/2] * filter_kernel[m+2][n+2]);
            }
            for (int m=-1; m<=1; m+=2) for (int n=-2; n<=2; n+=2)
            {
                out_img[1][x] += (in_img[(1-m)/2][(x-n)/2] * filter_kernel[m+2][n+2]);
                out_img[h-2][x] += (in_img[(h-2-m)/2][(x-n)/2] * filter_kernel[m+2][n+2]);
            }
            out_img[0][x] /= 0.225;//normalization (the sum of weights is 0.225)
            out_img[h-1][x] /= 0.225;//normalization (the sum of weights is 0.225)
            out_img[1][x] /= 0.25; 
            out_img[h-2][x] /= 0.25;
        }
        for (int x=3; x<w-2; x+=2)
        {
            for (int m=-2; m<=0; m+=2) for (int n=-1; n<=2; n+=2)
            {
                out_img[0][x] += (in_img[(-m)/2][(x-n)/2] * filter_kernel[m+2][n+2]);
            }
            for (int m=0; m<=2; m+=2) for (int n=-1; n<=2; n+=2)
            {
                out_img[h-1][x] += (in_img[(h-1-m)/2][(x-n)/2] * filter_kernel[m+2][n+2]);
            }
            for (int m=-1; m<=1; m+=2) for (int n=-1; n<=1; n+=2)
            {
                out_img[1][x] += (in_img[(1-m)/2][(x-n)/2] * filter_kernel[m+2][n+2]);
                out_img[h-2][x] += (in_img[(h-2-m)/2][(x-n)/2] * filter_kernel[m+2][n+2]);
            }
            out_img[0][x] /= 0.225;//normalization (the sum of weights is 0.225)
            out_img[h-1][x] /= 0.225;//normalization (the sum of weights is 0.225)
            out_img[1][x] /= 0.25; 
            out_img[h-2][x] /= 0.25;
        }

        //dealing with corners
        for (int m=-2; m<=0; m+=2) for (int n=-2; n<=0; n+=2)
        {
            out_img[0][0] += (in_img[(-m)/2][(-n)/2] * filter_kernel[m+2][n+2]);
        }
        out_img[0][0] /= 0.2025;
        for (int m=0; m<=2; m+=2) for (int n=-2; n<=0; n+=2)
        {
            out_img[h-1][0] += (in_img[(h-1-m)/2][(-n)/2] * filter_kernel[m+2][n+2]);
        }
        out_img[h-1][0] /= 0.2025;
        for (int m=-2; m<=0; m+=2) for (int n=0; n<=2; n+=2)
        {
            out_img[0][w-1] += (in_img[(-m)/2][(w-1-n)/2] * filter_kernel[m+2][n+2]);
        }
        out_img[0][w-1] /= 0.2025;
        for (int m=0; m<=2; m+=2) for (int n=0; n<=2; n+=2)
        {
            out_img[h-1][w-1] += (in_img[(h-1-m)/2][(w-1-n)/2] * filter_kernel[m+2][n+2]);
        }
        out_img[h-1][w-1] /= 0.2025;

        for (int m=-2; m<=0; m+=2) for (int n=-1; n<=1; n+=2)
        {
            out_img[0][1] += (in_img[(-m)/2][(1-n)/2] * filter_kernel[m+2][n+2]);
            out_img[0][w-2] += (in_img[(-m)/2][(w-2-n)/2] * filter_kernel[m+2][n+2]);
        }
        out_img[0][1] /= 0.225;
        out_img[0][w-2] /= 0.225;
        for (int m=0; m<=2; m+=2) for (int n=-1; n<=1; n+=2)
        {
            out_img[h-1][1] += (in_img[(h-1-m)/2][(1-n)/2] * filter_kernel[m+2][n+2]);
            out_img[h-1][w-2] += (in_img[(h-1-m)/2][(w-2-n)/2] * filter_kernel[m+2][n+2]);
        }
        out_img[h-1][1] /= 0.225;
        out_img[h-1][w-2] /= 0.225;

        for (int m=-1; m<=1; m+=2) for (int n=-2; n<=0; n+=2)
        {
            out_img[1][0] += (in_img[(1-m)/2][(-n)/2] * filter_kernel[m+2][n+2]);
            out_img[h-2][0] += (in_img[(h-2-m)/2][(-n)/2] * filter_kernel[m+2][n+2]);
        }
        out_img[1][0] /= 0.225;
        out_img[h-2][0] /= 0.225;
        for (int m=-1; m<=1; m+=2) for (int n=0; n<=2; n+=2)
        {
            out_img[1][w-1] += (in_img[(1-m)/2][(w-1-n)/2] * filter_kernel[m+2][n+2]);
            out_img[h-2][w-1] += (in_img[(h-2-m)/2][(w-1-n)/2] * filter_kernel[m+2][n+2]);
        }
        out_img[1][w-1] /= 0.225;
        out_img[h-2][w-1] /= 0.225;

        for (int m=-1; m<=1; m+=2) for (int n=-1; n<=1; n+=2)
        {
            out_img[1][1] += (in_img[(1-m)/2][(1-n)/2] * filter_kernel[m+2][n+2]);
            out_img[h-2][1] += (in_img[(h-2-m)/2][(1-n)/2] * filter_kernel[m+2][n+2]);
            out_img[1][w-2] += (in_img[(1-m)/2][(w-2-n)/2] * filter_kernel[m+2][n+2]);
            out_img[h-2][w-2] += (in_img[(h-2-m)/2][(w-2-n)/2] * filter_kernel[m+2][n+2]);
        }
        out_img[1][1] /= 0.25;
        out_img[h-2][1] /= 0.25;
        out_img[1][w-2] /= 0.25;
        out_img[h-2][w-2] /= 0.25;

        //imshow(out_img,h,w);
}

template<typename T1, typename T2>
inline void bao_gauss_convolution(T1** imgOut, T2** imgIn, int h, int w, BAO_FLOAT sigma, int radius)
{
    BAO_FLOAT* gFilter = bao_alloc<BAO_FLOAT>(radius*2+1);
    BAO_FLOAT** tempImg = bao_alloc<BAO_FLOAT>(h,w);
    BAO_FLOAT val;
    BAO_FLOAT sum = 0;
    sigma = sigma*sigma*2;
    for(int i=-radius; i<=radius; i++)
    {
        gFilter[i+radius] = exp(-(double)(i*i)/sigma);
        sum += gFilter[i+radius];
    }
    for(int i=0; i<2*radius+1; i++) gFilter[i]/=sum;

    //horizontal filtering
    for(int y=0; y<h; y++) for (int x=0; x<w; x++)
    {
        val = 0.f;
        for (int dx=-radius; dx<=radius; dx++)
        {
            int cx = __max(0,__min(w-1,x+dx));
            val += imgIn[y][cx] * gFilter[radius+dx];
        }
        tempImg[y][x] = val;
    }

    //vertical filtering
    for(int y=0; y<h; y++) for (int x=0; x<w; x++) 
    {
        val = 0.f;
        for (int dy=-radius; dy<=radius; dy++)
        {
            int cy = __max(0,__min(h-1,y+dy));
            val += tempImg[cy][x] * gFilter[radius+dy];
        }
        imgOut[y][x] = val;
    }
    bao_free(gFilter);
    bao_free(tempImg);
}

template<typename T1, typename T2>
inline void bao_gauss_convolution(T1*** imgOut, T2*** imgIn, int h, int w, int c, BAO_FLOAT sigma, int radius)
{
    BAO_FLOAT* gFilter = bao_alloc<BAO_FLOAT>(radius*2+1);
    BAO_FLOAT*** tempImg = bao_alloc<BAO_FLOAT>(h,w,c);
    BAO_FLOAT val;
    BAO_FLOAT sum = 0;
    sigma = sigma*sigma*2;
    for(int i=-radius; i<=radius; i++)
    {
        gFilter[i+radius] = exp(-(double)(i*i)/sigma);
        sum += gFilter[i+radius];
    }
    for(int i=0; i<2*radius+1; i++) gFilter[i]/=sum;

    //horizontal filtering
    for(int y=0; y<h; y++) for (int x=0; x<w; x++) for (int k=0; k<c; k++)
    {
        val = 0.f;
        for (int dx=-radius; dx<=radius; dx++)
        {
            int cx = __max(0,__min(w-1,x+dx));
            val += imgIn[y][cx][k] * gFilter[radius+dx];
        }
        tempImg[y][x][k] = val;
    }

    //vertical filtering
    for(int y=0; y<h; y++) for (int x=0; x<w; x++) for (int k=0; k<c; k++)
    {
        val = 0.f;
        for (int dy=-radius; dy<=radius; dy++)
        {
            int cy = __max(0,__min(h-1,y+dy));
            val += tempImg[cy][x][k] * gFilter[radius+dy];
        }
        imgOut[y][x][k] = val;
    }
    bao_free(gFilter);
    bao_free(tempImg);
}

template<typename T1, typename T2>
inline void bao_gauss_downsample(T1** imgOut, T2** imgIn, int outH, int outW, int h, int w, BAO_FLOAT ratio, BAO_FLOAT sigma, int radius)
{
    BAO_FLOAT* gFilter = bao_alloc<BAO_FLOAT>(radius*2+1);
    BAO_FLOAT** tempImg = bao_alloc<BAO_FLOAT>(h,w);
    BAO_FLOAT** tempImg2 = bao_alloc<BAO_FLOAT>(h,w);
    BAO_FLOAT val;
    BAO_FLOAT sum = 0;
    sigma = sigma*sigma*2;
    for(int i=-radius; i<=radius; i++)
    {
        gFilter[i+radius] = exp(-(double)(i*i)/sigma);
        sum += gFilter[i+radius];
    }
    for(int i=0; i<2*radius+1; i++) gFilter[i]/=sum;

    //horizontal filtering
    for(int y=0; y<h; y++) for (int x=0; x<w; x++)
    {
        val = 0.f;
        for (int dx=-radius; dx<=radius; dx++)
        {
            int cx = __max(0,__min(w-1,x+dx));
            val += imgIn[y][cx] * gFilter[radius+dx];
        }
        tempImg[y][x] = val;
    }

    //vertical filtering
    for(int y=0; y<h; y++) for (int x=0; x<w; x++) 
    {
        val = 0.f;
        for (int dy=-radius; dy<=radius; dy++)
        {
            int cy = __max(0,__min(h-1,y+dy));
            val += tempImg[cy][x] * gFilter[radius+dy];
        }
        tempImg2[y][x] = val;
    }

    //resize
    bao_bilinear_resize(imgOut,tempImg2,outH,outW,h,w);
    
    bao_free(gFilter);
    bao_free(tempImg);
    bao_free(tempImg2);
}

template<typename T1, typename T2>
inline void bao_gauss_downsample(T1*** imgOut, T2*** imgIn, int outH, int outW, int h, int w, int c, BAO_FLOAT ratio, BAO_FLOAT sigma, int radius)
{
    BAO_FLOAT* gFilter = bao_alloc<BAO_FLOAT>(radius*2+1);
    BAO_FLOAT*** tempImg = bao_alloc<BAO_FLOAT>(h,w,c);
    BAO_FLOAT*** tempImg2 = bao_alloc<BAO_FLOAT>(h,w,c);
    BAO_FLOAT val;
    BAO_FLOAT sum = 0;
    sigma = sigma*sigma*2;
    for(int i=-radius; i<=radius; i++)
    {
        gFilter[i+radius] = exp(-(BAO_FLOAT)(i*i)/sigma);
        sum += gFilter[i+radius];
    }
    for(int i=0; i<2*radius+1; i++) gFilter[i]/=sum;

    //horizontal filtering
    for(int y=0; y<h; y++) for (int x=0; x<w; x++) for (int k=0; k<c; k++)
    {
        val = 0.f;
        for (int dx=-radius; dx<=radius; dx++)
        {
            int cx = __max(0,__min(w-1,x+dx));
            val += imgIn[y][cx][k] * gFilter[radius+dx];
        }
        tempImg[y][x][k] = val;
    }

    //vertical filtering
    for(int y=0; y<h; y++) for (int x=0; x<w; x++) for (int k=0; k<c; k++)
    {
        val = 0.f;
        for (int dy=-radius; dy<=radius; dy++)
        {
            int cy = __max(0,__min(h-1,y+dy));
            val += tempImg[cy][x][k] * gFilter[radius+dy];
        }
        tempImg2[y][x][k] = val;
    }

    //resize
    //if (h==156 || w == 156)imshow(tempImg2,h,w);
    bao_bilinear_resize(imgOut,tempImg2,outH,outW,h,w,c);
    //bao_bilinear_resize(imgOut,tempImg2,outH,outW,c,ratio);
    //if (h==156 || w == 156)imshow(imgOut,outH,outW);

    bao_free(gFilter);
    bao_free(tempImg);
    bao_free(tempImg2);
}

template<typename T1, typename T2>
inline void bao_construct_gauss_pyramid(T1*** pPyr, T2** pImg, int nLevels, int* arrH, int* arrW, BAO_FLOAT ratio=0.5f)
{
    if (nLevels <= 0) return;
    bao_copy(pPyr[0],pImg,arrH[0],arrW[0]);
    BAO_FLOAT sigma=(1/ratio-1);
    for (int i=1; i<nLevels; i++)
    {
        bao_gauss_downsample(pPyr[i],pPyr[i-1],arrH[i],arrW[i],arrH[i-1],arrW[i-1],ratio,sigma,sigma*3);
        //_bao_downsample_5tap(pPyr[i],pPyr[i-1],arrH[i],arrW[i]); //only for .5f ratio
    }
}

template<typename T1, typename T2>
inline void bao_construct_gauss_pyramid(T1**** pPyr, T2*** pImg, int nLevels, int* arrH, int* arrW, int nChannels, BAO_FLOAT ratio=0.5f)
{
    if (nLevels <= 0) return;
    bao_copy(pPyr[0],pImg,arrH[0],arrW[0]);
    BAO_FLOAT baseSigma=(1/ratio-1);
    int n=log(0.25)/log(ratio);
    BAO_FLOAT nSigma=baseSigma*n;
    for (int i=1; i<nLevels; i++)
    {
        if(i<=n)
        {
            BAO_FLOAT sigma=baseSigma*i;
            bao_gauss_downsample(pPyr[i],pPyr[0],arrH[i],arrW[i],arrH[0],arrW[0],nChannels,pow(ratio,i),sigma,sigma*3);
        }
        else
        {
            bao_gauss_downsample(pPyr[i],pPyr[i-n],arrH[i],arrW[i],arrH[i-n],arrW[i-n],nChannels,(BAO_FLOAT)pow(ratio,i)*arrW[0]/arrW[i-n],nSigma,nSigma*3);
        }
    }
}

template<typename T>
inline void bao_sort(T* pArray, int len)
{
    for (int i=0; i<len; i++)
    {
        for (int j=i+1; j<len; j++)
        {
            if (pArray[i]>pArray[j])
            {
                T temp = pArray[j];
                pArray[j] = pArray[i];
                pArray[i] = temp;
            }
        }
    }
}

template<typename T1, typename T2>
inline void bao_median_filter(T1**img_out, T2**img_in, int h, int w, int radius)
{
    int numPix = 2*radius+1;
    numPix *= numPix;
    T2* pTempBuf = bao_alloc<T2>(numPix);
    for (int y=0; y<h; y++) for (int x=0; x<w; x++)
    {
        memset(pTempBuf,0,sizeof(T2)*numPix);
        int nCount = 0;
        for (int dy=-radius; dy<=radius; dy++) for (int dx=-radius; dx<=radius; dx++)
        {
            int cx = x+dx;
            int cy = y+dy;
            if (cx<0 || cx>w-1 || cy<0 || cy>h-1) continue;
            pTempBuf[nCount++] = img_in[cy][cx];
        }
        bao_sort(pTempBuf,nCount);
        img_out[y][x] = pTempBuf[nCount/2];
    }
    bao_free(pTempBuf);
}


template<typename T1, typename T2>
bool bao_image_check_equal(T1** img1, T2** img2, int h, int w)
{
    if (img1 == img2) return true;
    for (int y=0; y<h; y++) for (int x=0; x<w; x++)
    {
        if (abs(img1[y][x]-img2[y][x]) > BAO_ZERO) return false;
    }
    return true;
}

template<typename T1, typename T2>
bool bao_image_check_equal(T1*** img1, T2*** img2, int h, int w, int c=3)
{
    if (img1 == img2) return true;
    for (int y=0; y<h; y++) for (int x=0; x<w; x++) for (int k=0; k<c; k++)
    {
        if (abs(img1[y][x][k]-img2[y][x][k]) > BAO_ZERO) return false;
    }
    return true;
}

template<typename T1, typename T2>
bool bao_image_check_equal_relax(T1*** img1, T2*** img2, int h, int w, int c=3) //returns true if no more than 1%¡£
{
    if (img1 == img2) return true;
    long countDiff = 0;
    for (int y=0; y<h; y++) for (int x=0; x<w; x++) for (int k=0; k<c; k++)
    {
        if (abs(img1[y][x][k]-img2[y][x][k]) > 20) countDiff++;
    }
    if (countDiff > h*w/1000) return false; //more than 1%
    else return true;
}

int bao_loadimage_ppm(char* filename,unsigned char *image,int h,int w,int *nr_channel);

#endif