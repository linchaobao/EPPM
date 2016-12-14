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

#ifndef _DEF_PARAMETERS_H_
#define _DEF_PARAMETERS_H_


#define PYR_MAX_DEPTH    3
#define PYR_MIN_WIDTH    20
#define PYR_RATIO        0.5f
#define PYR_RATIO_UP     (1.f/PYR_RATIO)

#define  SEARCH_RANGE      30
#define  SEARCH_RADIUS_MIN 1
#define  NUM_RAND_GUESS    6

#define  PM_SCALE_RANGE    9
#define  PM_SCALE_MIN      4

//parameters for patchmatch
#define PATCH_R   9 
#define NUM_ITER  10 

#define PM_SIG_S  (0.5f*PATCH_R) 
#define PM_SIG_R  0.1f 


#define LAMBDA_AD             0.1f  
#define LAMBDA_CENSUS         0.3f  

#define MAX_FLOW_VAL          200  


//parameters for refinement
#define WMF_RADIUS  4 
#define WMF_SIG_S   (WMF_RADIUS*1.0f)
#define WMF_SIG_R   0.02f 


//post blf
#define POSTPROC_BLF_SIG_S       5 


//outlier removal
#define  STAT_RADIUS  6 


//subpixel
#define SUBPIX_UP_FACTOR        2.f
#define SUBPIX_UP_FACTOR_INV    (1.f/SUBPIX_UP_FACTOR)
#define SUBPIX_PATCH_R          9   //30
#define SUBPIX_SIG_S            (1.0f*SUBPIX_PATCH_R) //30.0
#define SUBPIX_SIG_R            0.2f  //0.1



////////////////////////////////////////////////////////////////////////////////////
//invalid flow
// the "official" threshold - if the absolute value of either 
// flow component is greater, it's considered unknown
#ifndef UNKNOWN_FLOW_THRESH
#define UNKNOWN_FLOW_THRESH 1e9
#endif

// value to use to represent unknown flow
#ifndef UNKNOWN_FLOW
#define UNKNOWN_FLOW 1e10
#endif


#endif