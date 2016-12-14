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

#include <stdio.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

bao_timer_gpu::bao_timer_gpu()
{
    cudaEventCreate(&m_start);
    cudaEventCreate(&m_stop);
}

bao_timer_gpu::~bao_timer_gpu()
{
    cudaEventDestroy(m_start);
    cudaEventDestroy(m_stop);
}

void bao_timer_gpu::start()
{
    cudaEventRecord(m_start, 0);
}

double bao_timer_gpu::stop()
{
    cudaEventRecord(m_stop, 0);
    float elapsed;
    cudaEventSynchronize(m_stop);
    cudaEventElapsedTime(&elapsed, m_start, m_stop);
    return elapsed;
}

double bao_timer_gpu::time_display(char *disp,int nr_frame)
{ 
    double sec = stop()/nr_frame;
    printf("Running time (%s) is: %5.4f ms.\n",disp,sec);
    return sec;
}

double bao_timer_gpu::fps_display(char *disp,int nr_frame)
{ 
    double fps = (double)nr_frame/(stop()*1.0e-3f);
    printf("Running time (%s) is: %5.2f fps.\n",disp,fps);
    return fps;
}


void bao_timer_gpu_cpu::start()
{
    checkCudaErrors(cudaDeviceSynchronize());
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

double bao_timer_gpu_cpu::stop()
{
    checkCudaErrors(cudaDeviceSynchronize());
#ifdef _WIN32
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return double(li.QuadPart-m_counter_start)/m_pc_frequency;
#else
    struct timeval timerStop, timerElapsed;
    gettimeofday(&timerStop, NULL);
    timersub(&timerStop, &timerStart, &timerElapsed);
    return timerElapsed.tv_sec+timerElapsed.tv_usec/1000000.0;
#endif
}

double bao_timer_gpu_cpu::time_display(char *disp,int nr_frame)
{ 
    double sec = stop()/nr_frame;
    printf("Running time (%s) is: %5.5f Seconds.\n",disp,sec);
    return sec;
}

double bao_timer_gpu_cpu::fps_display(char *disp,int nr_frame)
{ 
    double fps = (double)nr_frame/stop();
    printf("Running time (%s) is: %5.5f frame per second.\n",disp,fps);
    return fps;
}

