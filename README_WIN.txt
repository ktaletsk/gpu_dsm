list of changes to make it work on windows
1)cudautil.cu 
first line #include <windows.h>
2)binomial.cpp 
   replace lgamma with gammln
   note: looks like MSVS2013 have lgamma, so hopefully this fix is not required there
3)ensemble_kernel.cu gpu_random_cu uint undefined
added #define uint unsigned int in ensemble_kernel.cu gpu_random_cu
4)timer.h replace with timer_win.h
5)job_ID.h log10
 replace log10(*) with log10(double(*))