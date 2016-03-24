###**The Discrete Slip-Link Model (DSM)** is a mathematical model that describes the dynamics of flexible entangled polymer melts.

**GPU DSM** is a computational implementation of that model on CUDA/C++. GPU DSM is developed in [The Center for molecular study of condensed soft matter (μCoSM)](http://www.chbe.iit.edu/~schieber/index.html). GPU DSM is free open-source software under the GNU GPL v3.0 license.

**[Download latest Linux GUI version](https://github.com/ktaletsk/gpu_dsm/releases)**

####Compilation instructions:

####Linux (tested on Ubuntu/Kubuntu 14.04)

Requirements:

g++

**[Cuda Toolkit](https://developer.nvidia.com/cuda-toolkit)** (6.0, 6.5, 7.0, 7.5)

**[Qt](http://www.qt.io/download-open-source/)** (for GUI)

optional: make
    
1. Open terminal.
    
2a. Compile the command line interface (CLI) version, navigate to the directory where you extracted the zip file
`cd <path_to_repository>/gpu_dsm/CLI`.
    
2b. Compile the graphical user interface (GUI) version, navigate to the directory where you extracted the zip file
`cd <path_to_repository>/gpu_dsm/GUI`.

3a. Run `make all` to complie **gpu_DSM**.

3b. Run `<path_to_Qt>/<version_of_Qt>/gcc_64/bin/qmake -spec linux-g++ -o Makefile dsm.pro`. Current version of Qt is 5.6.

Run `make all` to compile **dsm**.
    
4a. You can test it by running `./gpu_DSM`.
    
4b. You can test it by running `./dsm` or clicking to the app icon in a file manager.

####Running (CLI):
    
**gpu_DSM** command line parameters:

first parameter is seed/job_ID
example: 
`./gpu_DSM 1`
all the files generated in this run will have "_1" in the filename.
additionally 1 will be used as a seed number for pseudo random number generator

-s filename 
saves chain conformations in "filename" file in the end of run.

-l filename
loads previously saved chain conformations from the "filename" file in the beginning of run.

-d number
selects GPU to use. Useful if multiple GPU are present in the system. Numberring starts from 0.
    
-distr
saves final Z,N,Q distributions in .dat files
