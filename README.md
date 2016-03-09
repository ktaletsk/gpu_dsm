Discrete Slip-Link Model (DSM) is mathematical model that describes the dynamics of flexible entangled polymer melts.

GPU DSM is computational implementation of that model on CUDA/C++. GPU DSM is developed in [Center for molecular study of condensed soft matter (μCoSM)](http://www.chbe.iit.edu/~schieber/dsm-software.html). GPU DSM is free open-source software under GNU GPL v3.0 license.

[Download latest Linux GUI version](https://github.com/ktaletsk/gpu_dsm/releases)

Compilation:

1) Windows. (tested on Windows 7,10)
Requirements:
Microsoft Visual Studio (2010, 2012, 2013)
Cuda Toolkit (6.0, 6.5, 7.0)
optional: make (available in cygwin and minGW)

1. Run microsoft visual studio developer command prompt
(it can be found in start menu->all programs->microsoft visual studio-> visual studio tools)
    
2. To comile command line interface (CLI) version of code navigate to source directory 
(example cd C:\gpu_dsm\CLI)
    
3. Run make.bat
you installed make, you can run make
it should create gpu_DSM.exe 
    
4. You can test it by running gpu_DSM.exe
        
2) Linux (tested on Ubuntu/Kubuntu 14.04/15.04)
Requirements:
g++
Cuda Toolkit (6.0, 6.5, 7.0)
Qt 5.4 (for GUI)
optional: make
    
1. Run terminal
    
2a. To comile command line interface (CLI) version navigate to source directory
(example cd C:\gpu_dsm\CLI)
    
2b. To comile graphical user interface (GUI) version navigate to source directory
(example cd C:\gpu_dsm\GUI)

3a. Run "make all"
it creates gpu_DSM

3b. Run "<path_to_Qt>/5.4/gcc_64/bin/qmake -spec linux-g++ -o Makefile dsm.pro"
    
4a. You can test it by running ./gpu_DSM
    
4b. You can test it by running ./dsm or clicking to app icon in file manager

    
Running (CLI):
    
gpu_DSM command line parameters:

first parameter is seed/job_ID
example: 
./gpu_DSM 1
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
