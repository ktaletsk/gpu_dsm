    +--------------------------+
    |        GPU DSM           |
    |      readme file         |
    +--------------------------+

	Compilation:
     
    1)Windows. (tested on Windows 7)
    Requirements:
    Microsoft Visual Studio (2010, 2012, 2013)
    Cuda Toolkit (6.0, 6.5)
    optional: make (available in cygwin and minGW)
    
    1.run microsoft visual studio developer command prompt
    (it can be found in start menu->all programs->microsoft visual studio-> visual studio tools)
    
    2.navigate to source directory 
    (example cd C:\gpu_dsm_master)
    
    3.run make.bat
    you installed make, you can run make
    it should create gpu_DSM.exe 
    
    4. you can test it by running gpu_DSM.exe
        
    2)Linux. (tested on Ubuntu/Kubuntu 12.04/14.04)
    Requirements:
    g++
    Cuda Toolkit (6.0, 6.5)
    optional: make 
    
    1.run terminal
    
    2.navigate to source directory 
    (example cd Downloads/gpu_dsm_master)
    
    3.run make 
    alternatively you can run make.bat
    it should create gpu_DSM
    
    4. you can test it by running ./gpu_DSM

    
	Running:
    
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
