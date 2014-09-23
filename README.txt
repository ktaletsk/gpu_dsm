gpu_DSM command line parameters:

first parameter is seed/job_ID
example: 
./gpu_DSM 1
all the files generates in this run will have "_1" in the filename.
additionally 1 will be used as a seed number for pseudo random number generator

-s filename 
saves chain conformations in "filename" file in the end of run.

-l filename
loads previously saved chain conformations from the "filename" file in the beginning of run.

-d number
selects GPU to use. Useful if multiple GPU are present in the system. Numberring starts from 0.