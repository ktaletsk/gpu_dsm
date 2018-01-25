import uuid
import paramiko
import os
from fdt_fit import fdt_fit, fdt, fdtvec
import numpy as np
import matplotlib.pyplot as plt

server = 'example.com'
username = 'user'
password = 'password'

def ssh_init():
    #Load server and login information from config file '.clustrc'
    global server, username, password
    with open('.clustrc') as f:
        server = f.readline().rstrip()
        username = f.readline().rstrip()
        password = f.readline().rstrip()

    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()

    return ssh

def job_start(ssh, job_folder, njobs):
    ssh.connect(server, username=username, password=password)

    #Create temporary folder for job, clone and compile the code
    command1 = 'cd temp_jobs; mkdir {name}; cd {name}; git clone https://github.com/ktaletsk/gpu_dsm.git .; git fetch; git checkout {branch}; cd CLI; make all clean'.format(name=job_folder, branch="star_branched")
    (stdin, stdout, stderr) = ssh.exec_command(command1)
    for line in stdout.readlines():
        print line

    #Transfer input file
    source= '{name}/input.dat'.format(name=job_folder)
    if os.path.isfile(source):
        sftp=ssh.open_sftp()
        destination ='temp_jobs/{name}/CLI/input.dat'.format(name=job_folder)
        sftp.put(source,destination)
        sftp.close()

    #Transfer p^cr input table
    source= '{name}/pcd_MMM.dat'.format(name=job_folder)
    if os.path.isfile(source):
        sftp=ssh.open_sftp()
        destination ='temp_jobs/{name}/CLI/pcd_MMM.dat'.format(name=job_folder)
        sftp.put(source,destination)
        sftp.close()

    jobIds = []
    #Submit job to qsub
    command2 = './param_sub.sh {nnodes} temp_jobs/{name}/CLI {name}'.format(nnodes=njobs,name=job_folder)
    (stdin, stdout, stderr) = ssh.exec_command(command2)
    for line in stdout.readlines():
        jobIds.append(line)
        print line

    ssh.close()
    return jobIds

def job_done(ssh, jobIds):
    jobsComplete = []
    ssh.connect(server, username=username, password=password)
    for job in jobIds:
        (stdin, stdout, stderr) = ssh.exec_command('qstat ' + job.rstrip())
        lines = []
        for line in stdout.readlines():
            lines.append(line)

        if not lines: #qsub returned nothing
            (stdin, stdout, stderr) = ssh.exec_command('tracejob ' + job.rstrip())
            lines = []
            for line in stdout.readlines():
                lines.append(line)
            for line in lines:
                if "COMPLETE" in line:
                    jobsComplete.append(True)
        else:
            jobsComplete.append(False)
    ssh.close()

    return all(jobsComplete)

def job_transfer_clean(ssh, job_folder):
    if not os.path.exists(job_folder):
        os.makedirs(job_folder)
    ssh.connect(server, username=username, password=password)
    sftp=ssh.open_sftp()
    remote_dir = 'temp_jobs/{name}/CLI/'.format(name=job_folder)
    dir_items = sftp.listdir_attr(remote_dir)
    for item in dir_items:
        source = remote_dir + '/' + item.filename
        destination = os.path.join(job_folder, item.filename)
        sftp.get(source, destination)
    sftp.close()
    (stdin, stdout, stderr) = ssh.exec_command('rm -rf temp_jobs/{name}; rm {name}.*'.format(name=job_folder))
    for line in stdout.readlines():
        print line
    ssh.close()

class Calculation(object):
    #Input parameters
    beta = 1
    nChains = None
    velocityGradient = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    CDtoggle = None
    PDtoggle = 0
    calcMode = 1
    timeStep = 1
    simTime = None
    token = None #folder name
    ssh = None #storing Paramiko SSH object
    jobIds = None #array of job IDs on cluster
    jobStatus = 0
    num_gpu = None
    fdt_x = None
    fdt_y = None
    fdt_result_x = None
    fdt_result_y = None
    lambdaArr = None
    gArr = None
    pcd_cr_input_x = None
    pcd_cr_input_y = None

    #Function to generate input file
    def generate_token(self):
        self.token = str(uuid.uuid4())
        return

    def set_pcd_cr_input(self, pcd_cr_input_x, pcd_cr_input_y):
        self.pcd_cr_input_x = pcd_cr_input_x
        self.pcd_cr_input_y = pcd_cr_input_y
        pcd_input=zip(pcd_cr_input_x, pcd_cr_input_y)

        file = open(self.token + "/pcd_MMM.dat","w")
        file.write(str(np.size(self.pcd_cr_input_x)))
        for i in pcd_input:
            file.write('\n'+str(i[0])+'\t'+str(i[1]))
        file.close()

        return

    #Function to start calculation on cluster
    def calc(self):
        self.ssh=ssh_init()
        self.jobIds = job_start(self.ssh, self.token, self.num_gpu)
        self.jobStatus = 1
        return

    #Function to check the status of calculation
    def check_self(self):
        if self.jobStatus==0:
            print('Calculation is not started yet')
        if self.jobStatus==1:
            if not self.ssh:
                self.ssh=ssh_init()
            if job_done(self.ssh, self.jobIds):
                self.jobStatus=2
                print('Calculation is done. Transfering results')
                job_transfer_clean(self.ssh, self.token)

                #Read f_d(t) simulation results and fit
                with open(self.token + '/fdt_aver.dat') as f:
                    lines = f.readlines()
                    self.fdt_x = np.array([float(line.split()[0]) for line in lines])
                    ydata = np.array([float(line.split()[1]) for line in lines])

                self.fdt_y = 1.0-np.cumsum(ydata)/np.sum(ydata)

                #Read result_f_d(t)
                with open(self.token + '/fdt_result.dat') as f:
                    lines = f.readlines()
                    self.fdt_result_x = np.array([float(line.split()[0]) for line in lines])
                    self.fdt_result_y = np.array([float(line.split()[1]) for line in lines])

                with open(self.token + '/fdt_MMM_fit.dat') as f:
                    lines = f.readlines()
                self.lambdaArr = np.array([float(line.split()[0]) for line in lines[1:]])
                self.gArr = np.array([float(line.split()[1]) for line in lines[1:]])
            else:
                print('Still running, check back later')
        else:
            print('Calculation is finished')

    #Plot fit results
    def plot_fit_results(self):
        fig = plt.figure(figsize=(24, 6))

        ax1 = fig.add_subplot(131)

        ax1.set_title("Entanglement lifetime distribution")
        ax1.set_xlabel(r'$t/\tau_c$')
        ax1.set_ylabel(r'$f_d(t)$')

        ax1.scatter(self.fdt_result_x, self.fdt_result_y, c='r', label=r'$f_d(t)$')
        ax1.plot(self.fdt_result_x, fdtvec(time=self.fdt_result_x, params=np.append(self.lambdaArr, self.gArr)), c='b', label=r'MMM fit')
        leg = ax1.legend()
        ax1.set_xscale('log')
        ax1.set_yscale('log')

        ax2 = fig.add_subplot(132)

        ax2.set_title(r'$p^{cr}\left(\tau\right)$')
        ax2.set_xlabel(r'$\lambda$')
        ax2.set_ylabel(r'$g$')

        ax2.scatter(self.lambdaArr, self.gArr, c='b')

        leg = ax2.legend()
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        plt.xlim(xmin=0.5*min(self.lambdaArr), xmax=2*max(self.lambdaArr))
        plt.ylim(ymin=0.5*min(self.gArr), ymax=2*max(self.gArr))

        ax3 = fig.add_subplot(133)

        ax3.set_title(r'$p^{eq}\left(\tau\right)$')
        ax3.set_xlabel(r'$\lambda$')
        ax3.set_ylabel(r'$g$')

        ax3.scatter(self.lambdaArr, np.multiply(self.lambdaArr, self.gArr)/np.dot(self.lambdaArr, self.gArr), c='b')

        leg = ax3.legend()
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        plt.xlim(xmin=0.5*min(self.lambdaArr), xmax=2*max(self.lambdaArr))
        plt.ylim(ymin=0.5*min(np.multiply(self.lambdaArr, self.gArr)/np.dot(self.lambdaArr, self.gArr)), ymax=2*max(np.multiply(self.lambdaArr, self.gArr)/np.dot(self.lambdaArr, self.gArr)))

        plt.show()

    #Initialize object of class
    def __init__(self, CDtoggle, nChains, num_gpu, simTime):
        self.generate_token()
        if not os.path.exists(self.token):
            os.makedirs(self.token)
        self.CDtoggle = CDtoggle
        self.nChains = nChains
        self.num_gpu = num_gpu
        self.simTime = simTime
        self.generate_input()

class CalculationStar(Calculation):
    architecture = 'star' #Type of chain
    nArms = None #Example: nArms = 3
    nkArms = None #Example: nkArms = [6 6 6]

    def __init__(self, nArms, nkArms, CDtoggle, nChains, num_gpu, simTime):
        self.nArms = nArms
        self.nkArms = nkArms
        super(CalculationStar, self).__init__(CDtoggle, nChains, num_gpu, simTime)

    def generate_input(self):
        file = open(self.token + "/input.dat","w")

        file.write(str(self.beta)+'\n')
        file.write(self.architecture+'\n')
        file.write(str(self.nArms)+'\n')
        file.write(" ".join(map(str,self.nkArms)) + '\n')
        file.write(str(self.nChains)+'\n')
        file.write(" ".join(map(str,self.velocityGradient)) + '\n')
        file.write(str(self.CDtoggle)+'\n')
        file.write(str(self.PDtoggle)+'\n')
        file.write(str(self.calcMode)+'\n')
        file.write(str(self.timeStep)+'\n')
        file.write(str(self.simTime))

        file.close()

class CalculationLinear(Calculation):
    architecture = 'linear' #Type of chain
    nk = None #Example: nk = 20

    def __init__(self, nk, CDtoggle, nChains, num_gpu, simTime):
        self.nk = nk
        super(CalculationLinear, self).__init__(CDtoggle, nChains, num_gpu, simTime)

    def generate_input(self):
        file = open(self.token + "/input.dat","w")

        file.write(str(self.beta)+'\n')
        file.write(self.architecture+'\n')
        file.write(str(self.nk)+'\n')
        file.write(str(self.nChains)+'\n')
        file.write(" ".join(map(str,self.velocityGradient)) + '\n')
        file.write(str(self.CDtoggle)+'\n')
        file.write(str(self.PDtoggle)+'\n')
        file.write(str(self.calcMode)+'\n')
        file.write(str(self.timeStep)+'\n')
        file.write(str(self.simTime))

        file.close()
