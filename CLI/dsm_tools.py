import uuid
import paramiko
import os

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

def job_start(ssh, job_folder):
    ssh.connect(server, username=username, password=password)

    #Create temporary folder for job, clone and compile the code
    command1 = 'cd temp_jobs; mkdir {name}; cd {name}; git clone https://github.com/ktaletsk/gpu_dsm.git .; git fetch; git checkout {branch}; cd CLI; make all clean'.format(name=job_folder, branch="star_branched")
    (stdin, stdout, stderr) = ssh.exec_command(command1)
    for line in stdout.readlines():
        print line

    #Transfer input file
    sftp=ssh.open_sftp()
    source= 'input.dat'
    destination ='temp_jobs/{name}/CLI/input.dat'.format(name=job_folder)
    sftp.put(source,destination)
    sftp.close()

    jobIds = []
    #Submit job to qsub
    command2 = './param_sub.sh 2 temp_jobs/{name}/CLI {name}'.format(name=job_folder)
    (stdin, stdout, stderr) = ssh.exec_command(command2)
    for line in stdout.readlines():
        jobIds.append(line)
        print line

    ssh.close()
    return jobIds

def jobs_done(ssh, jobIds):
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
