# Vector Addition

## Objective

The purpose of this lab is for you to become familiar with using the CUDA API by implementing a simple vector addition kernel and its associated host code as shown in the lectures.

### Retrieving Assignments

To retrieve (or update) released assignments, go to your ece408git folder and run the following:

* `git fetch release`
* `git merge release/main -m "some comment" --allow-unrelated-histories`
* `git push origin main`

where "some comment" is a comment for your submission. The last command pushes the newly merged files to your remote repository. If something ever happens to your repository and you need to go back in time, you will be able to revert your repository to when you first retrieved an assignment.

## Instructions

Edit the code in `lab1.cu` to perform the following:

* Allocate device memory
* Copy host memory to device
* Initialize thread block and kernel grid dimensions
* Invoke CUDA kernel
* Copy results from device to host
* Free device memory
* Implement the CUDA kernel

Instructions about where to place each part of the code is demarcated by the //@@ comment lines.

## Running the code using slurm

For basic usage of slurm to compile and run the code, refer to the instructions in [lab0](https://github.com/illinois-cs-coursework/fa24_ece408_.release/tree/main/lab0#to-compile-and-execute-your-program). Here we give a slightly more detailed introduction to slurm, including an explanation for the job.slurm file and other useful commands.

Delta cluster uses [slurm](https://slurm.schedmd.com/documentation.html) to manage GPU/CPU resources. Slurm is an open-source and scalable system for cluster management and job scheduling, which is widely used in high-performance clusters all over the world.

### Slurm job script

Slurm allows users to get GPU/CPU resources in two ways: 1) submitting batch jobs to clusters and 2) getting interactive sessions through the srun command. For labs in this course, we only ask students to submit batch jobs through sbatch, as in lab0. Here we give a more detailed explanation to the batch job script, job.slurm.

Here is a copy of job.slrum for lab1.

```
#!/bin/bash
#SBATCH --job-name="Lab1"
#SBATCH --output="lab1.out"
#SBATCH --error="lab1.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:01:00

module reset
module load cuda

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

srun ./lab1 -e data/0/output.raw -i data/0/input0.raw,data/0/input1.raw -o /tmp/myoutput.raw -t vector >> lab1.out
srun ./lab1 -e data/1/output.raw -i data/1/input0.raw,data/1/input1.raw -o /tmp/myoutput.raw -t vector >> lab1.out
srun ./lab1 -e data/2/output.raw -i data/2/input0.raw,data/2/input1.raw -o /tmp/myoutput.raw -t vector >> lab1.out
srun ./lab1 -e data/3/output.raw -i data/3/input0.raw,data/3/input1.raw -o /tmp/myoutput.raw -t vector >> lab1.out
srun ./lab1 -e data/4/output.raw -i data/4/input0.raw,data/4/input1.raw -o /tmp/myoutput.raw -t vector >> lab1.out
srun ./lab1 -e data/5/output.raw -i data/5/input0.raw,data/5/input1.raw -o /tmp/myoutput.raw -t vector >> lab1.out
srun ./lab1 -e data/6/output.raw -i data/6/input0.raw,data/6/input1.raw -o /tmp/myoutput.raw -t vector >> lab1.out
srun ./lab1 -e data/7/output.raw -i data/7/input0.raw,data/7/input1.raw -o /tmp/myoutput.raw -t vector >> lab1.out
srun ./lab1 -e data/8/output.raw -i data/8/input0.raw,data/8/input1.raw -o /tmp/myoutput.raw -t vector >> lab1.out
srun ./lab1 -e data/9/output.raw -i data/9/input0.raw,data/9/input1.raw -o /tmp/myoutput.raw -t vector >> lab1.out
```

The first line `#!/bin/bash` simply means this is a script. The following lines start with `#SBATCH ` indicate the parameters for slurm in this lab. The remaining lines are commands that get run on the GPU node.

#### Slurm parameters

The lines start with `#SBATCH ` contain the slurm parameters. `job-name` is a human-readable name for your job. `output` and `error` are the path to the stdout and stderr files. You can change these parameters freely. For example, if you want to save all your output and error files in a folder. You can use `mkdir slurm_logs` and change the script to:

```
#SBATCH --output="./slurm_logs/lab1.%j.out"
#SBATCH --error="./slurm_logs/lab1.%j.err"
```

Here %j means the jobid for each job. In that way, all your history output and error files will be stored in the folder `slurm_logs` without being automatically overwritten.

Other parameters indicate the resources we apply for on Delta. In lab 1, we use one CPU and one A40 GPU on one node using only one task. We are using credit in bche-delta-gpu account, and the maximum time for the job is 1 minute. Please do not change these parameters.

#### Commands executed on delta

The remaining lines are commands that will be executed on the GPU node. The lines starting with `module` will handle the environment on the cluster, and the `echo` line prints the string to your output file. The rest of the lines are where we execute your complied `lab1` with every test data to test whether your code is correct.

#### Other useful slurm commands

```
squeue -u <your net id>
```

This command shows the status of your job. Occasionally, when the cluster is very busy, your job will potentially get stuck in the queue. You can use this command to check your job.

In the output of `squeue`, if the row`NODELIST(REASON)` shows `Priority` or `Resource`, it means the cluster is busy with other jobs and you may need to wait for a few minutes.

In addition, the simple command `squeue` without your netid shows the entire job list in the queue, including all the users in Delta. We do not recommend doing that since this list is usually too long and can waste computational resources on the login nodes. Please use this with caution.



```
scancel  <your job id>
```

The `squeue -u [your net id]` command will print your job id under the `JOBID` row. Using this ID, you can simply cancel your job if you find you need to modify your code or for any other reasons.



## Lab Submission

Every time you want to save the work, you will need to add, commit, and push your work to your git repository. This can always be done using the following commands on a command line while within your ECE 408 directory:

* ```git add -u```
* ```git commit -m "REPLACE THIS WITH YOUR COMMIT MESSAGE"```
* ```git push origin main```
