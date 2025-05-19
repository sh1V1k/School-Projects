# Milestone 1: CPU Convolution, Basic GPU Convolution, Profiling


***Deadline: October 11th, 2024 8PM***

For each milestone, you will also need to complete a report on Canvas. The table below contains all of the deliverables.

| Deliverables                                             |
| -------------------------------------------------------- |
| Create a CPU convolution implementation                  |
| Profile your CPU implementation with `gprof`             |
| Implement a basic GPU Convolution kernel from Lecture 12 |
| Correctness and timing with 3 different dataset sizes    |
| Profile your GPU implementation with `nsys` and `ncu`    |
| Complete your report on Canvas                           |
| Submit your work for grading                             |

## Table of Contents

* [Project Setup](#project-setup)
* [CPU Implementation](#create-a-cpu-implementation)
* [GPU Implementation](#create-a-gpu-implementation)
* [Report](#report)
* [Submission](#submitting-milestone-1-for-grading)
* [Rubric](#rubric)

## Project Setup

1. To start, you will need to clone this repository to your folder in the Delta server. Go to your `ece408git` folder and run the following:

   * `git fetch release`
   * `git merge release/main -m "Project checkpoint 1" --allow-unrelated-histories`
   * `git push origin main`

   If you are asked to enter your git credentials (PAT) each time you try to pull or push, you can try to store your git credentials once and for all: just type git config credential.helper store in your Delta terminal.

2. We have already set up the dataset for you in the Delta server under the path `/projects/bche/project/data/fmnist-86/`. Please do not modify it!

3. To compile, simply type `./run.sh build`. This will attempt to clean unrelated files and compile your code. If there are any errors, you will see a message in the terminal. Fix the errors and try to recompile again. Please note that the compilation takes place on the cluster head node which has all the tools but does not have any GPUs. Therefore, you cannot execute your compiled code on the head node.

4. Take the CPU implementation as an example, to execute your code, type `sbatch m1_cpu.slurm`. This will schedule the execution of your program on one of the next available compute nodes. The error message during the execution will be input into `Milestone1_CPU.err`. Unlike the head node, compute nodes have GPUs and this is where we want our program to be executed. You will get a message like this `Submitted batch job ID` where the last number is your job ID. Typically, jobs will be executed within a few seconds. However, if your job is not getting executed immediately, you can check its status by typing `squeue --job ID` (do not forget to replace "ID" number with your actual job ID reported by sbatch).

5. To clean, type `./run.sh clean`. This will remove all the files generated during the compilation and execution process.

***Understanding m1_cpu.slurm***

`./m1_cpu 100` runs the code specified in `./project/src/layer/custom/cpu-new-forward.cc` program for a batch of 100 input images.

You should see the following output in m1_cpu.out file:

    Test batch size: 100
    Loading fashion-mnist data...Done
    Loading model...Done
    Conv-CPU==
    Op Time: 1451.97 ms
    Conv-CPU==
    Op Time: 4132.6 ms

    Test Accuracy: 0.86


It is okay for the accuracy to be low here since you haven't implemented the convolutional layers yet.

Modify `m1_cpu.slurm` to use `time` to measure the elapsed time of the whole program.
You can use the following command to redirect the output of your program (./m1_cpu 100) to `m1_cpu.out` and the detailed running time information from the time command to `time.out`.

    { time srun ./m1_cpu 100 > m1_cpu.out; } 2> time.out

## Create a CPU Implementation

See the [description](#skeleton-code-description) of the skeleton code for a brief overview of what each file does.

**Modify `./project/src/layer/custom/cpu-new-forward.cc` to implement the forward convolution described in Chapter 16 of the textbook.**
The performance of the CPU convolution is not part of the project evaluation. We only evaluate for correctness.

The algorithm is also below, for your convenience

    for b = 0 .. Batch                     // for each image in the batch
        for m = 0 .. Map_out               // for each output feature maps
            for h = 0 .. Height_out        // for each output element
                for w = 0 .. Width_out
                {
                    output[b][m][h][w] = 0;
                    for c = 0 .. Channel   // sum over all input feature maps
                        for p = 0 .. K // KxK filter
                            for q = 0 .. K
                                output[b][m][h][w] += input[b][c][h + p][w + q] * k[m][c][p][q]
                }

Unlike the convolutions described in the class, note that this one is not centered on the input image. There is no padding and the strides are 1. The following illustration may help you visualize this better.

![ConvExample](https://stanford.edu/~shervine/teaching/cs-230/illustrations/convolution-layer-a.png?1c517e00cb8d709baf32fc3d39ebae67)

*Source: https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks#layer*

Modify `m1_cpu.slurm` to invoke

    srun ./m1_cpu 100 > m1_cpu.out

Please be patient as the CPU implementation is slow and will take several minutes to run. (For instance, a correct implementation with 10k images may take 13+ mins to run). If you want to iterate quickly when developing code using smaller batch sizes, see [Specifying Batch Size](#specifying-batch-size). When your implementation is correct, you should see output like this:

    Test batch size: 100
    Loading fashion-mnist data...Done
    Loading model...Done
    Conv-CPU==
    Op Time: 1451.97 ms
    Conv-CPU==
    Op Time: 4132.6 ms

    Test Accuracy: 0.86


Every time your layer is invoked, it will print the "Op Time," the time spent working on that layer.
Since the network has two convolutional layers, two times will be printed.
You can time the whole program execution by modifying `m1_cpu.slurm` with

    { time srun ./m1_cpu 100 > m1_cpu.out; } 2> time.out

### Specifying Batch Size

`./m1_cpu` and `./m1_gpu` both take one optional argument: the dataset size.
If the correctness for each possible batch size is as below, you can be reasonably confident your implementation is right. The correctness does depend on the data size.

For example, to check your accuracy on the full data size of 10,000, you could modify `m1_cpu.slurm` to run

    srun ./m1_cpu 10000 > m1_cpu.out

| Number of Images | Accuracy |
| ---------------- | -------- |
| 100              | 0.86     |
| 1000             | 0.886    |
| 10000            | 0.8714   |

### Use Gprof to profile your CPU implementation

You will use `gprof` to profile the execution of your CPU forward convolution implementation.

We compile and link your `cpu-new-forward.cc` with the `-pg` flag in the file `run.sh`, which creates a `gmon.out` artifact containing profile information when the binary `m1_cpu` is executed.  To analyze this information in human readable form, modify `m1_cpu.slurm` and modify the line to redirect `gprof` output as `outfile`.

    srun ./m1_cpu 1000 && gprof -Q ./m1_cpu gmon.out > outfile

By default, `gprof` prints both a flat profile and a call graph (see "Interpreting gprof's Output" in the [GNU gprof Documentation](https://sourceware.org/binutils/docs/gprof/index.html)).  With the `-Q` flag, we only print the flat profile.  The information you need can be found near the beginning of `gprof`'s output. You can download your build folder and process the output `outfile` with `grep` (with your function's name) or `head`. You can also open it with a text editor if you want to examine the complete output.

## Create a GPU Implementation

Modify `./project/src/layer/custom/new-forward.cu` to create GPU implementation of the forward convolution. In your template, the host code is separated in 3 parts. `conv_forward_gpu_prolog` allocates memory and copies data from host to device (Note: the device pointers given to you in this function are double pointers). `conv_forward_gpu` computes kernel dimensions and invokes kernel. `conv_forward_gpu_epilog` copies output back to host and free the device memory. You should implement your kernel code from Lecture 12 in `conv_forward_kernel`.

Modify `m1_gpu.slurm` to run with batch_size=100. Run

    srun ./m1_gpu 100 > m1_gpu.out

to runs the code specified in `./project/src/layer/custom/new-forward.cu` program for a batch of 100 input images.
If your implementation is correct, it will show the same correctness as Milestone 1.

The file m1_gpu.out includes two performance metrics. "Op time" refers to the time taken by `conv_forward_gpu`. "Layer time" represents the total duration of `conv_forward_gpu_prolog`, `conv_forward_gpu`, and `conv_forward_gpu_epilog` combined.

The sum of OP times on batch_size=10000 should be approximately 70 ms if you implement the basic kernel from Lecture 12 correctly. You must have correct accuracies and total OP time less than 210 ms to earn full credits on the coding part. To quicken development time, `m1_gpu.cc` takes one optional argument: the dataset size. See [Specifying Batch Size](#specifying-batch-size).

`m1_gpu.slurm` will run your code in a single A40 GPU. If you use a different GPU model (such as your personal GPU), the first run may be slower due to JIT caching. For more information, refer to the [Appendix: JIT Caching](#jit-caching).

**To speed up testing, replace `#SBATCH --constraint="projects,perf,nvperf"` with `#SBATCH --constraint="projects"` when testing your code.**

Your jobs run on A40x4 nodes, where each node has four A40 GPUs. Since each job uses only one GPU, a single node can handle up to four jobs simultaneously. However, when profiling, it's important to avoid external interference. Adding the `perf,nvperf` constraints ensures exclusive access to the node. The trade-off is longer wait times if the cluster is busy. It's recommended to remove `perf,nvperf` when testing the correctness of your code, and include it only during profiling.

### Use Nsight-Systems and Nsight-Compute for initial Performance Results

**Before you do any profiling, make sure your implementation achieves desired accuracy. Also make sure you do not have any memory errors by running `cuda-memcheck`. See [Checking for Errors](README.md#checking-for-errors) on how to run this.**

***System level profiling using Nsight-Systems***

We will learn how to use `nsys` (Nsight Systems) to profile the execution at the application level.

Once you've gotten the appropriate accuracy results, generate a profile using `nsys`. Make sure `m1_gpu.slurm` is configured for a GPU run.
You have to remove `-DCMAKE_CXX_FLAGS=-pg` in `run.sh` and make line of your `run.sh`:

    cmake ./project/ && make -j8

Then, modify `m1_gpu.slurm` to generate a profile instead of just executing the code the output is inside `profile.out` file.

    srun nsys profile --stats=true ./m1_gpu > profile.out

You should see something that looks like the following (but not identical):

```bash 
......

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)   Max (ns)    StdDev (ns)          Name         
 --------  ---------------  ---------  ------------  -------------  --------  -----------  -----------  ---------------------
     99.9  351,122,724,860      3,519  99,779,120.4  100,089,303.0     2,855  100,130,281  5,413,528.2  poll                 
      0.1      283,382,530        925     306,359.5       14,207.0     1,051   20,208,549  1,050,067.9  ioctl                
     ......               
      0.0            1,913          1       1,913.0        1,913.0     1,913        1,913          0.0  bind                 

[5/8] Executing 'cudaapisum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)     Med (ns)    Min (ns)   Max (ns)    StdDev (ns)            Name         
 --------  ---------------  ---------  ------------  -----------  --------  -----------  ------------  ----------------------
     ......     

[6/8] Executing 'gpukernsum' stats report

 Time (%)  Total Time (ns)  Instances    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)      GridXYZ         BlockXYZ                                               Name                                          
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  ---------------  --------------  ----------------------------------------------------------------------------------------
     ......                                                                   

[7/8] Executing 'gpumemtimesum' stats report

 Time (%)  Total Time (ns)  Count    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)       Operation     
 --------  ---------------  -----  -------------  -------------  -----------  -----------  ------------  ------------------
     ......

[8/8] Executing 'gpumemsizesum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)   StdDev (MB)      Operation     
 ----------  -----  --------  --------  --------  ---------  -----------  ------------------
     ......

```

The CUDA API Statistics section shows the CUDA API calls that are executed. The CUDA Kernel Statistics lists all the kernels that were executed during the profiling session. There are also more details on the CUDA memory operations (CudaMemcpy) listed.
There are columns corresponding to percentage of time consumed, total time, number of calls, and average/min/max time of those calls. Use **your** `nsys` profiling output corresponding to the section above to answer the questions for your report.

Think about the distinction between a CUDA API call and a kernel launch, and describe it briefly in your report.
The CUDA documentation describes [kernels](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels) and the [programming interface](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-interface).

You can find more information about `nsys` in the [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/UserGuide/#cli-profiling)

***Kernel level profiling using Nsight-Compute***

Nsight-Systems does not give you detailed kernel level performance metrics. For that, we will need to use `ncu` (Nsight-Compute).

Modify `m1_gpu.slurm` to use `ncu` to save some timeline and analysis information.

    srun ncu -f -o analysis_file <your command here>

This will generate `analysis_file.ncu-rep`. `ncu` can also be invoked through its alias, `nv-nsight-cu-cli`.

You can use the NVIDIA Nsight Compute GUI (`nv-nsight-cu`) to view those files.
You will need to install NVIDIA NSight Compute on your own machine. It can be downloaded from NVIDIA's [website](https://developer.nvidia.com/nsight-compute) as a standalone application.

## Report

You will complete your report on Canvas.

| Report Questions                                                                                                                                                                                                                                                                                              |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Show output of running Mini-DNN on the CPU for batch size of 1k images                                                                                                                                                                                                                                        |
| List Op Times and whole program execution time (CPU convolution) for batch size of 1k images                                                                                                                                                                                                                  |
| Show percentage of total execution time of your CPU program spent in your forward pass function with `gprof`                                                                                                                                                                                                  |
| Show OP Times, whole program execution time, and accuracy of the GPU convolution for different batch sizes                                                                                                                                                                                                    |
| Demonstrate `nsys` profiling the GPU execution                                                                                                                                                                                                                                                                |
| Include a list of all kernels that cumulatively consume more than 90% of the program time (listing from the top of your `nsys` results until the cumulative `Time` is greater than 90%)                                                                                                                       |
| Include a list of all CUDA API calls that cumulatively consume more than 90% of the program time                                                                                                                                                                                                              |
| Include an explanation of the difference between kernels and API calls                                                                                                                                                                                                                                        |
| Screenshot of the GPU SOL utilization in Nsight-Compute GUI for your kernel profiling data (for the first kernel launch of the two convolution kernels). On the upper right corner, you have a drop-down option "Save as image". The default selection is "Copy as image". Use this image as your screenshot. |

## Submitting milestone 1 for grading

To submit your work for grading, add, commit, and push your files:

* ```git add -u```
* ```git commit -m "some comment"```
* ```git push origin main```
  Make sure to complete your report on Canvas. Double check you include all items listed in the Deliverables for this milestone.

## Rubric

Milestone 1 contributes to 20% of the overall project score. The score is determined by the correctness and timing of your code and the report on Canvas.

Milestone 1 ( 20% )
 * CPU Implementation ( 5% )
 * GPU Implementation ( 5% )
 * Report ( 10% )

## Appendix

### JIT Caching

`nvcc`, the CUDA compiler driver, uses a two-stage compilation model. The first stage compiles source device code to PTX virtual assembly, and the second stage compiles the PTX to binary code for the target architecture. The CUDA driver can execute the second stage compilation at run time, compiling the PTX virtual assembly “Just In Time” to run it.

JIT compilation may introduce a delay during the first run of an executable. However, once compiled, the binary code is cached, allowing subsequent runs to be faster. For instance, the sum of Op Times of the reference `m1_gpu` implementation is around 120 ms on its first run, but drops to about 70 ms on following runs due to caching.

To eliminate JIT overhead, we instruct `nvcc` to generate binary code for the target architecture ahead of time. In [CMakeLists.txt](project/CMakeLists.txt), we specify the following:

```CMake
list( APPEND CUDA_NVCC_FLAGS "; -arch=sm_86; -std=c++11; -lineinfo")
```

The `-arch=sm_86` flag compiles binary code directly for the sm_86 architecture (such as the A40 GPU), ensuring that JIT overhead is avoided when running jobs on Delta.

Optional reading: [CUDA Pro Tip: Understand Fat Binaries and JIT Caching](https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/)
