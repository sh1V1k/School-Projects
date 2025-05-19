# Milestone 2: Implementing Convolution with Matrix Multiplication


***Deadline: November 8th, 2024 8PM***

Please check [this](https://campuswire.com/c/GF7DDC41F/feed/653) Campuswire post regularly for project FAQ and updates.

For each milestone, you will also need to complete a report on Canvas. The table below contains all of the deliverables.

| Deliverables                                                             |
| ------------------------------------------------------------------------ |
| Implement the input matrix unrolling kernel                              |
| Complete the code of implementing convolution with matrix multiplication |
| Profile your implementation with `ncu`                                   |
| Complete your report on Canvas                                           |
| Submit your work for grading                                             |

## Table of Contents

- [Setup](#setup)
- [Input Matrix Unrolling](#input-matrix-unrolling)
- [Report](#report)
- [Submitting milestone 2 for grading](#submitting-milestone-2-for-grading)
- [Rubric](#rubric)
- [Appendix](#appendix)

## Setup

1. To retrieve updates for Milestone 2, go to your `ece408git` folder and run the following:

   * `git fetch release`
   * `git merge release/main -m "Project checkpoint 2" --allow-unrelated-histories`
   * `git push origin main`

   If you are asked to enter your git credentials (PAT) each time you try to pull or push, you can try to store your git credentials once and for all: just type git config credential.helper store in your Delta terminal.

2. To compile, simply type `./run.sh build`. This will build everything from Milestone 1 and additionally generate a binary called `m2`, which you will use for Milestone 2.

3. To execute your code, type `sbatch m2.slurm`. This will schedule the execution of your program on one of the next available compute nodes.

4. To clean, type `./run.sh clean`. This will remove all the files generated during the compilation and execution process.

## Input Matrix Unrolling

In lecture 12, we learned how to use matrix multiplication to implement convolution. In order to do so, we need to unroll the input features. In Milestone 2, you will implement the input unrolling kernel.

Modify `./project/src/layer/custom/matmul_unroll.cu` to complete the GPU convolution implementation with matrix multiplication.

The convolution forward process consists of the following steps:
- Unroll the input matrix
- Perform matrix multiplication
- Permute the result of the matrix multiplication. The output feature map initially has the shape `Map_out` x `Batch` x `Height_out` x `Width_out`, which needs to be permuted to `Batch` x `Map_out` x `Height_out` x `Width_out`.

The matrix multiplication kernel and the permute kernel are provided. You will focus on implementing the input matrix unrolling kernel.

In lecture 12, we covered how to unroll the input features for a single image. To unroll a batch of images, the unrolled matrix for each image in the batch should be concatenated along the row dimension. In other words, if the unrolled matrix of a single image has a shape of `H` x `W`, then the unrolled matrix of a batch of images will have a shape of `H` x `(Batch * W)`.

In your template, the host code is separated into 3 parts. `conv_forward_gpu_prolog` allocates memory and copies data from host to device (Note: the device pointers given to you in this function are double pointers). `conv_forward_gpu` invokes input unrolling and matrix multiplication kernel. `conv_forward_gpu_epilog` copies output back to host and frees the device memory.

To sum up, your task is to:
- Implement the `matrix_unrolling_kernel`.
- Complete host code in `conv_forward_gpu_prolog`, `conv_forward_gpu`, and `conv_forward_gpu_epilog`.

Same to Milestone 1, `m2` takes a command-line argument batch size. For example, in `m2.slurm`, the line

```bash
srun ./m2 100 > m2.out
```

runs the code specified in `./project/src/layer/custom/matmul_unroll.cu` program for a batch of 100 input images.

If your implementation is correct, it will show the same accuracy as Milestone 1.

The sum of OP times on batch_size=10000 should be approximately 200 ms. You must have correct accuracies and total OP time less than 1200 ms to earn full credits on the coding part.

**To speed up testing, replace `#SBATCH --constraint="projects,perf,nvperf"` with `#SBATCH --constraint="projects"` when testing your code.**

### Profiling

**Before you do any profiling, make sure your implementation achieves desired accuracy. Also make sure you do not have any memory errors by running `cuda-memcheck`. See [Checking for Errors](README.md#checking-for-errors) on how to run this.**


In Milestone 1, you practiced using `ncu`. In Milestone 2, you will collect more detailed profiling information by instructing ncu to use the `--set full` option:

```bash
srun ncu --set full -f -o analysis_file <your command here>
```

In your report, you will explain why matrix multiplication with input unrolling is faster or slower than the basic GPU convolution in Milestone 1, based on the profiling results. The Memory Chart will be helpful, located in the Memory Workload Analysis section of the details page.

When using the Nsight Compute GUI, you can set your Milestone 1 GPU code as the baseline. This allows you to compare the performance between the two Milestones.

## Report

You will complete your report in the Quizzes section on Canvas. Below is a brief preview of the questions.

| Report Questions                                                                                                                                                                                                                                                                                                                                                            |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| How does this optimization work in theory? Expected behavior?                                                                                                                                                                                                                                                                                                               |
| How did you implement your code? Explain thoroughly and show code snippets.                                                                                                                                                                                                                                                                                                 |
| List 2 Op Times, whole program execution time, and accuracy for batch size of 100, 1k, and 10k images.                                                                                                                                                                                                                                                                      |
| For batch size of 10k images, is matrix multiplication with input unrolling faster or slower than your Milestone 1 basic GPU convolution in terms of Op Time? Explain why. Support your explanation with specific profiling data. (Include a Memory Chart screenshot from the first launch of your matrix multiplication kernel, located in the Memory Workload Analysis section of the details page.) |
| Propose an optimization for your implementation and explain how it improves performance or efficiency.                                                                                                                                                                                                                                                                      |
| List your references used while implementing this technique. (you must mention textbook pages at the minimum)                                                                                                                                                                                                                                                               |

## Submitting milestone 2 for grading

To submit your work for grading, add, commit, and push your files:

* ```git add -u```
* ```git commit -m "some comment"```
* ```git push origin main```
  Make sure to complete your report on Canvas. Double check you include all items listed in the Deliverables for this milestone.

## Rubric

Milestone 2 contributes to 20% of the overall project score. The score is determined by the correctness and timing of your code and the report on Canvas.

Milestone 2 ( 20% )
 * Code ( 10% )
 * Report ( 10% )

## Appendix

### Specifying the Convolution Implementation

In C++, a function's declaration can be separated from its implementation (definition), which enables the use of multiple implementations for the same function. For example, in the file [gpu-new-forward.h](project/src/layer/custom/gpu-new-forward.h), the member functions of the `GPUInterface` class are declared. [new-forward.cu](project/src/layer/custom/new-forward.cu) and [matrix-unroll.cu](project/src/layer/custom/matmul-unroll.cu) are two independent implementaions.

To specify which implementation to use, the root [CMakeLists.txt](project/CMakeLists.txt) file includes the corresponding source file.

```CMake
cuda_add_executable(m1_gpu m1_gpu.cc ./src/layer/custom/new-forward.cu)
target_link_libraries(m1_gpu ece408net MiniDNNLib)

cuda_add_executable(m2 m2.cc ./src/layer/custom/matmul-unroll.cu)
target_link_libraries(m2 ece408net MiniDNNLib)
```

In this example, the `m1_gpu` target uses the implementation from `new-forward.cu`, while `m2` uses the implementation from `matmul-unroll.cu`.
