# Bamboo-SMP (B-SMP)

The Stable Marriage Problem (SMP) is a combinatorial optimization problem aimed at creating stable pairings between two
groups, traditionally referred to as men and women.
SMP has been widely applied across various domains, including healthcare, education, and cloud computing, to optimize
resource allocation, matching, and utilization. The classical approach to solving SMP is based on the Gale-Shapley
algorithm,
which constructs stable pairings sequentially.
However, this algorithm is both time-consuming and data-intensive, resulting in significant slowdowns even for a
moderate number of participants.
Despite various attempts to parallelize the Gale-Shapley algorithm over the years, progress has been consistently
impeded by three major bottlenecks:
(1) frequent and expensive data movement, (2) high synchronization overhead, and (3) workload-dependent and irregular
data accessing and parallel processing patterns.

To resolve the above-mentioned bottlenecks, in this paper, we introduce Bamboo-SMP, a highly efficient parallel SMP
algorithm, and its implementation in a hybrid environment of GPUs and CPUs.
We have made three key development efforts to achieve high performance for Bamboo-SMP. First, Bamboo-SMP effectively
exploits the data accessing locality with a lightweight data structure to maximize the **"shared residence
space"**. Second, Bamboo-SMP employs an advanced hardware atomic operation to decrease execution latency with **"
low contention"**. Third, Bamboo-SMP is implemented in a hybrid environment of GPU and CPU, leveraging the high
bandwidth
of the GPU for massive parallel operations and the low latency of CPU for fast sequential tasks. By fostering **"
mutual complementarity"** between CPU and GPU, Bamboo-SMP attains superior performance, consistently exceeding the best
existing methods by **6.69x to 21.44x** across a wide range of workloads.
Moreover, Bamboo-SMP demonstrates excellent scalability, efficiently solving large-scale SMP instances while achieving
sustained speedups of **5.6× to 13.8×** on 4 GPUs.

<br>

## Hardware Requirements

* At least one **NVIDIA GPU**

<br>

## Software Requirements

Please ensure your system meets the following minimum software versions (greater than or equal to these values):

* **bash**
* **wget**
* **CUDA:** 12.6
* **GCC:** 11.4.0
* **CMake:** 3.22.1

<br>

## Setup & Execution

``To run **BambooSMP**, execute the following commands from the project root directory.

### Build

```bash
./scripts/runme.sh
```

This script automatically downloads dependencies, installs them, and compiles the project.
The compiled binary `bsmp_exe` will be stored inside the `bin/` directory.

### Run

Execute the BambooSMP engine using:

```bash
./bin/bsmp_exe 10 CONGESTED
```

Here:

* The first argument specifies the **size of the workload**.
* The second argument specifies the **workload type**, which can be one of:
    * `CONGESTED`
    * `SOLO`
    * `RANDOM`
    * `PERFECT`

<br>

### Clean

To reset the project directory to the initial state, run this script:

```bash
./scripts/clean.sh
```
