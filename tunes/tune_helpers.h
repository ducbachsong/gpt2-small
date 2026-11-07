// tune_helpers.h — what every tune here uses: timing on the GPU (gpu_milliseconds, warm_up_gpu), the
// GPU a run was on (print_gpu, at the top of every result file), and memory bandwidth: the GB/s a kernel
// reached and its MBU (memory bandwidth utilization), that speed over the GPU's peak.
//
// Most of our kernels do a few flops per byte they read or write, so memory is what limits them, and MBU
// says how near a kernel comes to the most this GPU can give:
//
//     GB/s = bytes read and written / time          MBU = GB/s / peak GB/s
//
//     GPU          peak GB/s (about, from the spec sheets; print_gpu prints what the GPU itself reports)
//     T4                 320
//     L4                 300
//     A100 40GB        1,555
//     A100 80GB        2,039
//     H100 PCIe        2,039
//     H100 SXM         3,350
//     H200             4,800
//     B200             8,000
//
// The whole model is limited by the tensor cores instead (its matrix multiplies), and MFU (model flops
// utilization) says how much of them a training step uses:
//
//     MFU = tokens/s x flops per token / peak bfloat16 flops a second
#ifndef TUNES_TUNE_HELPERS_H
#define TUNES_TUNE_HELPERS_H

#include "../llmc/kernels/cuda_common.h" // cudaCheck
#include "../tests/test_helpers.h"       // start_run_log

#define TIMED_STEPS 20                // steps (or updates, or norms) timed of each side
#define WARM_UP_STEPS 2               // first steps left out of the averages: first-time work
#define SWEEP_ROUNDS 20               // rounds of a launch sweep: each times every launch once
#define WARM_UP_MILLISECONDS 1000.0f  // how long warm_up_gpu keeps the GPU busy
#define BITS_PER_BYTE 8
#define TRANSFERS_PER_CLOCK 2         // GPU memory moves data on both edges of its clock (double data rate)
#define GIGA 1e9
#define TERA 1e12
#define MILLISECONDS_PER_SECOND 1e3
#define KILO 1e3

/// Measures how long `run_val` takes on the GPU, in milliseconds: a mark on the GPU's queue
/// before it and one after; the GPU writes down the time as it reaches each.
template <typename Run>
static float gpu_milliseconds(Run run_val)
{
    cudaEvent_t start_ptr;
    cudaEvent_t end_ptr;
    cudaCheck(cudaEventCreate(&start_ptr));
    cudaCheck(cudaEventCreate(&end_ptr));
    cudaCheck(cudaEventRecord(start_ptr));
    run_val();
    cudaCheck(cudaEventRecord(end_ptr));
    cudaCheck(cudaEventSynchronize(end_ptr)); // wait until the GPU has passed the second mark
    float milliseconds_val = 0.0f;
    cudaCheck(cudaEventElapsedTime(&milliseconds_val, start_ptr, end_ptr));
    cudaCheck(cudaEventDestroy(start_ptr));
    cudaCheck(cudaEventDestroy(end_ptr));
    return milliseconds_val;
}

/// Keeps the GPU busy with `run_val` for about WARM_UP_MILLISECONDS before timing, so the first launches timed
/// do not run while its clock is still rising from idle.
template <typename Run>
static void warm_up_gpu(Run run_val)
{
    float total_val = 0.0f;
    while (total_val < WARM_UP_MILLISECONDS)
        total_val += gpu_milliseconds(run_val);
}

/// Gives this GPU's peak memory bandwidth in GB/s, to say how near a memory-bound kernel comes to it (its MBU):
/// TRANSFERS_PER_CLOCK x the memory clock x the bus width in bytes. An H100 PCIe: 2 x 1,593 MHz x 5,120 bits / 8
/// = 2,039 GB/s; a T4: 2 x 5,001 MHz x 256 bits / 8 = 320 GB/s.
static double peak_gigabytes_per_second(void)
{
    int memory_clock_khz_val = 0;
    int bus_bits_val = 0;
    cudaCheck(cudaDeviceGetAttribute(&memory_clock_khz_val, cudaDevAttrMemoryClockRate, 0));
    cudaCheck(cudaDeviceGetAttribute(&bus_bits_val, cudaDevAttrGlobalMemoryBusWidth, 0));
    double transfers_per_second_val = TRANSFERS_PER_CLOCK * (memory_clock_khz_val * 1e3);
    return transfers_per_second_val * (bus_bits_val / BITS_PER_BYTE) / GIGA;
}

/// Gives this GPU's peak dense bfloat16 flops a second on its tensor cores, for MFU: SMs x clock x the flops
/// one SM's tensor cores do each clock, which its generation (compute capability) sets. 0 for a generation not
/// in the table. The clock is the GPU's top boost clock, so this is a little above the spec sheets, which
/// count at a lower clock (an H100 PCIe: 820 TFLOPS here, 756 on its sheet), and an MFU from it a little below.
///
///     compute    GPUs         flops an SM a clock   SMs x clock x that
///     7.0, 7.5   V100, T4                   1,024   T4: 40 x 1,590 MHz = 65 T (float16: a T4 has no bfloat16)
///     8.0        A100                       2,048   108 x 1,410 MHz = 312 T
///     8.6, 8.9   A10, L4                    1,024   L4: 58 x 2,040 MHz = 121 T
///     9.0        H100, H200                 4,096   H100 PCIe: 114 x 1,755 MHz = 820 T
///     10.0       B200                       8,192   (its sheet: 2,250 T)
static double peak_bf16_flops(void)
{
    // 1. the generation, SMs and clock
    int major_val = 0;
    int minor_val = 0;
    int sms_val = 0;
    int clock_khz_val = 0;
    cudaCheck(cudaDeviceGetAttribute(&major_val, cudaDevAttrComputeCapabilityMajor, 0));
    cudaCheck(cudaDeviceGetAttribute(&minor_val, cudaDevAttrComputeCapabilityMinor, 0));
    cudaCheck(cudaDeviceGetAttribute(&sms_val, cudaDevAttrMultiProcessorCount, 0));
    cudaCheck(cudaDeviceGetAttribute(&clock_khz_val, cudaDevAttrClockRate, 0));

    // 2. an SM's tensor-core flops a clock, by generation
    int compute_val = major_val * 10 + minor_val; // 90 for 9.0
    double flops_per_sm_per_clock_val = 0.0;
    if (compute_val == 70 || compute_val == 75 || compute_val == 86 || compute_val == 87 || compute_val == 89)
        flops_per_sm_per_clock_val = 1024.0;
    else if (compute_val == 80)
        flops_per_sm_per_clock_val = 2048.0;
    else if (compute_val == 90)
        flops_per_sm_per_clock_val = 4096.0;
    else if (compute_val == 100)
        flops_per_sm_per_clock_val = 8192.0;

    // 3. all of them
    return sms_val * (clock_khz_val * KILO) * flops_per_sm_per_clock_val;
}

/// Gives the GB/s a kernel reached: the bytes it moved over its time.
static double gigabytes_per_second(double bytes_val, double milliseconds_val)
{
    return bytes_val / GIGA / (milliseconds_val / MILLISECONDS_PER_SECOND);
}

/// Prints the GPU and CUDA a run is on, first thing, so a saved result file says where its numbers came from:
///
///     GPU: NVIDIA H100 PCIe   compute 9.0   114 SMs   85.5 GB   peak memory bandwidth 2039 GB/s   peak bfloat16 820 TFLOPS   CUDA 12.8
static void print_gpu(void)
{
    // 1. what the GPU says about itself
    cudaDeviceProp properties_val;
    cudaCheck(cudaGetDeviceProperties(&properties_val, 0));
    int runtime_version_val = 0;
    cudaCheck(cudaRuntimeGetVersion(&runtime_version_val)); // 12080 for 12.8

    // 2. one line
    printf("GPU: %s   compute %d.%d   %d SMs   %.1f GB   peak memory bandwidth %.0f GB/s   peak bfloat16 %.0f "
           "TFLOPS   CUDA %d.%d\n\n",
           properties_val.name, properties_val.major, properties_val.minor, properties_val.multiProcessorCount,
           properties_val.totalGlobalMem / GIGA, peak_gigabytes_per_second(), peak_bf16_flops() / TERA,
           runtime_version_val / 1000, (runtime_version_val % 1000) / 10);
}

/// Gives the results folder for this GPU, from its name: "Tesla T4" -> T4, "NVIDIA H100 PCIe" -> H100, and so on
/// for L4, A100, H200 and B200 (an L40 or L40S is not an L4: checked first). Any other GPU gets its own name,
/// spaces as _, without "NVIDIA ".
static std::string gpu_folder_name(void)
{
    // 1. the name
    cudaDeviceProp properties_val;
    cudaCheck(cudaGetDeviceProperties(&properties_val, 0));
    std::string name_val = properties_val.name;
    // 2. a folder we keep, if the name has one
    const char *known_list[] = {"H100", "H200", "B200", "A100", "L40S", "L40", "L4", "T4"};
    for (const char *known_ptr : known_list)
        if (name_val.find(known_ptr) != std::string::npos)
            return known_ptr;
    // 3. else the name itself
    if (name_val.rfind("NVIDIA ", 0) == 0)
        name_val = name_val.substr(7);
    for (char &letter_val : name_val)
        if (letter_val == ' ')
            letter_val = '_';
    return name_val;
}

/// Starts a tune's run, first thing in its main: its output goes to the console and to a new result file,
/// tunes/results/<this GPU>/<kernel>_<date>_<time>.txt (start_run_log), which opens with the GPU line (print_gpu).
///
///   file_ptr   __FILE__ of the tune's main: tunes/tune_gelu.cu gives gelu
static void start_tune_log(const char *file_ptr)
{
    std::string folder_val = "tunes/results/" + gpu_folder_name();
    start_run_log(folder_val.c_str(), file_ptr, "tune_", "txt");
    print_gpu();
}

#endif
