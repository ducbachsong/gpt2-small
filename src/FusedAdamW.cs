// FusedAdamW.cs — AdamW with its whole step in one CUDA kernel (cuda/adamw.cu).
//
// The same optimiser as AdamW.cs, on the same flat buffers, with the same
// results bit for bit; only how the step runs on the GPU changes:
//
//     AdamW.Step         11 tensor operations, each a kernel that reads its
//                        inputs from GPU memory and writes its result back:
//                        26 reads and writes per value of the model
//     FusedAdamW.Step    1 kernel: each value's p, g, m and v are read once,
//                        updated in registers, and written once: 8
//
// The maths per value is a handful of multiply-adds, far less than the GPU
// can do while it waits for memory. A step is limited by memory traffic, so
// 8 trips instead of 26 is where the speed comes from.
//
// ClipGradientNorm stays outside the kernel. The norm needs every gradient
// before any value can be updated, and the threads of one kernel cannot all
// wait for each other: its blocks run in no set order, and a block waiting
// for one that has not started yet would wait forever. So a training step is
// 2 kernels: the norm (LibTorch's reduction), then this step. The 4 small
// operations AdamW does on the norm to get the clip factor are done by the
// kernel itself.
//
// CUDA only; on the CPU, AdamW. Trainer picks this one on a GPU.
using System.Runtime.InteropServices;
using TorchSharp;
using static TorchSharp.torch;

namespace Gpt2Trainer;

public sealed class FusedAdamW : IOptimizer
{
    const int ThreadsPerBlock = 256;
    const int ValuesPerThread = 4;         // one float4 per buffer: see adamw.cu

    readonly FlatParameters flat;
    readonly Tensor firstMoment, secondMoment;
    readonly Config config;
    readonly CudaKernel kernel;
    readonly IntPtr parameters, gradients, firstMomentAddress, secondMomentAddress;   // in GPU memory, fixed for good
    long stepCount;                        // steps taken, for the bias correction
    Tensor? gradientNorm;                  // from ClipGradientNorm, read on the GPU by the next Step
    double maxNorm;

    /// Compiles the kernel for the model's GPU (the first time in this process),
    /// then sets up the moments as AdamW does: 0, in the model's layout.
    public FusedAdamW(FlatParameters flat, Config config)
    {
        // What the kernel relies on, checked once here, not by a wrong answer later.
        // FlatParameters' 64-value segments keep the counts multiples of 4.
        if (flat.Parameters.device_type != DeviceType.CUDA)
            throw new ArgumentException("FusedAdamW runs on a CUDA GPU; on the CPU, use AdamW");
        if (flat.Parameters.dtype != ScalarType.Float32)
            throw new ArgumentException($"FusedAdamW works on float32, not {flat.Parameters.dtype}");
        if (flat.Parameters.numel() % ValuesPerThread != 0 || flat.DecayedCount % ValuesPerThread != 0)
            throw new ArgumentException($"the flat buffers' lengths must be multiples of {ValuesPerThread}");

        // The kernel first: it is what fails on a machine without NVRTC, and then
        // nothing has been allocated on the GPU yet.
        kernel = CudaKernel.Get("adamw.cu", "adamw_step", flat.Parameters.device_index);

        this.flat = flat;
        this.config = config;
        firstMoment = zeros_like(flat.Parameters).DetachFromDisposeScope();
        secondMoment = zeros_like(flat.Parameters).DetachFromDisposeScope();

        // The four buffers never move, so their addresses are looked up once. Each
        // must start on 16 bytes, for the float4 loads; LibTorch's allocator
        // starts every buffer on 512.
        parameters = CudaKernel.DevicePointer(flat.Parameters);
        gradients = CudaKernel.DevicePointer(flat.Gradients);
        firstMomentAddress = CudaKernel.DevicePointer(firstMoment);
        secondMomentAddress = CudaKernel.DevicePointer(secondMoment);
        if (new[] { parameters, gradients, firstMomentAddress, secondMomentAddress }.Any(address => address % 16 != 0))
            throw new ArgumentException("the flat buffers must start on 16 bytes, for float4 loads");
    }

    public void ZeroGradients() => flat.Gradients.zero_();    // 1 kernel

    // Clipping by the global norm, as AdamW.ClipGradientNorm (see there), split
    // differently: here only the norm, one reduction by LibTorch. The factor,
    // min(1, maxNorm / (norm + 1e-6)), is worked out by the kernel in the next
    // Step, from this norm, without it ever leaving the GPU.
    //
    /// Out: the norm before clipping, as a 0-d tensor still on the GPU.
    public Tensor ClipGradientNorm(double maxNorm)
    {
        using var _ = no_grad();
        var norm = flat.Gradients.norm();                     // the padding is 0: it adds nothing
        gradientNorm?.Dispose();
        gradientNorm = norm.alias().DetachFromDisposeScope();  // a handle of our own: the caller may dispose theirs
        this.maxNorm = maxNorm;
        return norm;
    }

    // AdamW step t, as AdamW.Step (see there for the maths), in one launch. The
    // kernel does the per-value work; here are the numbers that are the same
    // for every value, worked out once on the CPU in double and rounded to
    // float as LibTorch rounds a number an operation is given: AdamW.Step passes
    // these same numbers to its tensor operations, so both round alike.
    public void Step(double learningRate)
    {
        stepCount++;
        double biasCorrection1 = 1 - Math.Pow(config.Beta1, stepCount);
        double biasCorrection2 = 1 - Math.Pow(config.Beta2, stepCount);
        double stepSize = learningRate / biasCorrection1;

        var arguments = new AdamWArguments
        {
            Parameters = parameters,
            Gradients = gradients,
            FirstMoment = firstMomentAddress,
            SecondMoment = secondMomentAddress,
            GradientNorm = gradientNorm is null ? IntPtr.Zero : CudaKernel.DevicePointer(gradientNorm),
            Count = flat.Parameters.numel(),
            DecayedCount = flat.DecayedCount,
            MaxNorm = (float)maxNorm,
            WeightDecayFactor = (float)(1 - learningRate * config.WeightDecay),
            Beta1 = (float)config.Beta1,
            OneMinusBeta1 = (float)(1 - config.Beta1),
            Beta2 = (float)config.Beta2,
            OneMinusBeta2 = (float)(1 - config.Beta2),
            // AdamW.Step divides by sqrt(1-b2ᵗ) with div_(a number). LibTorch's CUDA kernel
            // turns that into a multiply by the reciprocal, 1 / sqrt(1-b2ᵗ) worked out in float.
            InverseSqrtBiasCorrection2 = 1f / (float)Math.Sqrt(biasCorrection2),
            Eps = (float)config.Eps,
            NegativeStepSize = (float)-stepSize,
        };
        long groups = arguments.Count / ValuesPerThread;       // one thread per 4 values
        kernel.Launch((groups + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock, arguments);

        // The kernel is only queued, but the norm can go now: GPU memory LibTorch
        // frees is reused only by work queued after this kernel, on the same stream.
        gradientNorm?.Dispose();
        gradientNorm = null;
    }

    /// The kernel's one argument: the twin of struct AdamWArguments in adamw.cu,
    /// which has what each field means. The same fields in the same order with
    /// the same sizes, so the bytes line up: 8-byte fields first, then 4-byte ones.
    [StructLayout(LayoutKind.Sequential)]
    struct AdamWArguments
    {
        public IntPtr Parameters, Gradients, FirstMoment, SecondMoment;
        public IntPtr GradientNorm;                            // 0, a null pointer: no clipping
        public long Count, DecayedCount;
        public float MaxNorm, WeightDecayFactor;
        public float Beta1, OneMinusBeta1, Beta2, OneMinusBeta2;
        public float InverseSqrtBiasCorrection2, Eps, NegativeStepSize;
    }
}
