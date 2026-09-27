// adamw.cu — AdamW's whole step as one CUDA kernel: what FusedAdamW.cs launches.
//
// AdamW.Step (AdamW.cs) runs 11 tensor operations over the flat buffers. Each is
// a kernel of its own, which reads its inputs from GPU memory and writes its
// result back, so each value of the model is read or written 26 times per step:
//
//                                   reads          writes     trips
//     1   g = g * clip              g              g          2
//     2   p = p * (1 - lr wd)       p              p          2
//     3   m = m * b1                m              m          2
//     4   m = m + (1-b1) g          m, g           m          3
//     5   v = v * b2                v              v          2
//     6   v = v + (1-b2) g·g        v, g, g        v          4
//     7   d = sqrt(v)               v              d          2
//     8   d = d / sqrt(1-b2ᵗ)       d              d          2
//     9   d = d + eps               d              d          2
//    10   p = p - (lr/(1-b1ᵗ)) m/d  p, m, d        p          4
//    11   g = 0                                    g          1
//                                                             --
//                                                             26
//
// d is the denominator: a temporary buffer as big as the model, which
// AdamW.Step allocates just to hold it between operations 7 and 10. Line 6
// reads g twice because addcmul_ reads each of its two factors, even when
// both are the same tensor. Line 2 covers only the matrices, but they are
// nearly all of the model's values.
//
// This kernel reads each value's p, g, m and v once, does all of AdamW in
// registers, and writes them back once: 8 reads and writes instead of 26. The
// maths is a handful of multiply-adds per value, far less than the GPU can do
// while it waits for memory, so a step is limited by memory traffic, and the
// saved trips are where the speed comes from.
//
// One thread per 4 neighbouring values, which it moves with one 16-byte load
// (a float4) per buffer instead of four 4-byte ones:
//
//     values   [ 0  1  2  3 | 4  5  6  7 | 8  9 10 11 | ... ]
//     threads  [  thread 0  |  thread 1  |  thread 2  | ... ]      256 threads per block
//
// FlatParameters starts every segment on a multiple of 64 values, so the buffer
// length and DecayedCount are multiples of 4: a thread's 4 values never run past
// the end, and are either all decayed or all not.
//
// Bit for bit with AdamW.Step: each line of adamw_update below is one of its
// tensor operations, rounded the way LibTorch's CUDA kernel for that operation
// rounds, in the same order. The intrinsics spell out each rounding:
//
//     __fmul_rn  __fadd_rn  __fdiv_rn  __fsqrt_rn   one operation, rounded to nearest,
//                                                  never merged with the next one
//     __fmaf_rn(a, b, c)                            a*b + c rounded once (fused multiply-add):
//                                                  what the compiler makes of LibTorch's
//                                                  add_(alpha), addcmul_ and addcdiv_
//
// A plain `a * b + c` would leave it to the compiler whether to round once or
// twice, and the other choice changes the last bit.

// The kernel's arguments, as one struct passed by value. FusedAdamW.cs fills its
// twin (AdamWArguments there): the same fields in the same order with the same
// sizes, so the bytes line up. The 8-byte fields come first and the 4-byte ones
// after, so neither side puts gaps between them.
struct AdamWArguments
{
    float* parameters;                  // p   flat.Parameters
    float* gradients;                   // g   flat.Gradients: read, then left at 0
    float* firstMoment;                 // m   moving average of g
    float* secondMoment;                // v   moving average of g²
    const float* gradientNorm;          // what ClipGradientNorm measured; nullptr: no clipping
    long long count;                    // values in each buffer, a multiple of 4
    long long decayedCount;             // values [0, decayedCount) get weight decay, a multiple of 4
    float maxNorm;                      // clipping: the norm g is scaled down to
    float weightDecayFactor;            // 1 - lr wd
    float beta1, oneMinusBeta1;         // b1, 1 - b1
    float beta2, oneMinusBeta2;         // b2, 1 - b2
    float inverseSqrtBiasCorrection2;   // 1 / sqrt(1 - b2ᵗ)
    float eps;
    float negativeStepSize;             // -lr / (1 - b1ᵗ)
};

// The clip factor, as AdamW.ClipGradientNorm works it out with 4 tensor
// operations on the norm, (norm + 1e-6).reciprocal_().mul_(maxNorm).clamp_max_(1.0):
//
//     clip = min(1, maxNorm / (norm + 1e-6))
//
// E.g. norm 13, maxNorm 1.3 -> 0.1: every gradient is scaled by 0.1. Norm 0.5,
// maxNorm 1 -> 2, capped at 1: nothing changes. Every thread works it out for
// itself from the same norm, 4 operations on one number: cheaper than 4 kernel
// launches.
__device__ __forceinline__ float clip_factor(const AdamWArguments& a)
{
    if (a.gradientNorm == nullptr) return 1.0f;                             // no ClipGradientNorm: g * 1 = g
    float clip = __fmul_rn(__fdiv_rn(1.0f, __fadd_rn(*a.gradientNorm, 1e-6f)), a.maxNorm);
    return clip > 1.0f ? 1.0f : clip;                                       // clamp_max_(1.0): NaN stays NaN, as there
}

// One value's AdamW step t. Each line is one of AdamW.Step's tensor operations:
//
//     g = g * clip                        gradients.mul_(clipScale)
//     p = p * (1 - lr wd)                 decayed.mul_(1 - lr wd)                      matrices only
//     m = b1 m + (1-b1) g                 firstMoment.mul_(b1).add_(g, alpha: 1-b1)
//     v = b2 v + (1-b2) g²                secondMoment.mul_(b2).addcmul_(g, g, value: 1-b2)
//     d = sqrt(v) / sqrt(1-b2ᵗ) + eps     secondMoment.sqrt().div_(sqrt(1-b2ᵗ)).add_(eps)
//     p = p - (lr / (1-b1ᵗ)) m / d        parameters.addcdiv_(m, d, value: -stepSize)
//     g = 0                               gradients.zero_()
//
// d multiplies by 1/sqrt(1-b2ᵗ) instead of dividing by sqrt(1-b2ᵗ): LibTorch's
// CUDA kernel divides by a plain number that way, so this must too, to round
// the same.
__device__ __forceinline__ void adamw_update(float& p, float& g, float& m, float& v,
                                             float clip, bool decayed, const AdamWArguments& a)
{
    g = __fmul_rn(g, clip);
    if (decayed) p = __fmul_rn(p, a.weightDecayFactor);
    m = __fmaf_rn(a.oneMinusBeta1, g, __fmul_rn(m, a.beta1));
    v = __fmaf_rn(a.oneMinusBeta2, __fmul_rn(g, g), __fmul_rn(v, a.beta2));
    float d = __fadd_rn(__fmul_rn(__fsqrt_rn(v), a.inverseSqrtBiasCorrection2), a.eps);
    p = __fmaf_rn(a.negativeStepSize, __fdiv_rn(m, d), p);
    g = 0.0f;
}

// The kernel. One launch starts many threads, all running this same function
// at the same time, so each must first work out which 4 values are its own.
// The GPU gives every thread three numbers:
//
//     blockIdx.x    which block it is in: 0, 1, 2, ...
//     blockDim.x    threads per block: 256 (FusedAdamW launches it that way)
//     threadIdx.x   which thread it is inside its block: 0 .. 255
//
// Its number over all blocks, `group`, is then like a seat number in rows of
// 256 seats: block 2, thread 5 -> 2 x 256 + 5 = 517, so it updates values
// 2068 .. 2071 (4 x 517 .. 4 x 517 + 3) of every buffer. The (long long) makes
// the multiply 64-bit, so it cannot overflow on a big model.
//
// Blocks come whole, so the launch rounds the number of threads up. With 2,000
// values (500 groups of 4), 2 blocks = 512 threads start: threads 500 .. 511
// have no values, and must stop before they read or write past the end of the
// buffers.
//
// extern "C" keeps the name as written, so the driver finds it as "adamw_step".
extern "C" __global__ void adamw_step(const AdamWArguments a)
{
    long long group = (long long)blockIdx.x * blockDim.x + threadIdx.x;   // which 4 values are mine
    if (group * 4 >= a.count) return;                   // none: a spare thread in the last block

    float clip = clip_factor(a);
    bool decayed = group * 4 < a.decayedCount;          // all 4 or none: decayedCount is a multiple of 4

    // Read: one 16-byte load per buffer.
    float4 p = reinterpret_cast<float4*>(a.parameters)[group];
    float4 g = reinterpret_cast<float4*>(a.gradients)[group];
    float4 m = reinterpret_cast<float4*>(a.firstMoment)[group];
    float4 v = reinterpret_cast<float4*>(a.secondMoment)[group];

    // Update, in registers.
    adamw_update(p.x, g.x, m.x, v.x, clip, decayed, a);
    adamw_update(p.y, g.y, m.y, v.y, clip, decayed, a);
    adamw_update(p.z, g.z, m.z, v.z, clip, decayed, a);
    adamw_update(p.w, g.w, m.w, v.w, clip, decayed, a);

    // Write: one 16-byte store per buffer.
    reinterpret_cast<float4*>(a.parameters)[group] = p;
    reinterpret_cast<float4*>(a.gradients)[group] = g;
    reinterpret_cast<float4*>(a.firstMoment)[group] = m;
    reinterpret_cast<float4*>(a.secondMoment)[group] = v;
}
