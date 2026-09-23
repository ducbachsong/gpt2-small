// AdamW.cs — the optimiser, the learning-rate schedule and gradient clipping, by hand.
//
// Nothing here makes the CPU wait for the GPU: the learning rate and the bias
// corrections are plain doubles, and clipping keeps the norm on the GPU
// (torch.nn.utils.clip_grad_norm_ in TorchSharp returns a double, which would
// stop the CPU once per step).
using TorchSharp;
using static TorchSharp.torch;

namespace Gpt2Trainer;

public sealed class AdamW
{
    readonly Tensor[] parameters, m, v;
    readonly bool[] decay;
    readonly Config config;
    public long T { get; set; }            // steps taken, for the bias correction

    /// Weight decay on the matrices and embeddings only (dim >= 2), not on
    /// biases or LayerNorm, as in traingpt2.py.
    public AdamW(IEnumerable<Tensor> parameters, Config config)
    {
        this.config = config;
        this.parameters = parameters.Where(p => p.requires_grad).ToArray();
        using var _ = no_grad();
        m = this.parameters.Select(p => zeros_like(p).DetachFromDisposeScope()).ToArray();
        v = this.parameters.Select(p => zeros_like(p).DetachFromDisposeScope()).ToArray();
        decay = this.parameters.Select(p => p.dim() >= 2).ToArray();
    }

    public IReadOnlyList<Tensor> Parameters => parameters;
    public IReadOnlyList<Tensor> Moments => m.Concat(v).ToArray();

    public void ZeroGrad()
    {
        foreach (var p in parameters) p.grad?.zero_();
    }

    //   m = b1 m + (1-b1) g           v = b2 v + (1-b2) g²
    //   p = p - lr wd p               (decoupled weight decay: the W in AdamW)
    //   p = p - lr (m / (1-b1ᵗ)) / (sqrt(v / (1-b2ᵗ)) + eps)
    public void Step(double lr)
    {
        using var _ = no_grad();
        using var scope = NewDisposeScope();
        T++;
        double correction1 = 1 - Math.Pow(config.Beta1, T);
        double sqrtCorrection2 = Math.Sqrt(1 - Math.Pow(config.Beta2, T));
        for (int i = 0; i < parameters.Length; i++)
        {
            var g = parameters[i].grad;
            if (g is null) continue;
            m[i].mul_(config.Beta1).add_(g, alpha: 1 - config.Beta1);
            v[i].mul_(config.Beta2).addcmul_(g, g, value: 1 - config.Beta2);
            if (decay[i]) parameters[i].mul_(1 - lr * config.WeightDecay);
            var denominator = v[i].sqrt().div_(sqrtCorrection2).add_(config.Eps);
            parameters[i].addcdiv_(m[i], denominator, value: -lr / correction1);
        }
    }

    /// Divides the gradients by `unscale` (the fp16 loss scale, else 1), then
    /// scales them down so their global L2 norm is at most maxNorm.
    /// Out: the norm before clipping, as a 0-d tensor still on the GPU.
    public Tensor UnscaleAndClip(double unscale, double maxNorm)
    {
        using var _ = no_grad();
        var grads = parameters.Select(p => p.grad).Where(g => g is not null).Select(g => g!).ToArray();
        if (unscale != 1)
            foreach (var g in grads) g.mul_(1 / unscale);
        var norm = stack(grads.Select(g => g.norm()).ToArray()).norm();
        var coefficient = (norm + 1e-6).reciprocal().mul_(maxNorm).clamp_max_(1.0);
        foreach (var g in grads) g.mul_(coefficient);
        return norm;
    }
}

public static class Schedule
{
    /// Linear warmup to LearningRate, then a cosine down to MinLearningRate.
    public static double LearningRateAt(int step, Config c)
    {
        if (step < c.WarmupSteps) return c.LearningRate * (step + 1) / c.WarmupSteps;
        double progress = Math.Min(1.0, (double)(step - c.WarmupSteps) / Math.Max(1, c.MaxSteps - c.WarmupSteps));
        return c.MinLearningRate + (c.LearningRate - c.MinLearningRate) * 0.5 * (1 + Math.Cos(Math.PI * progress));
    }
}
