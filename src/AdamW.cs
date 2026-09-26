// AdamW.cs — the optimiser, the learning-rate schedule and gradient clipping, by hand.
//
// It works on the model's flat buffers (FlatParameters.cs): every parameter in
// one tensor, every gradient in another. Its own state, the two moments, is
// flat in the same layout. So a step is a few operations over the whole model
// at once, with no per-parameter loop:
//
//     flat.Parameters  [ decayed matrices ........ | biases, LayerNorm .. ]   the model's
//     flat.Gradients   [ same layout                                     ]   the model's
//     firstMoment      [ same layout ]  moving average of the gradients          (Adam's m)
//     secondMoment     [ same layout ]  moving average of the squared gradients  (Adam's v)
//
// A step is ClipGradientNorm (optional), then Step. ClipGradientNorm only
// works out the scale; Step applies it, updates, and leaves the gradients at
// 0, ready for the next backward. So no ZeroGradients is needed between steps.
using TorchSharp;
using static TorchSharp.torch;

namespace Gpt2Trainer;

public sealed class AdamW
{
    readonly FlatParameters flat;
    readonly Tensor firstMoment, secondMoment;
    readonly Config config;
    long stepCount;                        // steps taken, for the bias correction
    Tensor? clipScale;                     // from ClipGradientNorm, used by the next Step

    /// The moments start at 0, as AdamW defines them, in the model's layout and
    /// on its device; they live as long as the optimiser, not the caller's
    /// dispose scope.
    public AdamW(FlatParameters flat, Config config)
    {
        this.flat = flat;
        this.config = config;
        firstMoment = zeros_like(flat.Parameters).DetachFromDisposeScope();
        secondMoment = zeros_like(flat.Parameters).DetachFromDisposeScope();
    }

    public void ZeroGradients() => flat.Gradients.zero_();    // 1 kernel

    // Clipping by the global norm, as torch.nn.utils.clip_grad_norm_:
    //
    //   norm = sqrt(sum of g² over every gradient in the model)
    //        = sqrt(‖G₁‖² + ‖G₂‖² + ‖bias‖² + …)     one number for the whole model
    //   clip = min(1, maxNorm / (norm + 1e-6))
    //   g    = g * clip                               in Step: one factor for every entry
    //
    // E.g. G = [[3, 0, 4], [0, 12, 0]] has
    // norm sqrt(9 + 16 + 144) = 13; with maxNorm 1.3, clip = 0.1 and
    // G becomes [[0.3, 0, 0.4], [0, 1.2, 0]], norm 1.3. Below maxNorm, clip = 1:
    // nothing changes. The 1e-6 keeps an all-zero gradient from dividing by 0.
    //
    /// Out: the norm before clipping, as a 0-d tensor still on the GPU.
    public Tensor ClipGradientNorm(double maxNorm)
    {
        using var _ = no_grad();
        var norm = flat.Gradients.norm();                     // the padding is 0: it adds nothing
        clipScale = (norm + 1e-6).reciprocal_().mul_(maxNorm).clamp_max_(1.0);
        return norm;
    }

    // AdamW (Loshchilov & Hutter, "Decoupled Weight Decay Regularization"),
    // step t, with lr the learning rate and wd the weight decay:
    //
    //   g  = g * clip                              (if ClipGradientNorm ran)
    //   p  = p - lr wd p                           decoupled weight decay: the W in AdamW
    //   m  = b1 m + (1-b1) g                       first moment
    //   v  = b2 v + (1-b2) g²                      second moment
    //   m̂  = m / (1-b1ᵗ)     v̂ = v / (1-b2ᵗ)       bias correction
    //   p  = p - lr m̂ / (sqrt(v̂) + eps)
    //   g  = 0
    //
    // Each line below is one of these, in this order and with the same
    // it bit for bit, not only up to rounding (AdamWTests checks this).
    // sqrt(v̂) is computed as sqrt(v) / sqrt(1-b2ᵗ), and lr m̂ as (lr / (1-b1ᵗ)) m,
    public void Step(double learningRate)
    {
        using var _ = no_grad();
        stepCount++;
        double biasCorrection1 = 1 - Math.Pow(config.Beta1, stepCount);
        double biasCorrection2 = 1 - Math.Pow(config.Beta2, stepCount);
        double weightDecayFactor = 1 - learningRate * config.WeightDecay;
        double stepSize = learningRate / biasCorrection1;

        var gradients = flat.Gradients;
        if (clipScale is not null) gradients.mul_(clipScale);                                    // g = g * clip
        if (flat.DecayedCount > 0)
            using (var decayed = flat.Parameters.narrow(0, 0, flat.DecayedCount))
                decayed.mul_(weightDecayFactor);                                                 // p = p - lr wd p
        firstMoment.mul_(config.Beta1).add_(gradients, alpha: 1 - config.Beta1);                 // m
        secondMoment.mul_(config.Beta2).addcmul_(gradients, gradients, value: 1 - config.Beta2); // v
        using var denominator = secondMoment.sqrt().div_(Math.Sqrt(biasCorrection2)).add_(config.Eps); // sqrt(v̂) + eps
        flat.Parameters.addcdiv_(firstMoment, denominator, value: -stepSize);                    // p = p - lr m̂ / (...)
        gradients.zero_();                                                                       // g = 0
        clipScale = null;
    }
}

/// The learning rate at each step: a linear warmup up to LearningRate (the peak),
/// then a cosine curve down to MinLearningRate, then flat at the minimum.
///
///     lr
///     peak |      ____
///          |     /    ‾‾--__
///          |    /           ‾-_
///          |   /               ‾-_
///          |  /                   ‾--__
///      min |_/                         ‾‾‾‾‾‾‾‾‾‾
///          +-----|-------------------------|---------> step
///          0   WarmupSteps              MaxSteps
///
/// Why warmup: at the start the weights are random and Adam's m and v are still
/// unreliable (few steps averaged), so big steps could throw the model off.
/// Starting small and growing lets it settle first.
/// Why cosine down: big steps learn fast early on; small steps near the end let
/// the weights settle into a good spot instead of bouncing around it.
///
/// With the defaults of traingpt2cs.py (peak 6e-4, min 6e-5, 300 warmup steps,
/// 3000 max steps): step 0 -> 2e-6, step 299 -> 6e-4 (peak), step 1650 -> 3.3e-4
/// (halfway down the cosine), step 3000 and after -> 6e-5.
public static class LearningRateSchedule
{
    /// The learning rate for `step` (counting from 0).
    public static double At(int step, Config config)
    {
        // 1. Warmup, steps 0 .. WarmupSteps-1: a straight line up to the peak.
        //    step + 1, not step, so the first step already moves a little (lr is never 0):
        //    step 0 gets peak / WarmupSteps, the last warmup step gets the full peak.
        if (step < config.WarmupSteps)
            return config.LearningRate * (step + 1) / config.WarmupSteps;

        // 2. After warmup: how far along the cosine this step is, from 0 (just after
        //    warmup) to 1 (MaxSteps). Min(1.0, ...) keeps it at 1 past MaxSteps, so the
        //    rate stays at the minimum. Max(1, ...) avoids dividing by 0 if MaxSteps is
        //    not above WarmupSteps.
        double progress = Math.Min(1.0, (double)(step - config.WarmupSteps) / Math.Max(1, config.MaxSteps - config.WarmupSteps));

        // 3. The cosine: 0.5 * (1 + cos(pi * progress)) goes smoothly from 1 (progress 0)
        //    down to 0 (progress 1), slow at both ends and fastest in the middle. It picks
        //    how much of the gap between peak and minimum is left:
        //        progress 0   -> 1   -> the peak
        //        progress 0.5 -> 0.5 -> halfway: (peak + min) / 2
        //        progress 1   -> 0   -> the minimum
        double fractionLeft = 0.5 * (1 + Math.Cos(Math.PI * progress));
        return config.MinLearningRate + (config.LearningRate - config.MinLearningRate) * fractionLeft;
    }
}
