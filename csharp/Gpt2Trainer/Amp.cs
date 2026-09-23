// Amp.cs — mixed precision by hand; TorchSharp has no autocast.
//
// The same rule torch.autocast uses: the weights stay fp32 (the optimiser
// updates those), and each matmul casts its input and weight to the compute
// dtype on the way in. Autograd sends the gradients back through the casts,
// so they land on the fp32 weights as fp32. LayerNorm, the residual stream,
// softmax and the loss stay fp32. bf16 or fp16 also lets
// scaled_dot_product_attention use its flash kernel.
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace Gpt2Trainer;

public static class Amp
{
    public static ScalarType Dtype { get; private set; } = ScalarType.Float32;

    public static void Use(string precision) => Dtype = precision switch
    {
        "bf16" => ScalarType.BFloat16,
        "fp16" => ScalarType.Float16,
        "fp32" => ScalarType.Float32,
        _ => throw new ArgumentException($"precision is bf16, fp16 or fp32, not '{precision}'"),
    };

    public static Tensor Cast(Tensor x) => x.dtype == Dtype ? x : x.to(Dtype);

    /// x @ weightᵀ + bias in the compute dtype.
    public static Tensor Linear(Tensor x, TorchSharp.Modules.Linear layer) =>
        F.linear(Cast(x), Cast(layer.weight!), layer.bias is null ? null : Cast(layer.bias));
}

/// fp16's range is small, so the loss is multiplied by Scale before
/// backward() and the gradients divided by it after. When the gradients
/// overflow the step is skipped and Scale halved; after GrowthInterval clean
/// steps it doubles. Same rule as torch.amp.GradScaler. bf16 and fp32 don't
/// need it: Enabled is false and Scale stays 1.
public sealed class LossScaler
{
    public bool Enabled { get; }
    public double Scale { get; set; }
    public int GrowthInterval { get; } = 2000;
    int cleanSteps;

    public LossScaler(bool enabled) { Enabled = enabled; Scale = enabled ? 65536.0 : 1.0; }

    /// Whether this step's gradients were finite; updates Scale.
    public bool Update(bool finite)
    {
        if (!Enabled) return true;
        if (!finite) { Scale /= 2; cleanSteps = 0; return false; }
        if (++cleanSteps % GrowthInterval == 0) Scale *= 2;
        return true;
    }
}
