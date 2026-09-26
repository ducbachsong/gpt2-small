// FlatParameters.cs — the model's parameters and gradients, each in one big flat buffer.
//
// Gpt2 creates this at the end of its constructor, once its layers exist and
// are on their device. Every parameter tensor stays where the layers expect
// it, but its memory is moved into one 1-D buffer, and its gradient into a
// second one:
//
//     Parameters  [ decayed: every matrix ......... | not decayed: 1-D tensors ]
//     Gradients   [ same layout                                                ]
//
// The layers keep using their own tensors as before; the optimiser uses the
// two big buffers, and works on the whole model at once with no per-parameter
// loop (AdamW.cs). Same memory, seen two ways.
//
// Decayed parameters (dim >= 2) come first, so weight decay is one narrow() of
// [0, DecayedCount). Each segment starts on a multiple of SegmentAlignment
// elements; the padding is 0 and stays 0. Do not move the model (.to()) after
// this: the parameters would stop being views into the buffer.
using TorchSharp;
using static TorchSharp.torch;

namespace Gpt2Trainer;

public sealed class FlatParameters
{
    const long SegmentAlignment = 64;      // 256 bytes of fp32: aligned, vectorised loads

    /// Every parameter of the model, in one 1-D tensor.
    public Tensor Parameters { get; }
    /// Every gradient, same layout: backward() adds straight into it.
    public Tensor Gradients { get; }
    /// Elements [0, DecayedCount) are the matrices, which get weight decay.
    public long DecayedCount { get; }

    /// Moves every parameter into the flat buffers (see the top of this file).
    /// Weight decay on the matrices and embeddings only (dim >= 2), not on
    /// biases or LayerNorm, as in main's traingpt2.py.
    public FlatParameters(IEnumerable<Tensor> parameters)
    {
        // 1. Split the parameters by whether they get weight decay, and put the
        //    decayed ones first. The model lists its parameters mixed, layer by
        //    layer (ln_1.weight, ln_1.bias, c_attn.weight, c_attn.bias, ...);
        //    in the buffer they become two blocks:
        //
        //        [ decayed: every matrix            | not decayed: every 1-D tensor ]
        //          wte, wpe, c_attn/c_proj/c_fc       biases, LayerNorm weights
        //          weights (50 in GPT-2 small)        and biases (98)
        //
        //    So the optimiser decays one block, [0, DecayedCount), with a single
        //    mul_, instead of looping over 50 matrices or keeping a mask. Where a
        //    parameter sits in the buffer changes nothing else: the model still
        //    sees the same values.
        var decayedParameters = parameters.Where(p => p.dim() >= 2).ToArray();
        var notDecayedParameters = parameters.Where(p => p.dim() < 2).ToArray();
        var parametersInBufferOrder = decayedParameters.Concat(notDecayedParameters).ToArray();
        if (parametersInBufferOrder.Length == 0) throw new ArgumentException("no parameters");

        // 2. The layout: where each parameter's segment starts. Each segment is
        //    its size rounded up to a multiple of SegmentAlignment, so the next
        //    one starts aligned; the gap is padding. DecayedCount is where the
        //    decayed block ends and the not-decayed block begins.
        var segmentStarts = new long[parametersInBufferOrder.Length];
        long totalLength = 0;
        for (int i = 0; i < parametersInBufferOrder.Length; i++)
        {
            if (i == decayedParameters.Length) DecayedCount = totalLength;
            segmentStarts[i] = totalLength;
            totalLength += RoundUpToMultiple(parametersInBufferOrder[i].numel(), SegmentAlignment);
        }
        if (notDecayedParameters.Length == 0) DecayedCount = totalLength;

        // 3. The two flat buffers, all zeros, so the padding starts at 0. They
        //    live as long as the model, not the caller's dispose scope.
        //    The buffers must be on the same device (GPU or CPU) and hold the same
        //    number type (float32 here) as the parameters: step 4 makes each
        //    parameter a view into Parameters, and a view always has its
        //    buffer's device and dtype. The model is moved to one device as a
        //    whole and is all float32, so every parameter has the same device
        //    and dtype as the first one (not checked).
        var device = parametersInBufferOrder[0].device;
        var dtype = parametersInBufferOrder[0].dtype;
        using var _ = no_grad();
        Parameters = zeros(totalLength, dtype: dtype, device: device).DetachFromDisposeScope();
        Gradients = zeros(totalLength, dtype: dtype, device: device).DetachFromDisposeScope();

        // 4. Move each parameter into its segment: copy its values in, then make
        //    the parameter itself a view of that segment (set_), so the layers
        //    and the optimiser share the same memory. Its gradient becomes the
        //    same segment of Gradients: backward() adds into it in place, so
        //    the gradients land in the flat buffer with no copying.
        for (int i = 0; i < parametersInBufferOrder.Length; i++)
        {
            var parameter = parametersInBufferOrder[i];
            long count = parameter.numel();
            using var slot = Parameters.narrow(0, segmentStarts[i], count).view(parameter.shape);
            slot.copy_(parameter);
            parameter.set_(slot);                            // the parameter now lives inside Parameters
            parameter.grad = Gradients.narrow(0, segmentStarts[i], count).view(parameter.shape);
        }
    }

    /// value rounded up to the next multiple of `multiple`: (100, 64) -> 128,
    /// (64, 64) -> 64, (10, 64) -> 64. Integer division drops the remainder, so
    /// adding multiple - 1 first turns its round-down into a round-up; a value
    /// that already is a multiple does not reach the next one and stays.
    static long RoundUpToMultiple(long value, long multiple) =>
        (value + multiple - 1) / multiple * multiple;
}
