// FusedAdamWTests.cs — AdamW's step as one CUDA kernel (FusedAdamW.cs, cuda/adamw.cu).
//
//     dotnet test Gpt2Trainer.sln -p:TorchBackend=cuda-linux     (on a GPU: Colab; cuda-windows on Windows)
//
// The kernel does in one pass what AdamW.Step does in 11 tensor operations, and
// must give the same numbers, bit for bit. So each test trains the same
// original weights twice, on the same gradients: once with FusedAdamW and once
// with a reference, and the two must end up equal. The reference is
// TorchSharp's built-in torch.optim.AdamW (PyTorch's own), as in AdamWTests, or
// our AdamW where the built-in works differently (weight decay by shape,
// clipping on the flat buffer).
//
// Everything runs on the GPU: the kernel is CUDA only, and the reference must
// be rounded by LibTorch's CUDA kernels, not its CPU ones. Without a GPU the
// tests show as skipped ([CudaFact], CudaFact.cs), except the one that checks
// the CPU is refused.
using Gpt2Trainer;
using System.Diagnostics;
using TorchSharp;
using Xunit;
using Xunit.Abstractions;
using static TorchSharp.torch;
using static Gpt2Trainer.Tests.AdamWTests;             // SimulateGradients

namespace Gpt2Trainer.Tests;

[Collection("GPU")]      // never at the same time as AdamWTests: the two benchmarks share the GPU
public class FusedAdamWTests
{
    readonly ITestOutputHelper output;                     // where the benchmark prints its timings
    public FusedAdamWTests(ITestOutputHelper output) => this.output = output;

    /// A tensor's values, copied from the GPU into an array, row by row.
    static float[] Values(Tensor tensor) => tensor.detach().cpu().data<float>().ToArray();

    // ── what it runs on ────────────────────────────────────────────────────

    [Fact]
    public void RefusesParametersOnTheCpu()
    {
        // The kernel runs on a CUDA GPU only. Parameters on the CPU are refused when the
        // optimiser is made, with a message that says what to use instead, not later by a
        // launch that fails. This one runs without a GPU too.

        // A weight on the CPU, in flat buffers on the CPU.
        var weight = nn.Parameter(tensor(new[] { 1f, 2f,
                                                 3f, 4f }, new long[] { 2, 2 }));
        var flat = new FlatParameters(new[] { weight });

        // Check: FusedAdamW refuses it, and points to AdamW.
        var error = Assert.Throws<ArgumentException>(() => new FusedAdamW(flat, new Config()));
        Assert.Contains("use AdamW", error.Message);
    }

    // ── the update ─────────────────────────────────────────────────────────

    [CudaFact]
    public void OneStepMatchesTheBuiltInAdamW()
    {
        // The weight to train with FusedAdamW, and expectedWeight, trained by the built-in
        // one. Both on the GPU.
        var config = new Config();
        const double lr = 1e-2;
        float[] originalWeight = { 0.5f, -1.0f,      // row 0
                                   2.0f,  0.0f };    // row 1
        var weight = nn.Parameter(tensor(originalWeight, new long[] { 2, 2 }, device: CUDA));
        var optimizer = new FusedAdamW(new FlatParameters(new[] { weight }), config);

        // expectedWeight: starts equal to the original weight, and TorchSharp's built-in
        // AdamW (PyTorch's own) trains it with the same settings as ours. It is what
        // ours must turn weight into.
        var expectedWeight = nn.Parameter(tensor(originalWeight, new long[] { 2, 2 }, device: CUDA));
        var torchAdamW = optim.AdamW(new[] { expectedWeight }, lr: lr, beta1: config.Beta1, beta2: config.Beta2,
                                     eps: config.Eps, weight_decay: config.WeightDecay);

        // Train both one step, with the same gradient.
        float[] rawGradient = { 0.1f, -0.2f, 0.3f, 0.05f };
        SimulateGradients((weight, rawGradient), (expectedWeight, rawGradient));
        optimizer.Step(lr);
        torchAdamW.step();

        // Check: ours equals the built-in, bit for bit. weight is a view into the flat
        // buffer the kernel wrote, so this also checks the kernel wrote where the weight is.
        Assert.Equal(Values(expectedWeight), Values(weight));
    }

    [CudaFact]
    public void LargeRandomWeightMatchesTheBuiltInAdamW()
    {
        // As in AdamWTests: a big weight (256 x 256 random values) and 100 steps of random
        // gradients, big enough to see rounding. Every 7th step's gradient is tiny (x 1e-6),
        // so eps matters too. A kernel that rounded any one operation differently (merged a
        // multiply and an add into one FMA where LibTorch keeps them apart, say) would be
        // off in the last bit of some of the values. 65,536 values: 64 blocks of 256
        // threads, each thread 4 values.

        // The weight to train with FusedAdamW, starting from random values.
        var config = new Config();
        const double lr = 1e-3;
        manual_seed(0);                                        // the same random numbers on every run
        var originalWeight = randn(256, 256).to(CUDA);
        var weight = nn.Parameter(originalWeight.clone());
        var optimizer = new FusedAdamW(new FlatParameters(new[] { weight }), config);

        // expectedWeight: starts equal to the original weight, and the built-in AdamW
        // trains it with the same settings as ours.
        var expectedWeight = nn.Parameter(originalWeight.clone());
        var torchAdamW = optim.AdamW(new[] { expectedWeight }, lr: lr, beta1: config.Beta1, beta2: config.Beta2,
                                     eps: config.Eps, weight_decay: config.WeightDecay);

        // Train both 100 steps, with the same random gradient each step.
        for (int t = 1; t <= 100; t++)
        {
            bool tiny = t % 7 == 1;                            // steps 1, 8, 15, ...
            float[] rawGradient = (randn(256, 256) * (tiny ? 1e-6 : 1.0)).data<float>().ToArray();
            torchAdamW.zero_grad();                            // ours sets its gradient to 0 in Step
            SimulateGradients((weight, rawGradient), (expectedWeight, rawGradient));
            optimizer.Step(lr);
            torchAdamW.step();
        }

        // Check: ours equals the built-in, bit for bit, not only close.
        Assert.Equal(Values(expectedWeight), Values(weight));
    }

    // ── weight decay ───────────────────────────────────────────────────────

    [CudaFact]
    public void WeightDecayShrinksMatricesButNotBiases()
    {
        // With every gradient 0, the Adam part of the step is 0, so the only change left is
        // weight decay. In the flat buffer the matrices come first, in [0, DecayedCount),
        // and the kernel decays exactly those values: a weight (2-D) is multiplied by
        // (1 - lr * weightDecay), a bias (1-D) stays as it is. Each has a 64-value segment:
        //
        //     Parameters  [ weight: 4 values, 60 padding | bias: 2 values, 62 padding ]
        //                   0                             64 = DecayedCount            128

        // A weight and a bias to train, and FusedAdamW to train them.
        const double lr = 0.5, weightDecay = 0.1;
        float[] originalWeight = { 2f, -4f,          // row 0
                                   6f,  8f };        // row 1
        float[] originalBias = { 2f, -4f };
        var weight = nn.Parameter(tensor(originalWeight, new long[] { 2, 2 }, device: CUDA));
        var bias = nn.Parameter(tensor(originalBias, new long[] { 2 }, device: CUDA));
        var optimizer = new FusedAdamW(new FlatParameters(new[] { weight, bias }), new Config { WeightDecay = weightDecay });

        // Train one step. No SimulateGradients: every gradient is still 0.
        optimizer.Step(lr);

        // Check: the weight shrank by (1 - lr * weightDecay) = 0.95; the bias did not move,
        // exactly: its Adam part is lr * 0 / (0 + eps) = 0, so it gets p + 0 = p.
        double shrink = 1 - lr * weightDecay;
        double[] expectedWeight = originalWeight.Select(value => value * shrink).ToArray();
        Assert.Equal(expectedWeight, Values(weight).Select(value => (double)value),
                     (expected, actual) => Math.Abs(expected - actual) < 1e-6);
        Assert.Equal(originalBias, Values(bias));
    }

    // ── the whole model, with clipping ─────────────────────────────────────

    [CudaFact]
    public void WholeModelMatchesAdamWWithClipping()
    {
        // GPT-2's full set of parameter tensors (a tiny GPT-2: 2 layers, 16 wide), in two
        // copies, each in flat buffers of its own: FusedAdamW trains one, our AdamW the
        // other. Everything the kernel does is in play: 28 tensors with padding between
        // them, the border between the decayed matrices and the rest, and clipping, whose
        // factor the kernel works out itself from the norm. The reference is our AdamW,
        // not the built-in: the built-in would decay the 1-D tensors too.
        //
        // Odd steps get big random gradients (norm about 90, clipped down to 1), even steps
        // small ones (norm about 0.09, left alone), so both sides of the clip are checked.

        // A tiny GPT-2 on the GPU; FusedAdamW over its flat buffers.
        var config = new Config
        {
            NLayer = 2, NHead = 2, NEmbd = 16, SequenceLength = 8, VocabSize = 50, PaddedVocabSize = 64,
        };
        const double lr = 1e-2, maxNorm = 1.0;
        manual_seed(0);                                        // the same random numbers on every run
        var model = new Gpt2(config, CUDA);                    // its parameters are already flat: model.Flat
        var parameters = model.parameters().ToArray();
        var optimizer = new FusedAdamW(model.Flat, config);

        // expectedParameters: copies of the same values, moved into flat buffers of their
        // own, trained by our AdamW. The same list in the same order, so FlatParameters
        // lays them out the same way: the two buffers can be compared value by value.
        var expectedParameters = parameters.Select(parameter => nn.Parameter(parameter.detach().clone())).ToArray();
        var expectedFlat = new FlatParameters(expectedParameters);
        var expectedOptimizer = new AdamW(expectedFlat, config);

        // Train both 20 steps, as the training loop does: gradients, clip, step.
        for (int t = 1; t <= 20; t++)
        {
            // 1. Every parameter gets a new random gradient, the same for ours and its
            //    expected one: big on odd steps, 1000 times smaller on even ones.
            bool big = t % 2 == 1;
            for (int i = 0; i < parameters.Length; i++)
            {
                float[] rawGradient = (randn(parameters[i].shape) * (big ? 1.0 : 1e-3)).data<float>().ToArray();
                SimulateGradients((parameters[i], rawGradient), (expectedParameters[i], rawGradient));
            }

            // 2. Clip, then step. Ours only measures the norm here; its kernel works out
            //    the clip factor from it in Step. AdamW works it out here, with tensor operations.
            float norm = optimizer.ClipGradientNorm(maxNorm).item<float>();
            float expectedNorm = expectedOptimizer.ClipGradientNorm(maxNorm).item<float>();
            optimizer.Step(lr);
            expectedOptimizer.Step(lr);

            // 3. Check: both measured the same norm (the same gradients in the same layout),
            //    on the side of maxNorm this step is meant to be on...
            Assert.Equal(expectedNorm, norm);
            Assert.Equal(big, norm > maxNorm);

            // ...and the two whole parameter buffers are equal, bit for bit: every
            // parameter, and the padding between them (still 0 in both).
            Assert.Equal(Values(expectedFlat.Parameters), Values(model.Flat.Parameters));
        }
    }

    // ── the step consumes the gradients ────────────────────────────────────

    [CudaFact]
    public void StepLeavesTheGradientsAtZero()
    {
        // As AdamW.Step, the kernel sets every gradient back to 0 after using it (it
        // writes g = 0 along with p, m and v), so the next step starts clean. The
        // training loop relies on it.

        // A weight and a bias, and FusedAdamW to train them.
        var weight = nn.Parameter(tensor(new[] { 1f, 2f,
                                                 3f, 4f }, new long[] { 2, 2 }, device: CUDA));
        var bias = nn.Parameter(tensor(new[] { 1f, 2f }, new long[] { 2 }, device: CUDA));
        var optimizer = new FusedAdamW(new FlatParameters(new[] { weight, bias }), new Config());

        // Train one step, as the training loop does: gradients, clip, step.
        SimulateGradients((weight, new[] { 1f, 2f, 3f, 4f }), (bias, new[] { 5f, 6f }));
        optimizer.ClipGradientNorm(maxNorm: 1.0);
        optimizer.Step(learningRate: 0.1);

        // Check: every gradient is 0 again.
        Assert.All(Values(weight.grad!), gradient => Assert.Equal(0f, gradient));
        Assert.All(Values(bias.grad!), gradient => Assert.Equal(0f, gradient));
    }

    // ── benchmark ──────────────────────────────────────────────────────────

    /// How long one optimiser step takes on the GPU: FusedAdamW against AdamW, the
    /// 11-operation step it replaces. The same parameters as the benchmark in
    /// AdamWTests (GPT-2's 148 tensors, narrowed to 128 wide: ~9M values), so the
    /// numbers line up with it and with adamw-benchmark-torch.py.
    ///
    /// Timed two ways: Step alone, as AdamWTests times it; and ClipGradientNorm +
    /// Step, the optimiser's whole work in a training step. Fusing saves on the clip
    /// too: AdamW spends 4 more small kernels on the clip factor, FusedAdamW none.
    ///
    /// Prints the timings and does not fail on them (they vary from GPU to GPU and
    /// run to run). To see them, on a GPU:
    ///     dotnet test Gpt2Trainer.sln -p:TorchBackend=cuda-linux --filter Category=AdamW-Benchmark --logger "console;verbosity=detailed"
    [CudaFact]
    [Trait("Category", "AdamW-Benchmark")]
    public void BenchmarkFusedAgainstTheFlatAdamW()
    {
        const int warmupSteps = 3, timedSteps = 20;
        const double lr = 1e-3, maxNorm = 1.0;

        // The parameters to train: GPT-2's real set of 148 tensors, narrow, as in
        // AdamWTests' benchmark. Built by Gpt2, so FusedAdamW gets them already in its
        // flat buffers.
        var config = new Config
        {
            NLayer = 12, NHead = 4, NEmbd = 128, SequenceLength = 1024, VocabSize = 50257, PaddedVocabSize = 50304,
        };
        manual_seed(0);                                        // the same random numbers on every run
        var model = new Gpt2(config, CUDA);
        var parameters = model.parameters().ToArray();
        var fused = new FusedAdamW(model.Flat, config);
        var originalFirstParameter = parameters[0].detach().clone();   // to check at the end that training happened

        // adamWParameters: the same values again, in flat buffers of their own, for AdamW.
        var adamWParameters = parameters.Select(parameter => nn.Parameter(parameter.detach().clone())).ToArray();
        var adamW = new AdamW(new FlatParameters(adamWParameters), config);

        // One random gradient per parameter, made once and reused every step, so making
        // gradients costs nothing extra inside the loop.
        var rawGradients = parameters.Select(parameter => randn_like(parameter)).ToArray();

        // Gradients the usual way, by backward() on a made-up loss (see SimulateGradients):
        // sum over the parameters of sum(p * g), whose gradient for each p is its g.
        void Backward(Tensor[] trained)
        {
            var loss = trained.Select((parameter, i) => (parameter * rawGradients[i]).sum()).Aggregate((a, b) => a + b);
            loss.backward();
        }

        // Waits until the GPU has finished all queued work: operations run later than the
        // call that queues them, so the clock must not stop before they end. Reading a
        // number back to the CPU makes it wait.
        void WaitForDevice() => _ = parameters[0].sum().item<float>();

        // Times `step` over timedSteps steps, after warmupSteps untimed ones (the first
        // steps allocate memory and warm caches, and the first FusedAdamW launch loads the
        // kernel). Only the optimiser work is timed: the backward() before each step is
        // outside the clock.
        double MillisecondsPerStep(Tensor[] trained, Action step)
        {
            var clock = new Stopwatch();
            for (int t = 1; t <= warmupSteps + timedSteps; t++)
            {
                Backward(trained);
                WaitForDevice();
                if (t > warmupSteps) clock.Start();
                step();
                WaitForDevice();
                clock.Stop();
            }
            return clock.Elapsed.TotalMilliseconds / timedSteps;
        }

        // Step alone; then clip + step, the norm disposed by the caller as soon as it is
        // used (FusedAdamW keeps its own handle to it until Step).
        double adamWStep = MillisecondsPerStep(adamWParameters, () => adamW.Step(lr));
        double fusedStep = MillisecondsPerStep(parameters, () => fused.Step(lr));
        double adamWClipStep = MillisecondsPerStep(adamWParameters, () => { using var norm = adamW.ClipGradientNorm(maxNorm); adamW.Step(lr); });
        double fusedClipStep = MillisecondsPerStep(parameters, () => { using var norm = fused.ClipGradientNorm(maxNorm); fused.Step(lr); });

        output.WriteLine($"device cuda, {parameters.Length} parameter tensors, " +
                         $"{parameters.Sum(parameter => parameter.numel()) / 1e6:F1}M values, {timedSteps} timed steps");
        output.WriteLine($"ms per step        step   clip + step");
        output.WriteLine($"AdamW:         {adamWStep,8:F2}   {adamWClipStep,11:F2}");
        output.WriteLine($"FusedAdamW:    {fusedStep,8:F2}   {fusedClipStep,11:F2}");
        output.WriteLine($"fused is {adamWStep / fusedStep:F2}x the speed of AdamW (step), {adamWClipStep / fusedClipStep:F2}x (clip + step)");

        // Check only that both really trained: the first parameter moved away from its
        // original values in both (timing an optimiser that did nothing would mean nothing).
        // The speed itself is reported, not asserted.
        Assert.True((parameters[0].detach() - originalFirstParameter).abs().max().item<float>() > 0, "FusedAdamW did not train");
        Assert.True((adamWParameters[0].detach() - originalFirstParameter).abs().max().item<float>() > 0, "AdamW did not train");
    }
}
