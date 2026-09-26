// AdamWTests.cs — the hand-written AdamW, clipping and learning-rate schedule.
//
//     dotnet test Gpt2Trainer.sln        (from the repo root)
//
// Each parameter update is checked against TorchSharp's built-in
// torch.optim.AdamW (PyTorch's own AdamW): a second parameter with the same
// original weight is trained with it on the same gradients, and ours must end up equal.
// Gradients come from backward() on a loss whose gradient is known:
// d/dp sum(p * g) = g, so each test picks the gradient.
using Gpt2Trainer;
using TorchSharp;
using TorchSharp.Modules;
using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace Gpt2Trainer.Tests;

public class AdamWTests
{
    const double Tolerance = 1e-5;

    readonly ITestOutputHelper output;                     // where the benchmark prints its timings
    public AdamWTests(ITestOutputHelper output) => this.output = output;

    /// Simulates what training does to make gradients: builds a loss and runs
    /// backward() on it, so each parameter p ends up with the gradient g we chose
    /// for it. There is no model here, so the loss is made up: sum(p * g).
    ///
    /// The loss is picked for its gradient: d/dp sum(p * g) = g, so backward()
    /// writes exactly g into p.grad. E.g. a = [[a1, a2], [a3, a4]], g = [1, 2, 3, 4]:
    ///
    ///     loss = 1·a1 + 2·a2 + 3·a3 + 4·a4     ->   a.grad = [[1, 2], [3, 4]]
    ///
    /// One term per parameter, added into one scalar loss, so one backward() sets them all.
    /// Why not assign p.grad = g: p.grad is a view into flat.Gradients, and
    /// backward() adds into that view, so g lands in the buffer AdamW reads.
    /// A new tensor assigned to p.grad would not. Like any backward(), it adds:
    /// the gradients must be 0 before (they are at the start, and after each Step).
    static void SimulateGradients(params (Tensor parameter, float[] gradient)[] pairs)
    {
        var loss = pairs.Select(pair =>
            (pair.parameter * tensor(pair.gradient, pair.parameter.shape, device: pair.parameter.device)).sum())
            .Aggregate((a, b) => a + b);
        loss.backward();
    }

    // ── the update ─────────────────────────────────────────────────────────

    [Fact]
    public void OneStepMatchesTheBuiltInAdamW()
    {
        // The weight to train with our AdamW, and expectedWeight, trained by the built-in one.
        var config = new Config();
        const double lr = 1e-2;
        float[] originalWeight = { 0.5f, -1.0f,      // row 0
                                   2.0f,  0.0f };    // row 1
        var weight = nn.Parameter(tensor(originalWeight, new long[] { 2, 2 }));
        var optimizer = new AdamW(new FlatParameters(new[] { weight }), config);

        // expectedWeight: starts equal to the original weight, and TorchSharp's built-in
        // AdamW (PyTorch's own) trains it with the same settings as ours. It is what
        // ours must turn weight into.
        var expectedWeight = nn.Parameter(tensor(originalWeight, new long[] { 2, 2 }));
        var torchAdamW = optim.AdamW(new[] { expectedWeight }, lr: lr, beta1: config.Beta1, beta2: config.Beta2,
                                     eps: config.Eps, weight_decay: config.WeightDecay);

        // Train both one step, with the same gradient.
        float[] rawGradient = { 0.1f, -0.2f, 0.3f, 0.05f };
        SimulateGradients((weight, rawGradient), (expectedWeight, rawGradient));
        optimizer.Step(lr);
        torchAdamW.step();

        // Check: ours equals the built-in, bit for bit.
        Assert.Equal(expectedWeight.data<float>().ToArray(), weight.data<float>().ToArray());
    }

    [Fact]
    public void FirstStepMovesEachParameterByAboutTheLearningRate()
    {
        // At t = 1, m̂ = g and v̂ = g², so the Adam part of the step is lr * g / |g| = ±lr,
        // whatever the size of the gradient. With no weight decay that is the whole step.

        // The weight to train (all 1s), and AdamW to train it, with no weight decay.
        const double lr = 0.05;
        var weight = nn.Parameter(tensor(new[] { 1f, 1f,
                                                 1f, 1f }, new long[] { 2, 2 }));
        var optimizer = new AdamW(new FlatParameters(new[] { weight }), new Config { WeightDecay = 0 });

        // Train one step, with gradients from huge to tiny.
        SimulateGradients((weight, new[] { 1000f, 0.001f, -5f, -0.3f }));
        optimizer.Step(lr);

        // Check: each value moved by lr against its gradient's sign: 1 - 0.05 if g > 0, 1 + 0.05 if g < 0.
        float[] expectedWeight = { 0.95f, 0.95f, 1.05f, 1.05f };
        Assert.Equal(expectedWeight, weight.data<float>().ToArray(), (expected, actual) => Math.Abs(expected - actual) < 1e-4);
    }

    [Fact]
    public void ManyStepsMatchTheBuiltInAdamW()
    {
        // One weight, 10 steps, a new gradient each step (some of them 0). m and v carry
        // over from step to step, so ours must equal the built-in after every step.

        // The 2x2 weight to train with our AdamW, and expectedWeight, trained by the built-in one.
        // tensor() takes the values row by row, then the shape (2 rows, 2 columns).
        // { 0.8, -0.5, 1.2, 0.3 } is the original weight, before training: both weight and
        // expectedWeight begin from it (the same values are written in both calls below).
        // It must not be all 0s: weight decay shrinks the value itself (lr * wd * value),
        // so from 0 it would do nothing and the test could not see it.
        var config = new Config();
        const double lr = 3e-3;
        var weight = nn.Parameter(tensor(new float[]{ 0.8f, -0.5f, 1.2f,  0.3f }
                                        , new long[] { 2, 2 }));
        var optimizer = new AdamW(new FlatParameters(new[] { weight }), config);

        // expectedWeight: starts equal to the original weight, and TorchSharp's built-in
        // AdamW (PyTorch's own) trains it with the same settings as ours. It is what
        // ours must turn weight into.
        var expectedWeight = nn.Parameter(tensor(new float[]{ 0.8f, -0.5f, 1.2f,  0.3f }, new long[] { 2, 2 }));
        var torchAdamW = optim.AdamW(new[] { expectedWeight }, lr: lr, beta1: config.Beta1, beta2: config.Beta2,
                                     eps: config.Eps, weight_decay: config.WeightDecay);

        // Each entry's own gradient at each of the 10 steps: gradients[entry][t - 1].
        //     step t:             1      2      3     4      5     6     7      8     9     10
        float[][] gradients =
        {
            new[] {  0.3f,  -0.1f,  0.25f,  0.0f,  -0.4f,  0.2f,  0.1f, -0.05f,  0.5f,  -0.3f },   // entry 0 (row 0, col 0)
            new[] { -0.2f,   0.4f, -0.15f,  0.3f,   0.0f, -0.5f,  0.2f,  0.35f, -0.1f,   0.05f },  // entry 1 (row 0, col 1)
            new[] {  1.0f,   0.9f,  0.8f,   0.7f,   0.6f,  0.5f,  0.4f,  0.3f,   0.2f,   0.1f },   // entry 2 (row 1, col 0)
            new[] {  0.01f, -0.02f, 0.03f, -0.04f,  0.05f, 0.0f, -0.06f, 0.07f, -0.08f,  0.09f },  // entry 3 (row 1, col 1)
        };

        // Train both 10 steps, with the same gradients. After each, ours must equal the built-in.
        for (int t = 1; t <= 10; t++)
        {
            // This step's gradient for the whole weight: each entry's own number.
            float[] rawGradient = gradients.Select(entryGradients => entryGradients[t - 1]).ToArray();
            torchAdamW.zero_grad();                            // ours sets its gradient to 0 in Step
            SimulateGradients((weight, rawGradient), (expectedWeight, rawGradient));
            optimizer.Step(lr);
            torchAdamW.step();

            Assert.Equal(expectedWeight.data<float>().ToArray(), weight.data<float>().ToArray());   // bit for bit
        }
    }

    [Fact]
    public void LargeRandomWeightMatchesTheBuiltInAdamW()
    {
        // A big weight (256 x 256 random values) and 100 steps of random gradients: big
        // enough to see rounding. A rearranged but equivalent formula (eps folded into the
        // step size, say) differs from the built-in in the last bit of ~12% of the values,
        // so bit-for-bit equality here means ours does the same operations in the same order.
        // Every 7th step's gradient is tiny (x 1e-6): eps only matters when the gradient is
        // tiny, so it is tested too.

        // The weight to train with our AdamW, starting from random values.
        var config = new Config();
        const double lr = 1e-3;
        manual_seed(0);                                        // the same random numbers on every run
        var originalWeight = randn(256, 256);
        var weight = nn.Parameter(originalWeight.clone());
        var optimizer = new AdamW(new FlatParameters(new[] { weight }), config);

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
        Assert.Equal(expectedWeight.data<float>().ToArray(), weight.data<float>().ToArray());
    }

    // ── weight decay ───────────────────────────────────────────────────────

    [Fact]
    public void WeightDecayShrinksMatricesButNotBiases()
    {
        // With every gradient 0, the Adam part of the step is 0, so the only change left is
        // weight decay: a weight (2-D) is multiplied by (1 - lr * weightDecay); a bias (1-D)
        // gets no decay, so it stays as it is.

        // A weight and a bias to train, and our AdamW to train them.
        const double lr = 0.5, weightDecay = 0.1;
        float[] originalWeight = { 2f, -4f,          // row 0
                                   6f,  8f };        // row 1
        float[] originalBias = { 2f, -4f };
        var weight = nn.Parameter(tensor(originalWeight, new long[] { 2, 2 }));
        var bias = nn.Parameter(tensor(originalBias, new long[] { 2 }));
        var optimizer = new AdamW(new FlatParameters(new[] { weight, bias }), new Config { WeightDecay = weightDecay });

        // Train one step, with every gradient 0.
        SimulateGradients((weight, new float[4]), (bias, new float[2]));
        optimizer.Step(lr);

        // Check: the weight shrank by (1 - lr * weightDecay) = 0.95; the bias did not move.
        double shrink = 1 - lr * weightDecay;
        double[] expectedWeight = originalWeight.Select(value => value * shrink).ToArray();
        Assert.Equal(expectedWeight, weight.data<float>().ToArray().Select(value => (double)value),
                     (expected, actual) => Math.Abs(expected - actual) < 1e-6);
        Assert.Equal(originalBias, bias.data<float>().ToArray());
    }

    [Fact]
    public void BiasMatchesTheBuiltInAdamWWithoutDecay()
    {
        // The bias to train with our AdamW (weight decay on, but not for a bias), and
        // expectedBias, trained by the built-in one with no weight decay.
        var config = new Config { WeightDecay = 0.1 };
        const double lr = 5e-2;
        float[] originalBias = { 0.7f };
        var bias = nn.Parameter(tensor(originalBias, new long[] { 1 }));
        var optimizer = new AdamW(new FlatParameters(new[] { bias }), config);

        // expectedBias: starts equal to the original bias, and the built-in AdamW trains
        // it. The built-in decays everything it is given, so it gets weight_decay 0 here,
        // as ours gives a bias.
        var expectedBias = nn.Parameter(tensor(originalBias, new long[] { 1 }));
        var torchAdamW = optim.AdamW(new[] { expectedBias }, lr: lr, beta1: config.Beta1, beta2: config.Beta2,
                                     eps: config.Eps, weight_decay: 0);

        // Train both one step, with the same gradient.
        float[] rawGradient = { 0.2f };
        SimulateGradients((bias, rawGradient), (expectedBias, rawGradient));
        optimizer.Step(lr);
        torchAdamW.step();

        // Check: ours equals the built-in with no decay. Both went from 0.7 to about 0.69:
        // at t = 1, m̂ = g and v̂ = g², so the step is lr * g / |g| = 0.01 against the
        // gradient's sign (g = 0.2 > 0, so down). With decay it would lose 0.0007 more.
        Assert.Equal(expectedBias.data<float>().ToArray(), bias.data<float>().ToArray());
    }

    // ── what is left alone ─────────────────────────────────────────────────

    [Fact]
    public void AParameterTheLossDidNotUseOnlyDecays()
    {
        // A loss does not always use every parameter. In our AdamW every gradient is a view
        // into one flat buffer, so a parameter the loss did not use has a gradient of 0, not
        // "no gradient". Each step does two separate things to a weight:
        //     weight decay:  value -= lr * weightDecay * value   uses only the value, so it
        //                                                        still runs with a 0 gradient
        //     Adam part:     value -= lr * m̂ / (√v̂ + eps)       0 here: a 0 gradient keeps m at 0
        // So an unused weight still shrinks by weight decay, and nothing else. The same as
        // torch.optim.AdamW does with a zero gradient (it skips a parameter only when its
        // .grad is None, which ours never is).

        // Two weights to train, and our AdamW to train them.
        var config = new Config();                             // WeightDecay 0.1
        const double lr = 0.1;
        float[] originalUsedWeight = { 1f, 2f,
                                       3f, 4f };
        float[] originalUnusedWeight = { 5f, 6f,
                                         7f, 8f };
        var usedWeight = nn.Parameter(tensor(originalUsedWeight, new long[] { 2, 2 }));
        var unusedWeight = nn.Parameter(tensor(originalUnusedWeight, new long[] { 2, 2 }));
        var optimizer = new AdamW(new FlatParameters(new[] { usedWeight, unusedWeight }), config);

        // Train one step. The loss uses only usedWeight, so only it gets a gradient.
        SimulateGradients((usedWeight, new[] { 1f, 1f, 1f, 1f }));
        optimizer.Step(lr);

        // Check unusedWeight: weight decay only. value -= lr * weightDecay * value takes away
        // 0.1 * 0.1 = 1% of each value, the same as value x 0.99:
        //     5 - 0.05 = 4.95,  6 - 0.06 = 5.94,  7 - 0.07 = 6.93,  8 - 0.08 = 7.92
        double shrink = 1 - lr * config.WeightDecay;
        double[] expectedUnusedWeight = originalUnusedWeight.Select(value => value * shrink).ToArray();
        Assert.Equal(expectedUnusedWeight, unusedWeight.data<float>().ToArray().Select(value => (double)value),
                     (expected, actual) => Math.Abs(expected - actual) < 1e-6);

        // Check usedWeight: weight decay AND the Adam part. At t = 1 the Adam part moves each
        // value by lr = 0.1 against its gradient (g = 1 > 0, so down): value x 0.99 - 0.1.
        //     1 -> 0.89,  2 -> 1.88,  3 -> 2.87,  4 -> 3.86
        double[] expectedUsedWeight = originalUsedWeight.Select(value => value * shrink - lr).ToArray();
        Assert.Equal(expectedUsedWeight, usedWeight.data<float>().ToArray().Select(value => (double)value),
                     (expected, actual) => Math.Abs(expected - actual) < 1e-6);
    }

    [Fact]
    public void ZeroGradClearsEveryGradient()
    {
        // A weight and a bias, and our AdamW over them.
        var weight = nn.Parameter(tensor(new[] { 1f, 2f,
                                                 3f, 4f }, new long[] { 2, 2 }));
        var bias = nn.Parameter(tensor(new[] { 1f, 2f }, new long[] { 2 }));
        var optimizer = new AdamW(new FlatParameters(new[] { weight, bias }), new Config());

        // Give both a gradient, then clear them.
        SimulateGradients((weight, new[] { 1f, 2f, 3f, 4f }), (bias, new[] { 5f, 6f }));
        optimizer.ZeroGradients();

        // Check: every gradient is 0 again.
        Assert.All(weight.grad!.data<float>().ToArray(), gradient => Assert.Equal(0f, gradient));
        Assert.All(bias.grad!.data<float>().ToArray(), gradient => Assert.Equal(0f, gradient));
    }

    [Fact]
    public void GradientsAddUpOverBackwardCallsLikeTorch()
    {
        // Gradient accumulation relies on this: several backward() calls (one per
        // micro-batch) add their gradients together, and one step then uses the sum.
        // In a plain PyTorch parameter, backward() adds into .grad. Ours is different:
        // .grad is a view into our flat buffer. It must still add up the same way.

        // A bias whose gradient lives in our flat buffer. No AdamW needed: FlatParameters
        // is what turns its .grad into a view of the buffer.
        var bias = nn.Parameter(tensor(new[] { 0f, 0f }, new long[] { 2 }));
        _ = new FlatParameters(new[] { bias });

        // expectedBias: a plain PyTorch parameter, not in any flat buffer.
        var expectedBias = nn.Parameter(tensor(new[] { 0f, 0f }, new long[] { 2 }));

        // Two backward() calls on both, as for two micro-batches.
        SimulateGradients((bias, new[] { 1f, 2f }), (expectedBias, new[] { 1f, 2f }));
        SimulateGradients((bias, new[] { 10f, 20f }), (expectedBias, new[] { 10f, 20f }));

        // Check: ours equals PyTorch's, and both are the sum: 1 + 10 = 11 and 2 + 20 = 22.
        Assert.Equal(expectedBias.grad!.data<float>().ToArray(), bias.grad!.data<float>().ToArray());
        Assert.Equal(new[] { 11f, 22f }, bias.grad!.data<float>().ToArray());
    }

    // ── the flat buffers ───────────────────────────────────────────────────

    [Fact]
    public void ParametersKeepTheirValuesAndShapesAfterFlattening()
    {
        // FlatParameters moves every parameter's values into one flat buffer (weights
        // first, then biases). From outside, each parameter must look the same as before:
        // same shape, same values, and still a trainable leaf tensor.

        // A 2x3 weight and a bias.
        float[] originalWeight = { 1f, 2f, 3f,       // row 0
                                   4f, 5f, 6f };     // row 1
        float[] originalBias = { 7f, 8f };
        var weight = nn.Parameter(tensor(originalWeight, new long[] { 2, 3 }));
        var bias = nn.Parameter(tensor(originalBias, new long[] { 2 }));

        // Flatten them, bias first, so FlatParameters has to reorder them (weights first).
        _ = new FlatParameters(new[] { bias, weight });

        // Check: shapes and values are unchanged, and the weight can still be trained.
        Assert.Equal(new long[] { 2, 3 }, weight.shape);
        Assert.Equal(new long[] { 2 }, bias.shape);
        Assert.Equal(originalWeight, weight.data<float>().ToArray());
        Assert.Equal(originalBias, bias.data<float>().ToArray());
        Assert.True(weight.requires_grad && weight.is_leaf);
    }

    [Fact]
    public void MixedParametersMatchTorchSharpsBuiltInAdamW()
    {
        // Parameters of several shapes (1-D, 2-D, 3-D) in a mixed order. FlatParameters puts
        // each one at its own offset in the flat buffer; a mistake in the offsets would show
        // up here as a parameter that no longer matches the built-in.
        // No weight decay: the built-in would decay the 1-D ones too, and ours does not.

        // The parameters to train with our AdamW, starting from random values.
        var config = new Config { WeightDecay = 0 };
        const double lr = 1e-3;
        manual_seed(1);                                        // the same random numbers on every run
        long[][] shapes = { new long[] { 5 }, new long[] { 3, 4 }, new long[] { 7 }, new long[] { 2, 2, 3 } };
        var originalParameters = shapes.Select(shape => randn(shape)).ToArray();
        var parameters = originalParameters.Select(original => nn.Parameter(original.clone())).ToArray();
        var optimizer = new AdamW(new FlatParameters(parameters), config);

        // expectedParameters: start equal to the original ones, and the built-in AdamW
        // trains them with the same settings as ours.
        var expectedParameters = originalParameters.Select(original => nn.Parameter(original.clone())).ToArray();
        var torchAdamW = optim.AdamW(expectedParameters, lr: lr, beta1: config.Beta1, beta2: config.Beta2,
                                     eps: config.Eps, weight_decay: 0);

        // Train both 20 steps. Each step, every parameter gets a new random gradient,
        // the same for ours and for its expected one.
        for (int t = 1; t <= 20; t++)
        {
            torchAdamW.zero_grad();                            // ours sets its gradients to 0 in Step
            for (int i = 0; i < shapes.Length; i++)
            {
                float[] rawGradient = randn(shapes[i]).data<float>().ToArray();
                SimulateGradients((parameters[i], rawGradient), (expectedParameters[i], rawGradient));
            }
            optimizer.Step(lr);
            torchAdamW.step();
        }

        // Check: every parameter equals its expected one, bit for bit.
        for (int i = 0; i < shapes.Length; i++)
            Assert.Equal(expectedParameters[i].data<float>().ToArray(), parameters[i].data<float>().ToArray());
    }

    [Fact]
    public void WeightDecayFollowsTheShapeNotTheOrder()
    {
        // Who gets weight decay is decided by shape (2-D or more: yes; 1-D: no), not by
        // where a parameter sits in the list. So they are mixed: bias, weight, bias, weight.

        // Two weights and two biases to train, mixed, and our AdamW to train them.
        const double lr = 0.5, weightDecay = 0.1;
        var bias1 = nn.Parameter(tensor(new[] { 2f }, new long[] { 1 }));
        var weight1 = nn.Parameter(tensor(new[] { 2f, 4f }, new long[] { 1, 2 }));
        var bias2 = nn.Parameter(tensor(new[] { 2f, 4f, 6f }, new long[] { 3 }));
        var weight2 = nn.Parameter(tensor(new[] { 2f }, new long[] { 1, 1 }));    // 1x1 is still 2-D
        var optimizer = new AdamW(new FlatParameters(new[] { bias1, weight1, bias2, weight2 }), new Config { WeightDecay = weightDecay });

        // Train one step. No SimulateGradients: every gradient is still 0, so the only
        // change is weight decay.
        optimizer.Step(lr);

        // Check the weights: weight decay takes lr * weightDecay = 0.5 * 0.1 = 5% of each
        // value, so each is x 0.95. Compared to within 1e-6, not exactly: 0.95 cannot be
        // stored exactly in a float, so the last digits differ a little.
        //     weight1: 2 -> 1.9,  4 -> 3.8        weight2: 2 -> 1.9
        Assert.Equal(new[] { 1.9f, 3.8f }, weight1.data<float>().ToArray(), (expected, actual) => Math.Abs(expected - actual) < 1e-6);
        Assert.Equal(new[] { 1.9f }, weight2.data<float>().ToArray(), (expected, actual) => Math.Abs(expected - actual) < 1e-6);

        // Check the biases: they did not move at all, exactly. A bias gets no weight decay,
        // and its Adam part is lr * m̂ / (√v̂ + eps) = 0.5 * 0 / (0 + eps) = 0, because a 0
        // gradient keeps m at 0. lr only scales the step the gradient asks for; with no
        // gradient there is nothing to scale.
        Assert.Equal(new[] { 2f }, bias1.data<float>().ToArray());
        Assert.Equal(new[] { 2f, 4f, 6f }, bias2.data<float>().ToArray());
    }

    [Fact]
    public void BackwardThroughTheModelWritesIntoTheFlatGradients()
    {
        // The real case: gradients from backward() through the GPT-2 model, not from
        // SimulateGradients. Each parameter's .grad must be a view into the flat buffer,
        // and backward() must add into that view. If autograd replaced a .grad with a new
        // tensor instead, the flat buffer (what AdamW reads) would miss that gradient.
        // To check, the norm of all gradients is worked out twice: from the parameters'
        // own .grad tensors, and from the flat buffer. They must match. Two steps, so it
        // is also checked after the gradients were set back to 0.

        // A tiny GPT-2 (2 layers, 16 wide), and our AdamW over its parameters.
        var config = new Config
        {
            NLayer = 2, NHead = 2, NEmbd = 16, SequenceLength = 8, VocabSize = 50, PaddedVocabSize = 64,
        };
        manual_seed(0);                                        // the same random numbers on every run
        var model = new Gpt2(config);                          // its parameters are already flat: model.Flat
        var parameters = model.parameters().ToArray();
        var optimizer = new AdamW(model.Flat, config);
        var originalParameters = parameters.Select(parameter => parameter.detach().clone()).ToArray();

        // Train 2 steps, each on a random batch of 2 rows x 8 token ids.
        for (int t = 1; t <= 2; t++)
        {
            // 1. Clear every gradient, so this step starts from 0 (backward() adds).
            optimizer.ZeroGradients();

            // 2. A real forward and backward pass: random token ids in, loss out, then
            //    backward() works out every parameter's gradient through the whole model
            //    and adds it into that parameter's .grad, a view into the flat buffer.
            model.Loss(randint(0, config.VocabSize, new long[] { 2, 8 })).backward();

            // 3. The norm of all gradients, the parameters' way: each parameter's own .grad,
            //    the norm of each, then sqrt(norm1² + norm2² + ...), which is the norm of
            //    all of them as one long vector.
            double normFromParameters = Math.Sqrt(parameters.Sum(parameter => Math.Pow(parameter.grad!.norm().item<float>(), 2)));

            // 4. The same norm, AdamW's way: ClipGradientNorm reads the flat buffer, the one
            //    AdamW uses, and returns its norm. maxNorm 1e9 is far above any real norm, so
            //    it only measures and never clips.
            double normFromFlatBuffer = optimizer.ClipGradientNorm(maxNorm: 1e9).item<float>();

            // 5. Check: the model really made gradients (not all 0), and both norms agree.
            //    If any .grad had been a separate tensor, the flat buffer would miss it and
            //    its norm would come out smaller. Within 1e-5 of the norm, not exactly: the
            //    two sums add the same numbers in a different order, so the last digits differ.
            Assert.True(normFromParameters > 0);
            Assert.Equal(normFromParameters, normFromFlatBuffer, normFromParameters * 1e-5);

            // 6. The AdamW step: uses the gradients in the flat buffer to move every
            //    parameter, then sets the gradients back to 0.
            optimizer.Step(learningRate: 1e-3);
        }

        // Check: every parameter moved, so every gradient reached AdamW.
        for (int i = 0; i < parameters.Length; i++)
            Assert.True((parameters[i].detach() - originalParameters[i]).abs().max().item<float>() > 0,
                        $"parameter {i} did not move");
    }

    // ── clipping ───────────────────────────────────────────────────────────

    [Fact]
    public void ClipReturnsTheNormOfAllGradientsTogether()
    {
        // ClipGradientNorm returns the norm of all gradients together, as if they were one
        // long vector: the square root of every gradient squared, summed over all parameters.

        // A weight and a bias; their values do not matter here, only their gradients.
        var weight = nn.Parameter(tensor(new float[4], new long[] { 2, 2 }));
        var bias = nn.Parameter(tensor(new float[2], new long[] { 2 }));
        var optimizer = new AdamW(new FlatParameters(new[] { weight, bias }), new Config());

        // Give them gradients, and measure. maxNorm 1000 is far above the norm: nothing is clipped.
        SimulateGradients((weight, new[] { 1f, 2f, 3f, 4f }), (bias, new[] { 5f, 6f }));
        var norm = optimizer.ClipGradientNorm(maxNorm: 1000);

        // Check: sqrt(1² + 2² + 3² + 4² + 5² + 6²) = sqrt(91).
        double expectedNorm = Math.Sqrt(1 + 4 + 9 + 16 + 25 + 36);
        Assert.Equal(expectedNorm, norm.item<float>(), Tolerance);
    }

    /// Trains a weight 2 steps, clipping the gradient of step 2, and checks that
    /// clipping scaled that gradient by expectedScale.
    ///
    /// ClipGradientNorm only works out the scale; Step applies it. The check is at
    /// step 2 because at step 1 Adam ignores the size of the gradient (each value
    /// moves by about lr whatever g is), so a clipped and an unclipped step 1 look
    /// the same. At step 2, m and v mix step 1's gradient with step 2's, so step
    /// 2's size matters.
    static void AssertSecondStepClippedBy(float[] rawGradientStep2, double maxNorm, double expectedScale)
    {
        // The weight to train with our AdamW, and expectedWeight, trained by the built-in one.
        var config = new Config();
        const double lr = 1e-2;
        float[] originalWeight = { 0.5f, -1.0f,      // row 0
                                   2.0f,  0.25f };   // row 1
        var weight = nn.Parameter(tensor(originalWeight, new long[] { 2, 2 }));
        var optimizer = new AdamW(new FlatParameters(new[] { weight }), config);

        // expectedWeight: starts equal to the original weight, and TorchSharp's built-in
        // AdamW (PyTorch's own) trains it with the same settings as ours. It is what
        // ours must turn weight into.
        var expectedWeight = nn.Parameter(tensor(originalWeight, new long[] { 2, 2 }));
        var torchAdamW = optim.AdamW(new[] { expectedWeight }, lr: lr, beta1: config.Beta1, beta2: config.Beta2,
                                     eps: config.Eps, weight_decay: config.WeightDecay);

        // Train both step 1 as usual, with the same gradient.
        float[] rawGradientStep1 = { 0.1f, -0.2f, 0.3f, 0.05f };
        SimulateGradients((weight, rawGradientStep1), (expectedWeight, rawGradientStep1));
        optimizer.Step(lr);
        torchAdamW.step();

        // Train both step 2: ours gets the raw gradient and clips it; expectedWeight gets
        // the gradient already scaled by expectedScale, i.e. what clipping should give.
        torchAdamW.zero_grad();
        SimulateGradients((weight, rawGradientStep2));
        optimizer.ClipGradientNorm(maxNorm);
        optimizer.Step(lr);
        SimulateGradients((expectedWeight, rawGradientStep2.Select(g => (float)(g * expectedScale)).ToArray()));
        torchAdamW.step();

        // Check: ours equals the built-in (close, not bit for bit: the scale is
        // worked out on the tensor in ours and in doubles here).
        Assert.Equal(expectedWeight.data<float>().ToArray(), weight.data<float>().ToArray(), (expected, actual) => Math.Abs(expected - actual) < Tolerance);
    }

    [Fact]
    public void ClipScalesLargeGradientsDownToMaxNorm()
    {
        // Norm sqrt(1 + 4 + 9 + 16) = sqrt(30), above maxNorm 1: every gradient is divided
        // by the same number, sqrt(30), so the norm becomes exactly 1.
        AssertSecondStepClippedBy(new[] { 1f, 2f, 3f, 4f }, maxNorm: 1.0, expectedScale: 1 / Math.Sqrt(30));
    }

    [Fact]
    public void ClipLeavesSmallGradientsAlone()
    {
        // Norm sqrt(0.3² + 0.4²) = 0.5, below maxNorm 1: the gradient stays as it is.
        AssertSecondStepClippedBy(new[] { 0.3f, 0f, 0.4f, 0f }, maxNorm: 1.0, expectedScale: 1.0);
    }

    // ── the step consumes the gradients ────────────────────────────────────

    [Fact]
    public void StepLeavesTheGradientsAtZero()
    {
        // Our Step sets every gradient back to 0 after using it, so the next step starts
        // clean. The training loop relies on it, and so do the tests above that skip
        // ZeroGradients between steps.

        // A weight and a bias, and our AdamW to train them.
        var weight = nn.Parameter(tensor(new[] { 1f, 2f,
                                                 3f, 4f }, new long[] { 2, 2 }));
        var bias = nn.Parameter(tensor(new[] { 1f, 2f }, new long[] { 2 }));
        var optimizer = new AdamW(new FlatParameters(new[] { weight, bias }), new Config());

        // Train one step, as the training loop does: gradients, clip, step.
        SimulateGradients((weight, new[] { 1f, 2f, 3f, 4f }), (bias, new[] { 5f, 6f }));
        optimizer.ClipGradientNorm(maxNorm: 1.0);
        optimizer.Step(learningRate: 0.1);

        // Check: every gradient is 0 again.
        Assert.All(weight.grad!.data<float>().ToArray(), gradient => Assert.Equal(0f, gradient));
        Assert.All(bias.grad!.data<float>().ToArray(), gradient => Assert.Equal(0f, gradient));
    }

    // ── the learning-rate schedule ─────────────────────────────────────────

    /// Round numbers, so the expected learning rates are easy to work out by hand:
    /// peak 1.0, minimum 0.1, warmup over steps 0-9, cosine from step 10 down to step 110.
    static Config ScheduleSettings() => new()
    {
        LearningRate = 1.0, MinLearningRate = 0.1, WarmupSteps = 10, MaxSteps = 110,
    };

    [Theory]
    [InlineData(0, 0.1)]      // warmup starts at lr / warmup, never 0
    [InlineData(4, 0.5)]
    [InlineData(9, 1.0)]      // warmup ends at the peak
    [InlineData(10, 1.0)]     // the cosine starts at the peak
    [InlineData(60, 0.55)]    // halfway down the cosine: (peak + min) / 2
    [InlineData(110, 0.1)]    // MaxSteps: the minimum
    [InlineData(500, 0.1)]    // past MaxSteps it stays at the minimum
    public void ScheduleHitsItsLandmarks(int step, double expectedLearningRate)
    {
        Assert.Equal(expectedLearningRate, LearningRateSchedule.At(step, ScheduleSettings()), 1e-9);
    }

    [Fact]
    public void ScheduleRisesThenOnlyFalls()
    {
        // The learning rate at every step from 0 to 119.
        var config = ScheduleSettings();
        double[] learningRates = Enumerable.Range(0, 120).Select(step => LearningRateSchedule.At(step, config)).ToArray();

        // Check: it goes up every step of the warmup, then never goes up again.
        for (int step = 1; step < config.WarmupSteps; step++)
            Assert.True(learningRates[step] > learningRates[step - 1], $"warmup step {step}");
        for (int step = config.WarmupSteps + 1; step < learningRates.Length; step++)
            Assert.True(learningRates[step] <= learningRates[step - 1], $"decay step {step}");
    }

    // ── benchmark ──────────────────────────────────────────────────────────

    /// How long one optimiser step takes: ours against TorchSharp's built-in
    /// torch.optim.AdamW, used the plain way (one optimiser over the parameter list).
    ///
    /// Ours works on two flat buffers, so a step is a few operations over the whole
    /// model at once. The built-in loops over the parameters, doing its operations
    /// once per parameter tensor. GPT-2 has 148 parameter tensors, so the loop is
    /// what ours saves; the difference is largest on a GPU, where every operation
    /// is a kernel launch with a fixed cost.
    ///
    /// Prints the timings and does not fail on them (timings vary from machine to
    /// machine and run to run). To see the output, or to leave it out:
    ///     dotnet test Gpt2Trainer.sln --filter Category=AdamW-Benchmark --logger "console;verbosity=detailed"
    ///     dotnet test Gpt2Trainer.sln --filter Category!=AdamW-Benchmark
    [Fact]
    [Trait("Category", "AdamW-Benchmark")]
    public void BenchmarkOursAgainstTheBuiltInAdamW()
    {
        const int warmupSteps = 3, timedSteps = 20;
        const double lr = 1e-3;

        // The parameters to train: GPT-2's real set of 148 tensors (every layer, the
        // embeddings, LayerNorms, biases), but narrow (128 wide instead of 768) so the
        // test stays light: ~9M values instead of 124M. Built by Gpt2, so ours gets them
        // already in its flat buffers. On the GPU if there is one, else the CPU.
        var device = cuda.is_available() ? CUDA : CPU;
        var config = new Config
        {
            NLayer = 12, NHead = 4, NEmbd = 128, SequenceLength = 1024, VocabSize = 50257, PaddedVocabSize = 50304,
        };
        manual_seed(0);                                        // the same random numbers on every run
        var model = new Gpt2(config, device);
        var parameters = model.parameters().ToArray();
        var optimizer = new AdamW(model.Flat, config);
        var originalFirstParameter = parameters[0].detach().clone();   // to check at the end that training happened

        // expectedParameters: plain PyTorch parameters with the same values, not in any
        // flat buffer, trained by the built-in AdamW with the same settings.
        var expectedParameters = parameters.Select(parameter => nn.Parameter(parameter.detach().clone())).ToArray();
        var torchAdamW = optim.AdamW(expectedParameters, lr: lr, beta1: config.Beta1, beta2: config.Beta2,
                                     eps: config.Eps, weight_decay: config.WeightDecay);

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

        // Waits until the device has finished all queued work: on a GPU, operations run
        // later than the call that queues them, so the clock must not stop before they end.
        // Reading a number back to the CPU makes it wait.
        void WaitForDevice() => _ = parameters[0].sum().item<float>();

        // Times `step` over timedSteps steps, after warmupSteps untimed ones (the first
        // steps allocate memory and warm caches, which would skew the timing). Only the
        // optimiser work is timed: the backward() before each step is outside the clock.
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

        // Ours: Step also sets the gradients back to 0.
        double ours = MillisecondsPerStep(parameters, () => optimizer.Step(lr));

        // The built-in, the plain way: step(), then zero_grad() so the next backward()
        // starts from 0 (ours does that inside Step, so both are timed doing the same work).
        double builtIn = MillisecondsPerStep(expectedParameters, () => { torchAdamW.step(); torchAdamW.zero_grad(); });

        output.WriteLine($"device {device.type}, {parameters.Length} parameter tensors, " +
                         $"{parameters.Sum(parameter => parameter.numel()) / 1e6:F1}M values, {timedSteps} timed steps");
        output.WriteLine($"ours:      {ours,8:F2} ms per step");
        output.WriteLine($"built-in:  {builtIn,8:F2} ms per step");
        output.WriteLine($"ours is {builtIn / ours:F2}x the speed of the built-in");

        // Check only that both really trained: the first parameter moved away from its
        // original values in both (timing an optimiser that did nothing would mean nothing).
        // The speed itself is reported, not asserted.
        Assert.True((parameters[0].detach() - originalFirstParameter).abs().max().item<float>() > 0, "ours did not train");
        Assert.True((expectedParameters[0].detach() - originalFirstParameter).abs().max().item<float>() > 0, "the built-in did not train");
    }
}
