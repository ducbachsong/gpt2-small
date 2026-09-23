// Trainer.cs — the training loop.
//
// Each step: GradAccumSteps micro-batches forward and backward, then clip,
// then AdamW. The loss and grad norm of each step are copied into small
// buffers on the GPU; every PrintEvery steps the buffers come back to the CPU
// in one go and each step is printed as a line. So the CPU waits for the GPU
// once per PrintEvery steps, not once per step, and keeps queueing work ahead.
// At the end the model continues each prompt, and the run is over: nothing is
// saved.
//
// The console is the only thing the Python side reads (as in lab06):
//     [Config] params=124.4M device=cuda ...
//     [Step] step=25 tokens=819200 loss=7.0123 lr=5.2e-05 grad_norm=1.23 [tokens_per_second=... left=...]
//     [Eval] step=250 val_loss=5.4321
//     [Sample] ids=464,3290,...
//     [Done] step=3000 val_loss=3.21
using System.Diagnostics;
using TorchSharp;
using static TorchSharp.torch;

namespace Gpt2Trainer;

public sealed class Trainer
{
    readonly Config config;
    readonly Device device;
    volatile bool stopRequested;

    public Trainer(Config config, Device device)
    {
        this.config = config;
        this.device = device;
        Console.CancelKeyPress += (_, e) =>
        {
            if (stopRequested) return;          // a second Ctrl+C kills it
            e.Cancel = true;
            stopRequested = true;
            Log("[Stop] Ctrl+C: finishing this step");
        };
    }

    public void Train()
    {
        var model = new Gpt2(config).to(device);
        var optimizer = new AdamW(model.parameters(), config);
        using var reader = new ShardReader(config, device);

        long parameterCount = model.parameters().Sum(p => p.numel());
        Log($"[Config] params={parameterCount / 1e6:F1}M device={device.type.ToString().ToLowerInvariant()} " +
            $"micro_batch={config.MicroBatch} grad_accum={config.GradAccumSteps} " +
            $"tokens_per_step={config.TokensPerStep} max_steps={config.MaxSteps}");

        int step = 0;
        using var heldOut = reader.HeldOut();
        double valLoss = Evaluate(model, heldOut);
        Log($"[Eval] step={step} val_loss={valLoss:F4}");
        int evaluatedAt = step;

        using var lossLog = zeros(config.PrintEvery, device: device).DetachFromDisposeScope();
        using var normLog = zeros(config.PrintEvery, device: device).DetachFromDisposeScope();
        var lrLog = new double[config.PrintEvery];
        int logged = 0;
        var runClock = Stopwatch.StartNew();
        var windowClock = Stopwatch.StartNew();
        bool ranOut = false;

        void Flush()
        {
            if (logged == 0) return;
            float[] losses = lossLog.cpu().data<float>().ToArray();   // the one wait for the GPU
            float[] norms = normLog.cpu().data<float>().ToArray();
            double seconds = windowClock.Elapsed.TotalSeconds;
            for (int i = 0; i < logged; i++)
            {
                int s = step - logged + 1 + i;
                string line = $"[Step] step={s} tokens={s * config.TokensPerStep} loss={losses[i]:F4} " +
                              $"lr={lrLog[i]:0.00e+00} grad_norm={norms[i]:F3}";
                if (i == logged - 1)
                    line += $" tokens_per_second={logged * config.TokensPerStep / seconds:F0} " +
                            $"left={TimeLeft(step, config.MaxSteps - step, runClock.Elapsed.TotalSeconds)}";
                Log(line);
            }
            logged = 0;
            windowClock.Restart();
        }

        model.train();
        while (step < config.MaxSteps && !stopRequested)
        {
            using var scope = NewDisposeScope();
            double lr = Schedule.LearningRateAt(step, config);

            var stepLoss = zeros(1, device: device);
            for (int micro = 0; micro < config.GradAccumSteps; micro++)
            {
                var ids = reader.Next();
                if (ids is null) { ranOut = true; break; }
                var loss = model.Loss(ids) / config.GradAccumSteps;
                loss.backward();
                stepLoss.add_(loss.detach());
            }
            if (ranOut) { optimizer.ZeroGrad(); break; }

            var norm = optimizer.ClipGradNorm(config.GradClip);
            optimizer.Step(lr);
            optimizer.ZeroGrad();
            step++;

            using (no_grad())
            {
                lossLog[logged].copy_(stepLoss[0]);
                normLog[logged].copy_(norm);
            }
            lrLog[logged++] = lr;
            if (logged == config.PrintEvery) Flush();

            if (step % config.EvalEvery == 0)
            {
                Flush();
                valLoss = Evaluate(model, heldOut);
                evaluatedAt = step;
                Log($"[Eval] step={step} val_loss={valLoss:F4}");
            }
        }
        Flush();

        if (ranOut) Log($"[Stop] the dataset ran out at step {step}");
        if (evaluatedAt != step)
        {
            valLoss = Evaluate(model, heldOut);
            Log($"[Eval] step={step} val_loss={valLoss:F4}");
        }
        reader.Stop();
        foreach (var prompt in config.Prompts.Split(';', StringSplitOptions.RemoveEmptyEntries))
        {
            long[] ids = prompt.Split(',').Select(long.Parse).ToArray();
            var sample = model.Generate(ids, config.MaxNewTokens, config.Temperature, config.TopK, device);
            Log($"[Sample] ids={string.Join(",", sample)}");
        }
        Log($"[Done] step={step} val_loss={valLoss:F4}");
    }

    /// Mean loss over the held-out rows.
    double Evaluate(Gpt2 model, Tensor heldOut)
    {
        using var _ = no_grad();
        using var scope = NewDisposeScope();
        model.eval();
        var total = zeros(1, device: device);
        long rows = heldOut.shape[0];
        for (long start = 0; start < rows; start += config.MicroBatch)
        {
            long count = Math.Min(config.MicroBatch, rows - start);
            total.add_(model.Loss(heldOut.narrow(0, start, count)) * count);
        }
        model.train();
        return total.item<float>() / rows;
    }

    static string TimeLeft(int stepsDone, int stepsLeft, double seconds)
    {
        if (stepsDone <= 0) return "?";
        var left = TimeSpan.FromSeconds(seconds / stepsDone * stepsLeft);
        return left.TotalHours >= 1 ? $"{(int)left.TotalHours}h{left.Minutes:D2}m" : $"{left.Minutes}m{left.Seconds:D2}s";
    }

    static void Log(string line)
    {
        Console.WriteLine(line);
        Console.Out.Flush();
    }
}
