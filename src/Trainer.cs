// Trainer.cs — the training loop.
//
// Each step: GradAccumSteps microStep-batches forward and backward, then clip,
// then AdamW. The loss and grad norm of each step are copied into small
// buffers on the GPU; every PrintEvery steps the buffers come back to the CPU
// in one go and each step is printed as a line. So the CPU waits for the GPU
// once per PrintEvery steps, not once per step, and keeps queueing work ahead.
// At the end the model continues each prompt, and the run is over: nothing is
// saved.
//
// traingpt2cs.py calls it from inside its own process (pythonnet):
//
//     trainer = Trainer.Start(args, NextBatch(next_batch), Action[String](report))
//     while not trainer.Finished: ...        # Python's threads keep tokenizing
//     trainer.Error                          # None, or what went wrong
//
// args are the settings of Config.cs as "--kebab-case-name value" pairs.
// The trainer runs on a .NET thread of its own: a .NET call made from Python
// holds Python's lock (the GIL) until it returns, so training on Python's
// thread would stop the tokenizer threads for the whole run. This thread
// takes the GIL only inside the two callbacks.
//
// Every line goes back to Python through the report callback:
//     [Config] params=124.4M device=cuda ...
//     [Step] step=25 tokens=819200 loss=7.0123 lr=5.2e-05 grad_norm=1.23 [tokens_per_second=... left=...]
//     [Eval] step=250 val_loss=5.4321
//     [Sample] ids=464,3290,...
//     [Done] step=3000 val_loss=3.21
using System.Diagnostics;
using System.Globalization;
using TorchSharp;
using static TorchSharp.torch;

namespace Gpt2Trainer;

public sealed class Trainer
{
    readonly Config config;
    readonly Device device;
    readonly NextBatch nextBatch;
    readonly Action<string> reportLine;
    readonly Thread thread;
    volatile bool stopRequested;

    /// null while training and after a clean finish; the exception text otherwise.
    public string? Error { get; private set; }
    public bool Finished => !thread.IsAlive;

    Trainer(Config config, NextBatch nextBatch, Action<string> report)
    {
        manual_seed(config.Seed);
        bool cuda = config.Device == "cuda" || (config.Device == "auto" && torch.cuda.is_available());
        // Pure fp32: no TF32 anywhere. LibTorch allows it in cuDNN by default, so both are set.
        torch.backends.cuda.matmul.allow_tf32 = false;
        torch.backends.cudnn.allow_tf32 = false;
        this.config = config;
        this.device = cuda ? CUDA : CPU;
        this.nextBatch = nextBatch;
        this.reportLine = report;
        thread = new Thread(TrainOnThisThread) { Name = "gpt2-trainer", IsBackground = true };
    }

    /// Starts training on the trainer's own thread and returns at once.
    public static Trainer Start(string[] args, NextBatch nextBatch, Action<string> report)
    {
        var trainer = new Trainer(Config.FromArgs(args), nextBatch, report);
        trainer.thread.Start();
        return trainer;
    }

    /// Ctrl+C in Python: finish this step, evaluate and sample, then end.
    public void RequestStop()
    {
        if (stopRequested) return;
        stopRequested = true;
        Log("[Stop] Ctrl+C: finishing this step");
    }

    void TrainOnThisThread()
    {
        Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
        try
        {
            Train();
        }
        catch (Exception error)
        {
            Error = error.ToString();
        }
    }

    void Train()
    {
        var model = new Gpt2(config, device);              // on the device, parameters already flat
        var optimizer = new AdamW(model.Flat, config);
        using var feed = new TokenFeed(nextBatch, config, device, Log);

        long parameterCount = model.parameters().Sum(p => p.numel());
        Log($"[Config] params={parameterCount / 1e6:F1}M device={device.type.ToString().ToLowerInvariant()} " +
            $"micro_batch={config.MicroBatch} grad_accum={config.GradAccumSteps} " +
            $"tokens_per_step={config.TokensPerStep} max_steps={config.MaxSteps}");

        int step = 0;
        using var heldOutRows = feed.TakeHeldOutRows();
        double valLoss = ValidationLoss(model, heldOutRows);
        Log($"[Eval] step={step} val_loss={valLoss:F4}");
        int lastEvalStep = step;

        using var pendingLosses = zeros(config.PrintEvery, device: device).DetachFromDisposeScope();
        using var pendingGradNorms = zeros(config.PrintEvery, device: device).DetachFromDisposeScope();
        var pendingLearningRates = new double[config.PrintEvery];
        int pendingCount = 0;
        var trainingClock = Stopwatch.StartNew();
        var reportWindowClock = Stopwatch.StartNew();
        bool dataRanOut = false;

        void ReportPendingSteps()
        {
            if (pendingCount == 0) return;
            float[] losses = pendingLosses.cpu().data<float>().ToArray();   // the one wait for the GPU
            float[] norms = pendingGradNorms.cpu().data<float>().ToArray();
            double seconds = reportWindowClock.Elapsed.TotalSeconds;
            for (int i = 0; i < pendingCount; i++)
            {
                int reportedStep = step - pendingCount + 1 + i;
                string line = $"[Step] step={reportedStep} tokens={reportedStep * config.TokensPerStep} loss={losses[i]:F4} " +
                              $"lr={pendingLearningRates[i]:0.00e+00} grad_norm={norms[i]:F3}";
                if (i == pendingCount - 1)
                    line += $" tokens_per_second={pendingCount * config.TokensPerStep / seconds:F0} " +
                            $"left={TimeLeft(step, config.MaxSteps - step, trainingClock.Elapsed.TotalSeconds)}";
                Log(line);
            }
            pendingCount = 0;
            reportWindowClock.Restart();
        }

        model.train();
        while (step < config.MaxSteps && !stopRequested)
        {
            using var scope = NewDisposeScope();
            double learningRate = LearningRateSchedule.At(step, config);

            var stepLoss = zeros(1, device: device);
            for (int microStep = 0; microStep < config.GradAccumSteps; microStep++)
            {
                var microBatch = feed.NextMicroBatch();
                if (microBatch is null) { dataRanOut = true; break; }
                var loss = model.Loss(microBatch) / config.GradAccumSteps;
                loss.backward();
                stepLoss.add_(loss.detach());
            }
            if (dataRanOut) { optimizer.ZeroGradients(); break; }

            var gradNorm = optimizer.ClipGradientNorm(config.GradClip);
            optimizer.Step(learningRate);                 // also leaves the gradients at 0
            step++;

            using (no_grad())
            {
                pendingLosses[pendingCount].copy_(stepLoss[0]);
                pendingGradNorms[pendingCount].copy_(gradNorm);
            }
            pendingLearningRates[pendingCount++] = learningRate;
            if (pendingCount == config.PrintEvery) ReportPendingSteps();

            if (step % config.EvalEvery == 0)
            {
                ReportPendingSteps();
                valLoss = ValidationLoss(model, heldOutRows);
                lastEvalStep = step;
                Log($"[Eval] step={step} val_loss={valLoss:F4}");
            }
        }
        ReportPendingSteps();

        if (dataRanOut) Log($"[Stop] the dataset ran out at step {step}");
        if (lastEvalStep != step)
        {
            valLoss = ValidationLoss(model, heldOutRows);
            Log($"[Eval] step={step} val_loss={valLoss:F4}");
        }
        foreach (var prompt in config.Prompts.Split(';', StringSplitOptions.RemoveEmptyEntries))
        {
            long[] promptIds = prompt.Split(',').Select(long.Parse).ToArray();
            var sample = model.Generate(promptIds, config.MaxNewTokens, config.Temperature, config.TopK, device);
            Log($"[Sample] ids={string.Join(",", sample)}");
        }
        Log($"[Done] step={step} val_loss={valLoss:F4}");
    }

    /// Mean loss over the held-out rows.
    double ValidationLoss(Gpt2 model, Tensor heldOutRows)
    {
        using var _ = no_grad();
        using var scope = NewDisposeScope();
        model.eval();
        var lossSum = zeros(1, device: device);
        long rows = heldOutRows.shape[0];
        for (long start = 0; start < rows; start += config.MicroBatch)
        {
            long count = Math.Min(config.MicroBatch, rows - start);
            lossSum.add_(model.Loss(heldOutRows.narrow(0, start, count)) * count);
        }
        model.train();
        return lossSum.item<float>() / rows;
    }

    static string TimeLeft(int stepsDone, int stepsLeft, double seconds)
    {
        if (stepsDone <= 0) return "?";
        var left = TimeSpan.FromSeconds(seconds / stepsDone * stepsLeft);
        return left.TotalHours >= 1 ? $"{(int)left.TotalHours}h{left.Minutes:D2}m" : $"{left.Minutes}m{left.Seconds:D2}s";
    }

    void Log(string line) => reportLine(line);
}
