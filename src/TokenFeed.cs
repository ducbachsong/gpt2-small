// TokenFeed.cs — the C# end of the pipe: token batches straight from Python's TokenPool.
//
// traingpt2cs.py loads this dll into its own process (pythonnet) and hands the
// trainer a NextBatch function. Each call runs token_pool.get_token_batch() in
// Python and returns the address of that batch, an int32 (BatchRows,
// SequenceLength) numpy array, or 0 once the dataset ran out. Python keeps the
// array alive until the next call; by then it has been copied to the GPU.
// No files, no second process: one batch is one copy into the GPU.
//
// The first batch is held out (TakeHeldOutRows) and never trained on. Each
// batch goes to the GPU whole and micro-batches are views of it, so Python is
// asked only once per BatchRows / MicroBatch micro-batches, and TokenPool
// usually has the next batch ready.
using System.Diagnostics;
using System.Runtime.InteropServices;
using TorchSharp;
using static TorchSharp.torch;

namespace Gpt2Trainer;

/// Python's side of the pipe: the address of the next int32 batch, or 0 when there is none.
public delegate long NextBatch();

/// Hands out (MicroBatch, SequenceLength) micro-batches, taking TokenPool batches in order.
public sealed class TokenFeed : IDisposable
{
    readonly NextBatch nextBatch;
    readonly Config config;
    readonly Device device;
    readonly Action<string> report;
    Tensor? currentBatch;   // the TokenPool batch being used, already on the GPU
    long nextRow;           // first row of the next micro-batch in currentBatch

    public TokenFeed(NextBatch nextBatch, Config config, Device device, Action<string> report)
    {
        if (config.BatchRows < config.MicroBatch)
            throw new ArgumentException($"a batch has {config.BatchRows} rows, fewer than the micro-batch of {config.MicroBatch}");
        this.nextBatch = nextBatch;
        this.config = config;
        this.device = device;
        this.report = report;
    }

    /// The first batch's first EvalRows rows, for the validation loss.
    public Tensor TakeHeldOutRows()
    {
        using var firstBatch = FetchBatch() ?? throw new InvalidDataException("the dataset has no batch to hold out");
        return firstBatch.narrow(0, 0, Math.Min(config.EvalRows, firstBatch.shape[0])).clone().DetachFromDisposeScope();
    }

    /// The next micro-batch, a view into the current batch; null once the data ran out.
    public Tensor? NextMicroBatch()
    {
        int rows = config.MicroBatch;
        if (currentBatch is null || nextRow + rows > currentBatch.shape[0])
        {
            currentBatch?.Dispose();
            currentBatch = FetchBatch();
            nextRow = 0;
            if (currentBatch is null) return null;
        }
        var microBatch = currentBatch.narrow(0, nextRow, rows);
        nextRow += rows;
        return microBatch;
    }

    /// One batch from Python, as (BatchRows, SequenceLength) int64 token ids on the device.
    Tensor? FetchBatch()
    {
        var waitClock = Stopwatch.StartNew();
        long batchAddress = nextBatch();
        if (waitClock.Elapsed.TotalSeconds > 1)
            report($"[Wait] GPU idle {waitClock.Elapsed.TotalSeconds:F1}s for a token batch");
        if (batchAddress == 0) return null;
        var tokenIds = new int[config.BatchRows * config.SequenceLength];
        Marshal.Copy((IntPtr)batchAddress, tokenIds, 0, tokenIds.Length);
        using var scope = NewDisposeScope();
        return tensor(tokenIds, new long[] { config.BatchRows, config.SequenceLength })
            .to(device).to(ScalarType.Int64).DetachFromDisposeScope();
    }

    public void Dispose() => currentBatch?.Dispose();
}
