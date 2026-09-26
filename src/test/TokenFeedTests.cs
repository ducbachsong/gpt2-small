// TokenFeedTests.cs — the C# end of the pipe, with pinned int arrays standing in for Python.
//
//     dotnet test Gpt2Trainer.sln        (from the repo root)
//
// traingpt2cs.py's next_batch returns the address of a numpy int32 array; here
// NextBatch returns the address of a pinned .NET int[] laid out the same way,
// so the feed reads exactly what it would read from Python.
using System.Runtime.InteropServices;
using Gpt2Trainer;
using Xunit;
using static TorchSharp.torch;

namespace Gpt2Trainer.Tests;

public class TokenFeedTests
{
    const int Rows = 4, Length = 3;

    static Config Settings(int microBatch = 2, int evalRows = 2) => new()
    {
        BatchRows = Rows, SequenceLength = Length, MicroBatch = microBatch, EvalRows = evalRows,
    };

    /// Batch b holds the ids b*100 + 0, 1, 2, ... in row-major order.
    static int[] Batch(int b) => Enumerable.Range(0, Rows * Length).Select(i => b * 100 + i).ToArray();

    /// A NextBatch over `count` batches, then 0, like next_batch once TokenPool runs out.
    sealed class FakePython : IDisposable
    {
        readonly List<GCHandle> pinned = new();
        readonly int count;
        int given;

        public FakePython(int count) => this.count = count;

        public long Next()
        {
            if (given == count) return 0;
            var handle = GCHandle.Alloc(Batch(given++), GCHandleType.Pinned);
            pinned.Add(handle);
            return handle.AddrOfPinnedObject().ToInt64();
        }

        public void Dispose() => pinned.ForEach(h => h.Free());
    }

    static long[] Ids(Tensor t) => t.cpu().data<long>().ToArray();

    [Fact]
    public void HeldOutIsTheFirstBatchsFirstRows()
    {
        using var python = new FakePython(2);
        using var feed = new TokenFeed(python.Next, Settings(evalRows: 2), CPU, _ => { });
        using var heldOut = feed.TakeHeldOutRows();
        Assert.Equal(new long[] { 2, Length }, heldOut.shape);
        Assert.Equal(Batch(0).Take(2 * Length).Select(i => (long)i), Ids(heldOut));
        using var first = feed.NextMicroBatch()!;                 // training starts at the second batch
        Assert.Equal(Batch(1).Take(2 * Length).Select(i => (long)i), Ids(first));
    }

    [Fact]
    public void MicroBatchesWalkEachBatchInOrderThenEnd()
    {
        using var python = new FakePython(3);
        using var feed = new TokenFeed(python.Next, Settings(microBatch: 2), CPU, _ => { });
        feed.TakeHeldOutRows().Dispose();
        var seen = new List<long>();
        for (Tensor? ids; (ids = feed.NextMicroBatch()) is not null; )
        {
            Assert.Equal(new long[] { 2, Length }, ids.shape);
            Assert.Equal(ScalarType.Int64, ids.dtype);
            seen.AddRange(Ids(ids));
        }
        Assert.Equal(Batch(1).Concat(Batch(2)).Select(i => (long)i), seen);
    }

    [Fact]
    public void AMicroBatchLargerThanABatchIsRefused() =>
        Assert.Throws<ArgumentException>(() => new TokenFeed(() => 0, Settings(microBatch: Rows + 1), CPU, _ => { }));

    [Fact]
    public void AnEmptyDatasetIsAnError()
    {
        using var feed = new TokenFeed(() => 0, Settings(), CPU, _ => { });
        Assert.Throws<InvalidDataException>(() => feed.TakeHeldOutRows());
    }
}
