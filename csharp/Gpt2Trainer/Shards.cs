// Shards.cs — the C# end of the pipe: token shards the Python side writes.
//
// A shard is one TokenPool batch as a file, written by traingpt2cs.py:
//     [u32 magic 'GPT2'][u32 rows][u32 sequence_length][rows x sequence_length u16 token ids]
// Python writes shard_000000.bin, shard_000001.bin, ... (tmp file + rename,
// so a shard is never seen half written) and heldout.bin once, first.
// Everything between the two sides is files in one folder, as in lab06:
//     consumed.txt  C# -> Python: the highest shard index already on the GPU;
//                   Python deletes those files and writes more
//     STOP          either side: Python has no more data, or C# is done
// Each shard is copied to the GPU whole the moment it is read, so training
// never waits on a file between shards.
using System.Runtime.InteropServices;
using TorchSharp;
using static TorchSharp.torch;

namespace Gpt2Trainer;

public static class ShardFile
{
    public const uint Magic = 0x32545047;   // "GPT2" read little-endian

    /// A shard file -> (rows, sequence_length) int64 token ids on `device`.
    public static Tensor Load(string path, int sequenceLength, Device device)
    {
        byte[] bytes = File.ReadAllBytes(path);
        var header = MemoryMarshal.Cast<byte, uint>(bytes.AsSpan(0, 12));
        if (header[0] != Magic) throw new InvalidDataException($"{path} is not a token shard");
        int rows = (int)header[1], length = (int)header[2];
        if (length != sequenceLength)
            throw new InvalidDataException($"{path} has sequences of {length}, the model expects {sequenceLength}");
        var tokens = MemoryMarshal.Cast<byte, ushort>(bytes.AsSpan(12, rows * length * 2));
        var ids = new int[tokens.Length];
        for (int i = 0; i < ids.Length; i++) ids[i] = tokens[i];
        using var scope = NewDisposeScope();
        return tensor(ids, new long[] { rows, length }).to(device).to(ScalarType.Int64)
            .MoveToOuterDisposeScope();
    }
}

/// Hands out (MicroBatch, SequenceLength) batches, reading shards in order.
public sealed class ShardReader : IDisposable
{
    readonly string folder;
    readonly Config config;
    readonly Device device;
    Tensor? shard;          // the shard being used, already on the GPU
    long row;
    int nextIndex;
    bool announcedWait;

    public ShardReader(Config config, Device device)
    {
        this.config = config;
        this.device = device;
        folder = config.Data;
    }

    public string StopPath => Path.Combine(folder, "STOP");

    /// Waits for heldout.bin and returns its first EvalRows rows.
    public Tensor HeldOut()
    {
        string path = Path.Combine(folder, "heldout.bin");
        WaitFor(() => File.Exists(path), "heldout.bin");
        using var all = ShardFile.Load(path, config.SequenceLength, device);
        return all.narrow(0, 0, Math.Min(config.EvalRows, all.shape[0])).clone().DetachFromDisposeScope();
    }

    /// The next micro-batch, a view into the current shard; null once the data ran out.
    public Tensor? Next()
    {
        int batch = config.MicroBatch;
        if (shard is null || row + batch > shard.shape[0])
        {
            shard?.Dispose();
            shard = LoadNextShard();
            row = 0;
            if (shard is null) return null;
            if (shard.shape[0] < batch)
                throw new InvalidDataException($"a shard has {shard.shape[0]} rows, fewer than the micro-batch of {batch}");
        }
        var ids = shard.narrow(0, row, batch);
        row += batch;
        return ids;
    }

    Tensor? LoadNextShard()
    {
        string path = Path.Combine(folder, $"shard_{nextIndex:D6}.bin");
        WaitFor(() => File.Exists(path) || File.Exists(StopPath), $"shard_{nextIndex:D6}.bin");
        if (!File.Exists(path)) return null;
        var loaded = ShardFile.Load(path, config.SequenceLength, device).DetachFromDisposeScope();
        WriteAtomically(Path.Combine(folder, "consumed.txt"), $"{nextIndex}\n");
        nextIndex++;
        return loaded;
    }

    void WaitFor(Func<bool> ready, string what)
    {
        if (ready()) { announcedWait = false; return; }
        if (!announcedWait) Console.WriteLine($"[Wait] GPU idle, waiting for {what}");
        announcedWait = true;
        while (!ready()) Thread.Sleep(20);
    }

    public void Stop() => WriteAtomically(StopPath, "done\n");

    public static void WriteAtomically(string path, string text)
    {
        string temporary = path + ".tmp";
        File.WriteAllText(temporary, text);
        File.Move(temporary, path, overwrite: true);
    }

    public void Dispose() => shard?.Dispose();
}
