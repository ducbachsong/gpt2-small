// Config.cs — every setting of the C# trainer, with the defaults of traingpt2.py.
//
// traingpt2cs.py keeps the settings you edit and passes them here as
// "--name value" arguments, so the Python file stays the one place to change a
// run. Anything not passed keeps the default below.
using System.Globalization;

namespace Gpt2Trainer;

public sealed class Config
{
    // ── the model: GPT-2 small ──────────────────────────────────────────────
    public int NLayer { get; set; } = 12;
    public int NHead { get; set; } = 12;
    public int NEmbd { get; set; } = 768;
    public int SequenceLength { get; set; } = 1024;
    public int VocabSize { get; set; } = 50257;          // GPT-2 BPE ids: 0..50256
    public int PaddedVocabSize { get; set; } = 50304;    // a multiple of 64: faster matmuls

    // ── the optimiser: MicroBatch x SequenceLength x GradAccumSteps tokens/step ─
    public int MicroBatch { get; set; } = 4;
    public int GradAccumSteps { get; set; } = 8;
    public double LearningRate { get; set; } = 6e-4;
    public double MinLearningRate { get; set; } = 6e-5;
    public int WarmupSteps { get; set; } = 300;
    public double WeightDecay { get; set; } = 0.1;
    public double Beta1 { get; set; } = 0.9;
    public double Beta2 { get; set; } = 0.95;
    public double Eps { get; set; } = 1e-8;
    public double GradClip { get; set; } = 1.0;
    public int MaxSteps { get; set; } = 3000;

    // ── watching and saving ─────────────────────────────────────────────────
    public int EvalRows { get; set; } = 80;              // held-out rows for val_loss
    public int EvalEvery { get; set; } = 250;
    public int PrintEvery { get; set; } = 25;            // also: how often the CPU waits for the GPU

    // ── samples at the end: prompts as token ids, "464,3290;818,19473" ──────
    public string Prompts { get; set; } = "";
    public int MaxNewTokens { get; set; } = 60;
    public double Temperature { get; set; } = 0.8;
    public int TopK { get; set; } = 50;

    // ── the machine ─────────────────────────────────────────────────────────
    public string Device { get; set; } = "auto";         // auto | cuda | cpu
    public int Seed { get; set; } = 0;

    // ── where the data is ───────────────────────────────────────────────────
    public string Data { get; set; } = "pool";           // shard folder the Python side fills

    public long TokensPerStep => (long)MicroBatch * SequenceLength * GradAccumSteps;

    /// "--micro-batch 16" sets MicroBatch; the flag is the property in kebab-case.
    public static Config FromArgs(IEnumerable<string> args)
    {
        var config = new Config();
        var properties = typeof(Config).GetProperties().Where(p => p.CanWrite)
            .ToDictionary(p => Kebab(p.Name));
        var list = args.ToList();
        for (int i = 0; i < list.Count; i++)
        {
            if (!list[i].StartsWith("--"))
                throw new ArgumentException($"expected --name value, got '{list[i]}'");
            string name = list[i][2..];
            if (!properties.TryGetValue(name, out var property))
                throw new ArgumentException($"unknown setting --{name}");
            if (i + 1 >= list.Count)
                throw new ArgumentException($"--{name} needs a value");
            property.SetValue(config, Convert.ChangeType(list[++i], property.PropertyType,
                                                         CultureInfo.InvariantCulture));
        }
        return config;
    }

    static string Kebab(string name) =>
        string.Concat(name.Select((c, i) => char.IsUpper(c) && i > 0 ? "-" + char.ToLowerInvariant(c)
                                                                       : char.ToLowerInvariant(c).ToString()));
}
