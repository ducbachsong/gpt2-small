// Program.cs — the C# trainer's entry point. traingpt2cs.py runs it; you can too:
//
//   dotnet run -c Release -- train --data pool --out runs/x --precision bf16 --micro-batch 16 ...
//   dotnet run -c Release -- generate --checkpoint runs/x/checkpoints/final --prompt 464,3290
//                          [--max-new-tokens 60] [--temperature 0.8] [--top-k 50]
//
// train takes any setting in Config.cs as --kebab-case-name value.
// generate takes token ids and prints token ids ([Generated] ids=...): the
// tokenizer stays on the Python side, like every other data step.
using System.Globalization;
using TorchSharp;
using static TorchSharp.torch;

namespace Gpt2Trainer;

public static class Program
{
    public static int Main(string[] args)
    {
        CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
        try
        {
            string mode = args.Length > 0 ? args[0] : "train";
            var rest = args.Skip(1).ToList();
            switch (mode)
            {
                case "train":
                    var config = Config.FromArgs(rest);
                    new Trainer(config, Setup(config)).Train();
                    return 0;
                case "generate":
                    Generate(rest);
                    return 0;
                default:
                    Console.Error.WriteLine($"unknown mode '{mode}': train | generate");
                    return 2;
            }
        }
        catch (Exception error)
        {
            Console.Error.WriteLine($"[Fatal] {error}");
            return 1;
        }
    }

    /// Seed, device, precision, TF32. Out: the device to train on.
    static Device Setup(Config config)
    {
        manual_seed(config.Seed);
        bool cuda = config.Device == "cuda" || (config.Device == "auto" && torch.cuda.is_available());
        Amp.Use(config.Precision);
        if (cuda)
        {
            // What stays fp32 (and all of an fp32 run) uses TF32 tensor cores on Ampere and newer.
            torch.backends.cuda.matmul.allow_tf32 = true;
            torch.backends.cudnn.allow_tf32 = true;
        }
        return cuda ? CUDA : CPU;
    }

    static void Generate(List<string> args)
    {
        string Take(string name, string fallback)
        {
            int i = args.IndexOf("--" + name);
            if (i < 0) return fallback;
            string value = args[i + 1];
            args.RemoveRange(i, 2);
            return value;
        }
        string folder = Take("checkpoint", "");
        if (folder == "") throw new ArgumentException("generate needs --checkpoint <folder>");
        long[] prompt = Take("prompt", "").Split(',', StringSplitOptions.RemoveEmptyEntries).Select(long.Parse).ToArray();
        if (prompt.Length == 0) throw new ArgumentException("generate needs --prompt <id,id,...>");
        int maxNewTokens = int.Parse(Take("max-new-tokens", "60"));
        double temperature = double.Parse(Take("temperature", "0.8"));
        int topK = int.Parse(Take("top-k", "50"));

        // The model comes from the checkpoint; only where and how it runs may be changed.
        var config = Checkpoint.LoadConfig(folder);
        var overrides = Config.FromArgs(args);
        if (args.Contains("--device")) config.Device = overrides.Device;
        if (args.Contains("--precision")) config.Precision = overrides.Precision;
        if (args.Contains("--seed")) config.Seed = overrides.Seed;
        var device = Setup(config);
        var model = new Gpt2(config).to(device);
        Checkpoint.LoadModel(folder, model);
        var ids = model.Generate(prompt, maxNewTokens, temperature, topK, device);
        Console.WriteLine($"[Generated] ids={string.Join(",", ids)}");
    }
}
