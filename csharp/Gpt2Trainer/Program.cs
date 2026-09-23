// Program.cs — the C# trainer's entry point. traingpt2cs.py runs it; you can too:
//
//   dotnet bin/Release/net8.0/Gpt2Trainer.dll --data pool --micro-batch 8 --grad-accum-steps 4 \
//          --max-steps 3000 --prompts "464,3290;818,19473"
//
// It takes any setting in Config.cs as --kebab-case-name value. Prompts are
// token ids: the tokenizer stays on the Python side, like every other data step.
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
            var config = Config.FromArgs(args);
            manual_seed(config.Seed);
            bool cuda = config.Device == "cuda" || (config.Device == "auto" && torch.cuda.is_available());
            if (cuda)
            {
                // fp32 matmuls on TF32 tensor cores (Ampere and newer); no effect on a T4.
                torch.backends.cuda.matmul.allow_tf32 = true;
                torch.backends.cudnn.allow_tf32 = true;
            }
            new Trainer(config, cuda ? CUDA : CPU).Train();
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine($"[Fatal] {error}");
            return 1;
        }
    }
}
