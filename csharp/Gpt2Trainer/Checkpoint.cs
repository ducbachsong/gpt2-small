// Checkpoint.cs — everything needed to use the model or to continue training.
//
// A checkpoint is a folder:
//     model.bin      the weights (TorchSharp's format)
//     optimizer.bin  AdamW's m and v, in parameter order
//     config.json    the settings the model was built with
//     state.json     step, tokens, AdamW's step count, fp16 loss scale
// It is written to "<folder>.tmp" and renamed at the end, so a folder that
// exists is always complete, even after Ctrl+C or a crash mid-save.
using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;

namespace Gpt2Trainer;

public sealed record TrainingState(int Step, long Tokens, long AdamSteps, double LossScale);

public static class Checkpoint
{
    public static void Save(string folder, Gpt2 model, AdamW optimizer, TrainingState state)
    {
        string temporary = folder.TrimEnd('/', '\\') + ".tmp";
        if (Directory.Exists(temporary)) Directory.Delete(temporary, recursive: true);
        Directory.CreateDirectory(temporary);

        model.save(Path.Combine(temporary, "model.bin"));
        using (var writer = new BinaryWriter(File.Create(Path.Combine(temporary, "optimizer.bin"))))
            foreach (var moment in optimizer.Moments) { using var onCpu = moment.cpu(); onCpu.Save(writer); }
        File.WriteAllText(Path.Combine(temporary, "config.json"), model.Config.ToJson());
        File.WriteAllText(Path.Combine(temporary, "state.json"), JsonSerializer.Serialize(state));

        if (Directory.Exists(folder)) Directory.Delete(folder, recursive: true);
        Directory.Move(temporary, folder);
    }

    public static Config LoadConfig(string folder) =>
        Config.FromJson(File.ReadAllText(Path.Combine(folder, "config.json")));

    public static void LoadModel(string folder, Gpt2 model) =>
        model.load(Path.Combine(folder, "model.bin"));

    public static TrainingState LoadTraining(string folder, Gpt2 model, AdamW optimizer)
    {
        LoadModel(folder, model);
        using (var reader = new BinaryReader(File.OpenRead(Path.Combine(folder, "optimizer.bin"))))
        using (no_grad())
            foreach (var moment in optimizer.Moments)
            {
                using var saved = TensorExtensionMethods.Load(reader);   // read on the CPU, then copied over
                moment.copy_(saved);
            }
        var state = JsonSerializer.Deserialize<TrainingState>(File.ReadAllText(Path.Combine(folder, "state.json")))!;
        optimizer.T = state.AdamSteps;
        return state;
    }
}
