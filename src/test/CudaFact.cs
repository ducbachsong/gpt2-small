// CudaFact.cs — [CudaFact]: a test that needs a CUDA GPU, skipped without one.
//
// dotnet test builds against the CPU LibTorch unless told otherwise
// (-p:TorchBackend=cuda-linux or cuda-windows), and a machine may have no GPU
// at all. A [CudaFact] test then shows as skipped, with the reason, instead of
// failing. xUnit reads Skip from the attribute once it is made, so setting it
// here, in the constructor, decides for each run.
using Xunit;
using static TorchSharp.torch;

namespace Gpt2Trainer.Tests;

public sealed class CudaFactAttribute : FactAttribute
{
    public CudaFactAttribute()
    {
        if (!cuda.is_available())
            Skip = "needs a CUDA GPU, and LibTorch's CUDA build: -p:TorchBackend=cuda-linux (or cuda-windows)";
    }
}
