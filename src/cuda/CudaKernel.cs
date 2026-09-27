// CudaKernel.cs — our own CUDA kernels: compiled when first needed, launched from C#.
//
// LibTorch comes with its kernels already compiled, and TorchSharp has no way
// to add one. So a kernel of ours (a .cu file in src/cuda/, embedded in this
// dll) goes through two NVIDIA libraries, both plain C APIs called by P/Invoke:
//
//     adamw.cu --NVRTC--> cubin --driver--> kernel --cuLaunchKernel--> runs on the GPU
//              compile     machine    load    a function                 in turn with
//              (once)      code for           on the GPU                 LibTorch's work
//                          this GPU
//
//     NVRTC        the CUDA compiler as a library (libnvrtc, nvrtc64_*.dll): CUDA
//                  source in, a cubin out: machine code for this GPU's architecture.
//                  It comes with the CUDA toolkit, and LibTorch's CUDA build has a copy.
//     the driver   libcuda.so.1 / nvcuda.dll, installed with the GPU driver: loads the
//                  cubin and launches its kernels.
//
// So nothing is compiled ahead of time, and no CUDA toolkit is needed to build
// the trainer: each kernel is compiled for the GPU it runs on, once per
// process, the first time it is used (a fraction of a second).
//
// A launch goes on the GPU's default stream, the queue LibTorch puts all its
// work on (TorchSharp never switches streams). So a kernel runs after every
// tensor operation queued before it and before every one queued after it, and
// Launch returns at once, like a tensor operation, without waiting for the GPU.
using System.Reflection;
using System.Runtime.InteropServices;
using TorchSharp;
using static TorchSharp.torch;

namespace Gpt2Trainer;

public sealed unsafe class CudaKernel
{
    const int ComputeCapabilityMajor = 75, ComputeCapabilityMinor = 76;   // CUdevice_attribute values

    readonly IntPtr context;               // the GPU's primary context: the one LibTorch uses
    readonly IntPtr function;              // the kernel, loaded and ready to launch

    static readonly Dictionary<(string, string, int), CudaKernel> loaded = new();

    /// `kernelName` from `sourceFile` (in src/cuda/), for GPU number `deviceIndex`.
    /// The first call compiles and loads it; later calls get the same kernel.
    public static CudaKernel Get(string sourceFile, string kernelName, int deviceIndex)
    {
        lock (loaded)
        {
            var key = (sourceFile, kernelName, deviceIndex);
            if (!loaded.TryGetValue(key, out var kernel))
                loaded[key] = kernel = new CudaKernel(sourceFile, kernelName, deviceIndex);
            return kernel;
        }
    }

    CudaKernel(string sourceFile, string kernelName, int deviceIndex)
    {
        // 1. The GPU, and its primary context. A context is the GPU's view of one
        //    program: its memory, its loaded kernels. The CUDA runtime, and so
        //    LibTorch, uses each GPU's primary context; our kernel must be loaded
        //    into that same one to work on LibTorch's tensors.
        Check(Driver.cuInit(0));
        Check(Driver.cuDeviceGet(out int device, deviceIndex));
        Check(Driver.cuDeviceGetAttribute(out int major, ComputeCapabilityMajor, device));
        Check(Driver.cuDeviceGetAttribute(out int minor, ComputeCapabilityMinor, device));
        Check(Driver.cuDevicePrimaryCtxRetain(out context, device));

        // 2. Compile, for this GPU's architecture: sm_75 on a T4, sm_80 on an A100.
        byte[] cubin = Compile(ReadSource(sourceFile), sourceFile, $"sm_{major}{minor}");

        // 3. Load the cubin into the context, and find the kernel in it. Loaded for
        //    good: kept until the process ends, like LibTorch's own kernels.
        Check(Driver.cuCtxPushCurrent(context));
        try
        {
            Check(Driver.cuModuleLoadData(out IntPtr module, cubin));
            Check(Driver.cuModuleGetFunction(out function, module, kernelName));
        }
        finally
        {
            Check(Driver.cuCtxPopCurrent(out _));
        }
    }

    /// Queues the kernel on the GPU and returns at once: `blocks` blocks of
    /// `threadsPerBlock` threads. `arguments` is the kernel's one parameter, a
    /// struct copied byte for byte, so it must match the kernel's struct field
    /// for field.
    public void Launch<TArguments>(long blocks, int threadsPerBlock, TArguments arguments) where TArguments : unmanaged
    {
        if (blocks < 1 || blocks > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(blocks), blocks, "1 to 2³¹-1 blocks");
        // cuLaunchKernel takes an array with the address of each parameter; there is one.
        void* argument = &arguments;
        Check(Driver.cuCtxPushCurrent(context));   // made current on this thread only for the launch
        try
        {
            Check(Driver.cuLaunchKernel(function, (uint)blocks, 1, 1, (uint)threadsPerBlock, 1, 1,
                                        sharedMemoryBytes: 0, stream: IntPtr.Zero, &argument, extra: null));
        }
        finally
        {
            Check(Driver.cuCtxPopCurrent(out _));
        }
    }

    /// Where a float32 tensor's first value is in memory (GPU memory, for a CUDA
    /// tensor): the start of its storage, plus its offset into it. For a kernel's
    /// arguments.
    public static IntPtr DevicePointer(Tensor tensor)
    {
        using var storage = tensor.storage<float>();
        return storage.data_ptr() + (nint)(tensor.storage_offset() * tensor.element_size());
    }

    /// The .cu file's text, from inside this dll (Gpt2Trainer.csproj embeds src/cuda/*.cu).
    static string ReadSource(string sourceFile)
    {
        using var stream = typeof(CudaKernel).Assembly.GetManifestResourceStream(sourceFile)
            ?? throw new FileNotFoundException($"{sourceFile} is not embedded in the dll (src/cuda/*.cu in Gpt2Trainer.csproj)");
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }

    /// NVRTC: CUDA source in, a cubin for `architecture` out. If the source does
    /// not compile, the exception carries the compiler's messages.
    static byte[] Compile(string source, string fileName, string architecture)
    {
        Check(Nvrtc.nvrtcCreateProgram(out IntPtr program, source, fileName, 0, IntPtr.Zero, IntPtr.Zero));
        try
        {
            var result = Nvrtc.nvrtcCompileProgram(program, 1, new[] { $"--gpu-architecture={architecture}" });
            if (result != NvrtcResult.Success)
            {
                Check(Nvrtc.nvrtcGetProgramLogSize(program, out nuint logSize));
                var log = new byte[logSize];
                Check(Nvrtc.nvrtcGetProgramLog(program, log));
                throw new InvalidOperationException($"{fileName} does not compile for {architecture} ({ErrorText(result)}):\n" +
                                                    System.Text.Encoding.UTF8.GetString(log).TrimEnd('\0'));
            }
            Check(Nvrtc.nvrtcGetCUBINSize(program, out nuint size));
            var cubin = new byte[size];
            Check(Nvrtc.nvrtcGetCUBIN(program, cubin));
            return cubin;
        }
        finally
        {
            Nvrtc.nvrtcDestroyProgram(ref program);
        }
    }

    // ── errors ─────────────────────────────────────────────────────────────
    // Every driver and NVRTC call returns a status, 0 for success; anything else
    // becomes an exception with the status's name.

    enum CudaResult { Success = 0 }        // CUresult: only 0 is named; the driver names the others
    enum NvrtcResult { Success = 0 }       // nvrtcResult, the same way

    static void Check(CudaResult result)
    {
        if (result == CudaResult.Success) return;
        Driver.cuGetErrorName(result, out IntPtr name);
        throw new InvalidOperationException($"CUDA driver error {(int)result} ({Marshal.PtrToStringAnsi(name)})");
    }

    static void Check(NvrtcResult result)
    {
        if (result != NvrtcResult.Success) throw new InvalidOperationException($"NVRTC error {(int)result} ({ErrorText(result)})");
    }

    static string? ErrorText(NvrtcResult result) => Marshal.PtrToStringAnsi(Nvrtc.nvrtcGetErrorString(result));

    // ── the two libraries ──────────────────────────────────────────────────
    // [DllImport("cuda")] and [DllImport("nvrtc")] are names of our own: when
    // .NET first needs one, it asks FindLibrary which file that is on this
    // machine. Set before any of them is called, as a static constructor runs
    // before the class is first used.

    static CudaKernel() => NativeLibrary.SetDllImportResolver(typeof(CudaKernel).Assembly, FindLibrary);

    static IntPtr FindLibrary(string name, Assembly assembly, DllImportSearchPath? searchPath) => name switch
    {
        "cuda" => NativeLibrary.Load(OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1"),
        "nvrtc" => LoadNvrtc(),
        _ => IntPtr.Zero,                  // any other name: .NET's usual search
    };

    /// NVRTC: the CUDA toolkit's if there is one (CUDA_PATH, CUDA_HOME,
    /// /usr/local/cuda: Colab has it), else the copy in LibTorch's CUDA build,
    /// which TorchSharp's package puts next to TorchSharp.dll, else wherever the
    /// system's own search finds one.
    static IntPtr LoadNvrtc()
    {
        bool windows = OperatingSystem.IsWindows();
        var folders = new List<string>();
        foreach (var variable in new[] { "CUDA_PATH", "CUDA_HOME" })
            if (Environment.GetEnvironmentVariable(variable) is string toolkit)
                folders.Add(Path.Combine(toolkit, windows ? "bin" : "lib64"));
        if (!windows) folders.Add("/usr/local/cuda/lib64");
        folders.Add(Path.GetDirectoryName(typeof(torch).Assembly.Location)!);   // and its runtimes/*/native/

        // libnvrtc.so.12, libnvrtc-5b2e1ce4.so.12 (LibTorch's), nvrtc64_120_0.dll. Shortest
        // name first: the plain library before its variants (libnvrtc.so.12.8.93,
        // nvrtc64_120_0.alt.dll).
        string pattern = windows ? "nvrtc64_*.dll" : "libnvrtc*.so*";
        foreach (var folder in folders.Where(Directory.Exists))
            foreach (var file in Directory.EnumerateFiles(folder, pattern, SearchOption.AllDirectories)
                                          .OrderBy(file => Path.GetFileName(file).Length).ThenBy(file => file))
                if (!Path.GetFileName(file).Contains("builtins") && TryLoadWithBuiltins(file, out IntPtr library))
                    return library;
        foreach (var file in windows ? new[] { "nvrtc64_120_0.dll" } : new[] { "libnvrtc.so.12", "libnvrtc.so" })
            if (NativeLibrary.TryLoad(file, out IntPtr library))
                return library;
        throw new DllNotFoundException("NVRTC, the CUDA compiler library, was not found: install the CUDA toolkit, " +
                                       "or point CUDA_PATH (or CUDA_HOME) at it");

        // NVRTC needs a second library to compile, its builtins (nvrtc-builtins64_128.dll,
        // libnvrtc-builtins.so.12.8), and loads it by name alone. Windows does not look
        // for it next to NVRTC, only on PATH, and it is not there for LibTorch's copy.
        // A library already loaded is used whatever folder it came from, so the one
        // next to NVRTC is loaded first.
        bool TryLoadWithBuiltins(string file, out IntPtr library)
        {
            string builtins = windows ? "nvrtc-builtins64_*.dll" : "libnvrtc-builtins*.so*";
            foreach (var builtinsFile in Directory.EnumerateFiles(Path.GetDirectoryName(file)!, builtins))
                NativeLibrary.TryLoad(builtinsFile, out _);
            return NativeLibrary.TryLoad(file, out library);
        }
    }

    /// The CUDA driver API: the calls used here, from cuda.h. Handles (CUcontext,
    /// CUmodule, CUfunction, CUstream) are pointers; a CUdevice is an int.
    static class Driver
    {
        [DllImport("cuda")] public static extern CudaResult cuInit(uint flags);
        [DllImport("cuda")] public static extern CudaResult cuDeviceGet(out int device, int ordinal);
        [DllImport("cuda")] public static extern CudaResult cuDeviceGetAttribute(out int value, int attribute, int device);
        [DllImport("cuda")] public static extern CudaResult cuDevicePrimaryCtxRetain(out IntPtr context, int device);
        [DllImport("cuda", EntryPoint = "cuCtxPushCurrent_v2")] public static extern CudaResult cuCtxPushCurrent(IntPtr context);
        [DllImport("cuda", EntryPoint = "cuCtxPopCurrent_v2")] public static extern CudaResult cuCtxPopCurrent(out IntPtr context);
        [DllImport("cuda")] public static extern CudaResult cuModuleLoadData(out IntPtr module, byte[] image);
        [DllImport("cuda")] public static extern CudaResult cuModuleGetFunction(out IntPtr function, IntPtr module,
                                                                                [MarshalAs(UnmanagedType.LPStr)] string name);
        [DllImport("cuda")] public static extern CudaResult cuLaunchKernel(IntPtr function, uint gridX, uint gridY, uint gridZ,
                                                                           uint blockX, uint blockY, uint blockZ,
                                                                           uint sharedMemoryBytes, IntPtr stream,
                                                                           void** kernelParameters, void** extra);
        [DllImport("cuda")] public static extern CudaResult cuGetErrorName(CudaResult error, out IntPtr name);
    }

    /// NVRTC: the calls used here, from nvrtc.h. A program (nvrtcProgram) is a pointer.
    static class Nvrtc
    {
        [DllImport("nvrtc")] public static extern NvrtcResult nvrtcCreateProgram(out IntPtr program,
                                                                                 [MarshalAs(UnmanagedType.LPUTF8Str)] string source,
                                                                                 [MarshalAs(UnmanagedType.LPUTF8Str)] string name,
                                                                                 int headerCount, IntPtr headers, IntPtr includeNames);
        [DllImport("nvrtc")] public static extern NvrtcResult nvrtcCompileProgram(IntPtr program, int optionCount,
                                                                                  [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] options);
        [DllImport("nvrtc")] public static extern NvrtcResult nvrtcGetProgramLogSize(IntPtr program, out nuint size);
        [DllImport("nvrtc")] public static extern NvrtcResult nvrtcGetProgramLog(IntPtr program, byte[] log);
        [DllImport("nvrtc")] public static extern NvrtcResult nvrtcGetCUBINSize(IntPtr program, out nuint size);
        [DllImport("nvrtc")] public static extern NvrtcResult nvrtcGetCUBIN(IntPtr program, byte[] cubin);
        [DllImport("nvrtc")] public static extern NvrtcResult nvrtcDestroyProgram(ref IntPtr program);
        [DllImport("nvrtc")] public static extern IntPtr nvrtcGetErrorString(NvrtcResult result);
    }
}
