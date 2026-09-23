// Model.cs — GPT-2, written out layer by layer.
//
//   ids (B,T) -> wte[ids] + wpe[0..T)                      token + learned position
//             -> 12 x Block:  x += Attention(LayerNorm(x))   causal, 12 heads
//                             x += Mlp(LayerNorm(x))         768 -> 3072 -> 768, GELU
//             -> LayerNorm -> h @ wteᵀ                      logits; the head is tied to wte
//
// Pre-LayerNorm, no dropout, GPT-2's init, fp32 throughout. The names match
// Hugging Face's GPT2LMHeadModel (wte, h.0.attn.c_attn, ...). Only autograd
// and the kernels come from LibTorch; the model itself is here.
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using F = TorchSharp.torch.nn.functional;

namespace Gpt2Trainer;

public sealed class CausalSelfAttention : Module<Tensor, Tensor>
{
    readonly TorchSharp.Modules.Linear c_attn, c_proj;
    readonly int nHead;

    public CausalSelfAttention(Config config) : base(nameof(CausalSelfAttention))
    {
        nHead = config.NHead;
        c_attn = Linear(config.NEmbd, 3 * config.NEmbd);   // q, k and v in one matmul
        c_proj = Linear(config.NEmbd, config.NEmbd);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        long B = x.shape[0], T = x.shape[1], C = x.shape[2];
        // (B,T,3C) -> (3, B, heads, T, head_size): q, k, v with the heads split out
        var qkv = c_attn.forward(x).view(B, T, 3, nHead, C / nHead).permute(2, 0, 3, 1, 4);
        // Fused causal attention (memory-efficient kernel): never builds the TxT matrix.
        var y = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_casual: true);
        y = y.transpose(1, 2).contiguous().view(B, T, C);   // heads back side by side
        return c_proj.forward(y);
    }
}

public sealed class Mlp : Module<Tensor, Tensor>
{
    readonly TorchSharp.Modules.Linear c_fc, c_proj;

    public Mlp(Config config) : base(nameof(Mlp))
    {
        c_fc = Linear(config.NEmbd, 4 * config.NEmbd);
        c_proj = Linear(4 * config.NEmbd, config.NEmbd);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x) =>
        c_proj.forward(F.gelu(c_fc.forward(x), TorchSharp.Modules.GELU.Approximate.tanh));   // GPT-2's tanh GELU
}

public sealed class Block : Module<Tensor, Tensor>
{
    readonly TorchSharp.Modules.LayerNorm ln_1, ln_2;
    readonly CausalSelfAttention attn;
    readonly Mlp mlp;

    public Block(Config config) : base(nameof(Block))
    {
        ln_1 = LayerNorm(config.NEmbd);
        attn = new CausalSelfAttention(config);
        ln_2 = LayerNorm(config.NEmbd);
        mlp = new Mlp(config);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        x = x + attn.forward(ln_1.forward(x));
        return x + mlp.forward(ln_2.forward(x));
    }
}

public sealed class Gpt2 : Module<Tensor, Tensor>
{
    readonly TorchSharp.Modules.Embedding wte, wpe;
    readonly ModuleList<Block> h;
    readonly TorchSharp.Modules.LayerNorm ln_f;
    public Config Config { get; }

    public Gpt2(Config config) : base(nameof(Gpt2))
    {
        Config = config;
        wte = Embedding(config.PaddedVocabSize, config.NEmbd);
        wpe = Embedding(config.SequenceLength, config.NEmbd);
        h = new ModuleList<Block>(Enumerable.Range(0, config.NLayer).Select(_ => new Block(config)).ToArray());
        ln_f = LayerNorm(config.NEmbd);
        RegisterComponents();
        InitWeights();
    }

    /// GPT-2's init: weights ~ N(0, 0.02), biases 0, LayerNorm 1 and 0. The
    /// projections back into the residual stream get 0.02 / sqrt(2 x layers),
    /// so the stream's variance doesn't grow with depth.
    void InitWeights()
    {
        using var _ = no_grad();
        double residualStd = 0.02 / Math.Sqrt(2.0 * Config.NLayer);
        foreach (var (name, parameter) in named_parameters())
        {
            if (name.EndsWith("c_proj.weight")) init.normal_(parameter, 0, residualStd);
            else if (parameter.dim() >= 2) init.normal_(parameter, 0, 0.02);
            else if (name.EndsWith("bias")) init.zeros_(parameter);
        }
    }

    /// ids (B,T) -> final hidden states (B,T,C).
    public override Tensor forward(Tensor ids)
    {
        long T = ids.shape[1];
        if (T > Config.SequenceLength)
            throw new ArgumentException($"a sequence of {T} is longer than the {Config.SequenceLength} positions");
        var x = wte.forward(ids) + wpe.forward(arange(T, dtype: ScalarType.Int64, device: ids.device));
        foreach (var block in h) x = block.forward(x);
        return ln_f.forward(x);
    }

    /// Hidden states -> logits over the padded vocabulary. The head is wte itself.
    public Tensor Logits(Tensor hidden) => F.linear(hidden, wte.weight!);

    /// Mean next-token loss of a (B,T) batch: position t predicts token t+1.
    public Tensor Loss(Tensor ids)
    {
        long T = ids.shape[1];
        var hidden = forward(ids);
        // Only T-1 positions have a next token: drop the last before the big matmul.
        var logits = Logits(hidden.narrow(1, 0, T - 1));
        var targets = ids.narrow(1, 1, T - 1);
        return F.cross_entropy(logits.reshape(-1, Config.PaddedVocabSize), targets.reshape(-1));
    }

    /// The prompt followed by maxNewTokens sampled tokens: temperature, then top-k.
    public long[] Generate(long[] prompt, int maxNewTokens, double temperature, int topK, Device device)
    {
        using var _ = no_grad();
        eval();
        var tokens = prompt.ToList();
        for (int i = 0; i < maxNewTokens; i++)
        {
            using var scope = NewDisposeScope();
            var context = tokens.Skip(Math.Max(0, tokens.Count - Config.SequenceLength)).ToArray();
            var hidden = forward(tensor(context, device: device).unsqueeze(0));
            var logits = Logits(hidden.narrow(1, context.Length - 1, 1)).reshape(-1)
                .narrow(0, 0, Config.VocabSize) / temperature;       // padding ids are never sampled
            var (values, indices) = logits.topk(Math.Min(topK, Config.VocabSize));
            var pick = values.softmax(0).multinomial(1);
            tokens.Add(indices[pick].item<long>());
        }
        train();
        return tokens.ToArray();
    }
}
