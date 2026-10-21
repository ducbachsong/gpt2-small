// linear_cublas.cuh — the same linear layer as linear.cuh, made by NVIDIA's cuBLAS library instead of our own
// kernels: the speed to measure ours against, and a stand-in the trainer can use by including this file instead.
// The same two functions, called the same way:
//
//     output = input · weightᵀ + bias        F1:  y[t][j] = sum over i of x[t][i] · W[j][i] + b[j]
//
//     B1   bias_grad   += sum over all rows of output_grad    db = dyᵀ · 1
//     B2   weight_grad += output_gradᵀ · input                dW = dyᵀ x
//     B3   input_grad   = output_grad · weight                dx = dy W
//
// with the shapes, names and steps of linear.cuh's head comment. Include one of the two files, not both: they
// define the same functions.
//
// Each step is one cuBLAS call. On a T4 the test finds the products the same as PyTorch's to the last bit, so
// the same cuBLAS kernels; the bias is added in another order (output with bias differs by about 1e-6, without
// by 0), and PyTorch adds up bias_grad with a sum kernel of its own instead of B1's call:
//
//     step   cuBLAS call                                         what it does
//     F1     cublasLtMatmul, with the bias as its "epilogue"     the product, then + bias as each value is stored
//     B3     cublasSgemm, beta = 0                               the product, written over input_grad
//     B2     cublasSgemm, beta = 1                               the product, added onto weight_grad
//     B1     cublasSgemv, with a vector of 1s, beta = 1          each column of dy added up, onto bias_grad
//
// cuBLAS works out C = alpha · op(A) · op(B) + beta · C, op turning a matrix round or not; beta = 0 writes C
// over, beta = 1 adds onto it. cuBLASLt ("Lt", light) is the newer half of the library: the same products, plus
// a few small steps it can do while storing the result, such as adding a bias. Without it the bias would need
// a second pass over output, reading and writing its 37.7 MB again for the qkv.
//
// ── cuBLAS reads matrices by columns ─────────────────────────────────────────────
// cuBLAS comes from Fortran, which stores a matrix column after column. We store row after row. The same
// memory read the other way is the matrix turned round (transposed):
//
//     memory                    as rows {2, 3} (ours)     as columns {3, 2} (cuBLAS's)
//     a b c d e f               a b c                     a d
//                               d e f                     b e
//                                                         c f
//
// So every matrix we hand cuBLAS, it sees turned round, and its result we read turned round. Since
// (A · B)ᵀ = Bᵀ · Aᵀ, asking cuBLAS for the product of the turned matrices the other way round gives, read by
// rows, exactly ours. For F1 (qkv: rows 4096, in_features 768, out_features 2304):
//
//     we want         y {4096, 2304} = x {4096, 768} · Wᵀ                 (W is {2304, 768})
//     cuBLAS sees     x as xᵀ {768, 4096}, W as Wᵀ {768, 2304}, and we read its C as yᵀ {2304, 4096}
//     so we ask for   yᵀ = W · xᵀ = op(Wᵀ as it sees it) · (xᵀ as it sees it), with op = turn round
//
// The four steps that way (m x n is C's size as cuBLAS sees it, k the summed index; "lda" is how many floats
// one column of A takes in memory, which for our row-major matrices is a row's length):
//
//     step   C as cuBLAS sees it          = A, op         · B, op          m             n             k
//     F1     yᵀ  {out, rows}, ld out      = W  turned     · x  as is       out_features  rows          in_features
//     B3     dxᵀ {in, rows},  ld in       = W  as is      · dy as is       in_features   rows          out_features
//     B2     dWᵀ {in, out},   ld in       = x  as is      · dy turned      in_features   out_features  rows
//     B1     db  {out}                    = dy as is      · 1s             out_features  rows          –
#ifndef LINEAR_CUBLAS_CUH
#define LINEAR_CUBLAS_CUH

#include <limits.h>
#include <vector>
#include <cublas_v2.h>
#include <cublasLt.h>
#include "cuda_common.h"             // cudaCheck only: the rest is written out below
#include "../tensor.cuh"             // forward and backward take tensors: each knows its shape

// Room cuBLASLt may use while it works, for products it splits into parts and adds up at the end. 32 MB, as
// llm.c gives it; it is made once and kept.
#define LINEAR_CUBLAS_WORKSPACE_BYTES (32 * 1024 * 1024)

// ── forward and backward ──────────────────────────────────────────────────────
void linear_forward(const Tensor& output, const Tensor& input, const Tensor& weight, const Tensor& bias);
void linear_backward(const Tensor& input_grad, const Tensor& weight_grad, const Tensor& bias_grad,
                     const Tensor& output_grad, const Tensor& input, const Tensor& weight);

// ── what both use ─────────────────────────────────────────────────────────────

/// What every call needs and is made only once: the two handles (cuBLAS's record of the GPU it runs on and its
/// settings; making one takes milliseconds), cuBLASLt's workspace, and the 1s B1 multiplies by. They stay
/// until the program ends.
struct LinearCublas {
    cublasHandle_t handle;           // for cublasSgemm and cublasSgemv
    cublasLtHandle_t lt_handle;      // for cublasLtMatmul
    void* workspace;                 // LINEAR_CUBLAS_WORKSPACE_BYTES on the GPU, for cublasLtMatmul
    float* ones;                     // ones_length 1s on the GPU, for B1
    size_t ones_length;
};

LinearCublas& linear_cublas();                                       // the handles, made on first use
const float* linear_cublas_ones(size_t length);                      // a vector of at least `length` 1s
uint32_t linear_cublas_alignment(const void* pointer);               // the bytes `pointer` is a multiple of

// Every cuBLAS call returns a status; this stops with the file and line, as cudaCheck does.
static inline void cublas_check(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error at %s:%d: %s\n", file, line, cublasGetStatusString(status));
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(call) cublas_check((call), __FILE__, __LINE__)

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward ───────────────────────────────────────────────────────────────────

/// Writes output = input · weightᵀ + bias (F1) with one cuBLASLt call, which adds the bias as it stores each
/// value. input is {..., in_features} ({batch_size, seq_len, in_features} in GPT-2), output the same with
/// out_features last, weight {out_features, in_features}, bias {out_features} or Tensor() for none (the head).
///
/// First it stops with a message if the tensors do not fit together. Unlike linear.cuh, any sizes and any
/// start addresses work: cuBLAS picks a kernel that fits them.
///
/// A cuBLASLt call is described, not just made: a description of the product (its number types, which inputs
/// are turned round, the bias), one of each matrix's layout in memory, and how much workspace it may use. From
/// these, cuBLASLt's "heuristic" picks the kernel it expects to be fastest. The descriptions are made and freed
/// on every call, as PyTorch does: a few microseconds, against 0.6 ms for the smallest GPT-2 forward.
inline void linear_forward(const Tensor& output, const Tensor& input, const Tensor& weight, const Tensor& bias) {
    bool has_bias = bias.data() != nullptr;                     // Tensor() has no values: no bias
    bool fits = input.dim() >= 1 && weight.dim() == 2 && output.dim() == input.dim() &&
                input.size(input.dim() - 1) == weight.size(1) && output.size(output.dim() - 1) == weight.size(0) &&
                (!has_bias || bias.shape() == Shape({weight.size(0)}));
    for (int d = 0; fits && d < input.dim() - 1; d++) fits = output.size(d) == input.size(d);   // the same rows
    if (!fits) {
        fprintf(stderr, "error: a linear layer needs input {..., in_features}, output {..., out_features} with the "
                        "same rows, weight {out_features, in_features}, bias {out_features} or Tensor()\n");
        exit(EXIT_FAILURE);
    }
    size_t out_features = weight.size(0), in_features = weight.size(1), input_rows = input.numel() / in_features;
    if (input_rows > INT_MAX || in_features > INT_MAX || out_features > INT_MAX) {     // cuBLAS takes int sizes
        fprintf(stderr, "error: a linear layer's sizes must each be below 2^31; got %zu rows, in_features %zu, "
                        "out_features %zu\n", input_rows, in_features, out_features);
        exit(EXIT_FAILURE);
    }
    LinearCublas& cublas = linear_cublas();

    // The product: float numbers, float sums. A (weight) turned round, B (input) as it is; see the table at
    // the top, row F1. With a bias, the epilogue adds bias[j] to every value of C's row j as cuBLAS sees it,
    // which is our output column j.
    cublasLtMatmulDesc_t product;
    cublasCheck(cublasLtMatmulDescCreate(&product, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t turned = CUBLAS_OP_T, as_is = CUBLAS_OP_N;
    cublasCheck(cublasLtMatmulDescSetAttribute(product, CUBLASLT_MATMUL_DESC_TRANSA, &turned, sizeof(turned)));
    cublasCheck(cublasLtMatmulDescSetAttribute(product, CUBLASLT_MATMUL_DESC_TRANSB, &as_is, sizeof(as_is)));
    if (has_bias) {
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        const float* bias_values = bias.data();
        cublasCheck(cublasLtMatmulDescSetAttribute(product, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
                                                   sizeof(epilogue)));
        cublasCheck(cublasLtMatmulDescSetAttribute(product, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_values,
                                                   sizeof(bias_values)));
    }

    // Each matrix as cuBLAS sees it in memory, before turning: rows, columns, and the floats one column takes.
    // For the qkv: weight {768, 2304}, input {768, 4096}, output {2304, 4096}.
    cublasLtMatrixLayout_t weight_layout, input_layout, output_layout;
    cublasCheck(cublasLtMatrixLayoutCreate(&weight_layout, CUDA_R_32F, in_features, out_features, in_features));
    cublasCheck(cublasLtMatrixLayoutCreate(&input_layout, CUDA_R_32F, in_features, input_rows, in_features));
    cublasCheck(cublasLtMatrixLayoutCreate(&output_layout, CUDA_R_32F, out_features, input_rows, out_features));

    // What the kernel may count on. cuBLASLt assumes every matrix starts on 256 bytes unless told otherwise, and
    // may then pick a kernel that reads 16 bytes at a time: on a view that starts 4 bytes in, that kernel would
    // fail. So it is told each matrix's real alignment, as PyTorch does. Each setting is passed as a pointer and
    // a size, and cuBLASLt checks the size: the workspace is a uint64_t, the alignments uint32_t ("an unsupported
    // value or parameter" when passed as 8 bytes).
    cublasLtMatmulPreference_t preference;
    cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
    uint64_t workspace_bytes = LINEAR_CUBLAS_WORKSPACE_BYTES;
    uint32_t weight_alignment = linear_cublas_alignment(weight.data());
    uint32_t input_alignment = linear_cublas_alignment(input.data());
    uint32_t output_alignment = linear_cublas_alignment(output.data());
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                     &workspace_bytes, sizeof(workspace_bytes)));
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                                                     &weight_alignment, sizeof(weight_alignment)));
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                                                     &input_alignment, sizeof(input_alignment)));
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                                                     &output_alignment, sizeof(output_alignment)));
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
                                                     &output_alignment, sizeof(output_alignment)));

    // The heuristic's best kernel for these sizes, then the call: C and D are both output, beta = 0, so output
    // is written over (C is what beta multiplies, D where the result goes).
    cublasLtMatmulHeuristicResult_t best;
    int found = 0;
    cublasCheck(cublasLtMatmulAlgoGetHeuristic(cublas.lt_handle, product, weight_layout, input_layout, output_layout,
                                               output_layout, preference, 1, &best, &found));
    if (found == 0) {
        fprintf(stderr, "error: cuBLASLt has no kernel for a linear layer of %zu rows, in_features %zu, "
                        "out_features %zu%s\n", input_rows, in_features, out_features, has_bias ? ", with bias" : "");
        exit(EXIT_FAILURE);
    }
    float one = 1.0f, zero = 0.0f;
    cublasCheck(cublasLtMatmul(cublas.lt_handle, product, &one, weight.data(), weight_layout, input.data(),
                               input_layout, &zero, output.data(), output_layout, output.data(), output_layout,
                               &best.algo, cublas.workspace, LINEAR_CUBLAS_WORKSPACE_BYTES, 0));

    cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
    cublasCheck(cublasLtMatrixLayoutDestroy(output_layout));
    cublasCheck(cublasLtMatrixLayoutDestroy(input_layout));
    cublasCheck(cublasLtMatrixLayoutDestroy(weight_layout));
    cublasCheck(cublasLtMatmulDescDestroy(product));
}

// ── backward ──────────────────────────────────────────────────────────────────

/// Works out the gradients of input, weight and bias (B3, B2, B1) from output_grad and what the forward read:
/// input and weight, with three cuBLAS calls. As linear.cuh's: input_grad is written over; weight_grad and
/// bias_grad are added onto, so they must be 0 before the first backward of a step. The shapes are the
/// forward's: input_grad as input, weight_grad as weight, bias_grad as bias (Tensor() for none), output_grad as
/// output. It stops with a message if the tensors do not fit together.
///
/// The classic cuBLAS calls, not cuBLASLt: they need no description, add onto C with beta = 1, and take any
/// start address. For the qkv, B2 is a product of 2304 x 768 values, each a sum over the 4096 rows; cuBLAS
/// splits that sum into parts when it fills the GPU better (5, for the qkv on a T4), which our kernel does not.
inline void linear_backward(const Tensor& input_grad, const Tensor& weight_grad, const Tensor& bias_grad,
                            const Tensor& output_grad, const Tensor& input, const Tensor& weight) {
    bool has_bias = bias_grad.data() != nullptr;                // Tensor() has no values: no bias
    bool fits = input.dim() >= 1 && weight.dim() == 2 && output_grad.dim() == input.dim() &&
                input.size(input.dim() - 1) == weight.size(1) &&
                output_grad.size(output_grad.dim() - 1) == weight.size(0) && input_grad.shape() == input.shape() &&
                weight_grad.shape() == weight.shape() && (!has_bias || bias_grad.shape() == Shape({weight.size(0)}));
    for (int d = 0; fits && d < input.dim() - 1; d++) fits = output_grad.size(d) == input.size(d);   // the same rows
    if (!fits) {
        fprintf(stderr, "error: a linear layer's backward needs input and input_grad {..., in_features}, output_grad "
                        "{..., out_features} with the same rows, weight and weight_grad {out_features, in_features}, "
                        "bias_grad {out_features} or Tensor()\n");
        exit(EXIT_FAILURE);
    }
    size_t out_features = weight.size(0), in_features = weight.size(1), input_rows = input.numel() / in_features;
    if (input_rows > INT_MAX || in_features > INT_MAX || out_features > INT_MAX) {     // cuBLAS takes int sizes
        fprintf(stderr, "error: a linear layer's sizes must each be below 2^31; got %zu rows, in_features %zu, "
                        "out_features %zu\n", input_rows, in_features, out_features);
        exit(EXIT_FAILURE);
    }
    int rows = (int)input_rows, in = (int)in_features, out = (int)out_features;
    LinearCublas& cublas = linear_cublas();
    float one = 1.0f, zero = 0.0f;

    // B3: dxᵀ {in, rows} = W {in, out as cuBLAS sees it} · dyᵀ {out, rows}, written over input_grad (beta 0)
    cublasCheck(cublasSgemm(cublas.handle, CUBLAS_OP_N, CUBLAS_OP_N, in, rows, out, &one, weight.data(), in,
                            output_grad.data(), out, &zero, input_grad.data(), in));
    // B2: dWᵀ {in, out} = xᵀ {in, rows} · (dyᵀ {out, rows}) turned, added onto weight_grad (beta 1)
    cublasCheck(cublasSgemm(cublas.handle, CUBLAS_OP_N, CUBLAS_OP_T, in, out, rows, &one, input.data(), in,
                            output_grad.data(), out, &one, weight_grad.data(), in));
    // B1: db {out} = dyᵀ {out, rows} · a column of `rows` 1s: each output column of dy added up, onto bias_grad
    if (has_bias)
        cublasCheck(cublasSgemv(cublas.handle, CUBLAS_OP_N, out, rows, &one, output_grad.data(), out,
                                linear_cublas_ones(input_rows), 1, &one, bias_grad.data(), 1));
}

// ── what both use ─────────────────────────────────────────────────────────────

/// Gives the handles and workspace every call uses, making them on the first call. Both handles run their
/// work on the default stream, in order with our own kernels.
inline LinearCublas& linear_cublas() {
    static LinearCublas cublas = [] {
        LinearCublas made = {};
        cublasCheck(cublasCreate(&made.handle));
        cublasCheck(cublasLtCreate(&made.lt_handle));
        cudaCheck(cudaMalloc(&made.workspace, LINEAR_CUBLAS_WORKSPACE_BYTES));
        return made;
    }();
    return cublas;
}

/// Gives a vector of at least `length` 1s on the GPU, for B1's sums: dy times 1s adds up each column of dy.
/// Made on first use and made again, longer, when a call needs more: 4096 floats, 16 KB, for GPT-2's rows.
inline const float* linear_cublas_ones(size_t length) {
    LinearCublas& cublas = linear_cublas();
    if (cublas.ones_length < length) {
        if (cublas.ones != nullptr) cudaCheck(cudaFree(cublas.ones));  // waits for the GPU to finish with it
        std::vector<float> ones(length, 1.0f);
        cudaCheck(cudaMalloc(&cublas.ones, length * sizeof(float)));
        cudaCheck(cudaMemcpy(cublas.ones, ones.data(), length * sizeof(float), cudaMemcpyHostToDevice));
        cublas.ones_length = length;
    }
    return cublas.ones;
}

/// Gives the largest power of 2, up to 256, that `pointer`'s address is a multiple of: 256 for cudaMalloc's own
/// memory, 4 for a view that starts one float in. What cuBLASLt is told it may count on.
inline uint32_t linear_cublas_alignment(const void* pointer) {
    uint64_t address = (uint64_t)pointer;
    uint32_t alignment = 256;
    while (alignment > 1 && address % alignment != 0) alignment /= 2;
    return alignment;
}

#endif // LINEAR_CUBLAS_CUH
