// linear_cublas.cuh — linear.cuh's linear layer done by NVIDIA's cuBLAS: the speed to measure ours against.
// The same two functions; include one of the two files, not both.
//     forward:    output = input · weightᵀ + bias          cublasLtMatmul, bias added as each value is stored
//     backward:   input_grad   = output_grad · weight      gemm, beta 0: written over
//                 weight_grad += output_gradᵀ · input      gemm, beta 1: added onto
//                 bias_grad   += sum over rows of output_grad   gemv (gemm in bfloat16) with a vector of 1s
//
// cuBLAS reads matrices by columns, we store them by rows: it sees each of ours turned round (transposed).
// Since (A · B)ᵀ = Bᵀ · Aᵀ, we ask for the turned result with the inputs swapped:
//     step      C as cuBLAS sees it   = A, op         · B, op          m             n             k
//     forward   yᵀ  {out, rows}       = W  turned     · x  as is       out_features  rows          in_features
//     dx        dxᵀ {in, rows}        = W  as is      · dy as is       in_features   rows          out_features
//     dW        dWᵀ {in, out}         = x  as is      · dy turned      in_features   out_features  rows
//     db        db  {out}             = dy as is      · 1s             out_features  rows          –
// All float or all bfloat16; the sums always in float (CUBLAS_COMPUTE_32F). cuBLAS picks its own launches, so
// these two take no block size.
#ifndef LINEAR_CUBLAS_CUH
#define LINEAR_CUBLAS_CUH

#include <limits.h>
#include <vector>
#include <cublas_v2.h>
#include <cublasLt.h>
#include "cuda_common.h"             // cudaCheck only
#include "../tensor.cuh"

#define LINEAR_CUBLAS_WORKSPACE_BYTES (32 * 1024 * 1024)   // room cuBLASLt may use, as llm.c gives it
#define LINEAR_CUBLAS_LARGEST_ALIGNMENT 256                // the alignment cuBLASLt assumes unless told: cudaMalloc's

// ── forward and backward ──────────────────────────────────────────────────────
void linear_forward(const Tensor& output_tensor, const Tensor& input_tensor, const Tensor& weight_tensor,
                    const Tensor& bias_tensor);
void linear_backward(const Tensor& input_grad_tensor, const Tensor& weight_grad_tensor, const Tensor& bias_grad_tensor,
                     const Tensor& output_grad_tensor, const Tensor& input_tensor, const Tensor& weight_tensor);

// ── what both use ─────────────────────────────────────────────────────────────

/// What every call needs, made once and kept until the program ends.
struct LinearCublas {
    cublasHandle_t handle_ptr;            // for cublasSgemm, cublasSgemv and cublasGemmEx
    cublasLtHandle_t lt_handle_ptr;       // for cublasLtMatmul
    void* workspace_ptr;                  // LINEAR_CUBLAS_WORKSPACE_BYTES on the GPU, for cublasLtMatmul
    float* ones_ptr;                      // ones_length_val 1s on the GPU, for bias_grad
    size_t ones_length_val;
    nv_bfloat16* bfloat16_ones_ptr;       // the same as bfloat16s
    size_t bfloat16_ones_length_val;
};

LinearCublas& linear_cublas();                                              // the handles, made on first use
const float* linear_cublas_ones(size_t length_val);                         // a vector of at least `length_val` 1s
const nv_bfloat16* linear_cublas_bfloat16_ones(size_t length_val);          // the same, as bfloat16s
cudaDataType_t linear_cublas_number_type(const Tensor& tensor);             // CUDA_R_32F or CUDA_R_16BF
uint32_t linear_cublas_alignment(const void* address_ptr);                  // the bytes the address is a multiple of

/// Stops with the file and line if a cuBLAS call failed, as cudaCheck does.
///
///   status_val            what the call returned
///   file_ptr, line_val    where the call is
static inline void cublas_check(cublasStatus_t status_val, const char* file_ptr, int line_val) {
    if (status_val != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error at %s:%d: %s\n", file_ptr, line_val, cublasGetStatusString(status_val));
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(call) cublas_check((call), __FILE__, __LINE__)

// ══ the code, in the order of the list above ══════════════════════════════════

// ── forward ───────────────────────────────────────────────────────────────────

/// Writes output = input · weightᵀ + bias with one cuBLASLt call: checks, describes the product and the
/// matrices, lets cuBLASLt pick its fastest kernel, then runs it. Any sizes and start addresses work.
///
///   output_tensor   {..., out_features}
///   input_tensor    {..., in_features}: {batch_size, seq_len, in_features} in GPT-2
///   weight_tensor   {out_features, in_features}
///   bias_tensor     {out_features}, or Tensor() for none
inline void linear_forward(const Tensor& output_tensor, const Tensor& input_tensor, const Tensor& weight_tensor,
                           const Tensor& bias_tensor) {
    // 1. check the tensors; any of them wrong: stop with an error
    // shapes: the same rows, the features matching weight
    bool has_bias_val = bias_tensor.data_ptr() != nullptr;
    bool fits_val = input_tensor.dim() >= 1 && weight_tensor.dim() == 2 && output_tensor.dim() == input_tensor.dim() &&
                    input_tensor.size(input_tensor.dim() - 1) == weight_tensor.size(1) &&
                    output_tensor.size(output_tensor.dim() - 1) == weight_tensor.size(0) &&
                    (!has_bias_val || bias_tensor.shape() == Shape({weight_tensor.size(0)}));
    for (int dim_val = 0; fits_val && dim_val < input_tensor.dim() - 1; dim_val++)
        fits_val = output_tensor.size(dim_val) == input_tensor.size(dim_val);
    if (!fits_val) {
        fprintf(stderr, "error: a linear layer needs input {..., in_features}, output {..., out_features} with the "
                        "same rows, weight {out_features, in_features}, bias {out_features} or Tensor()\n");
        exit(EXIT_FAILURE);
    }
    // types: all float or all bfloat16
    DType dtype_val = input_tensor.dtype();
    if ((dtype_val != DType::Float32 && dtype_val != DType::BFloat16) || output_tensor.dtype() != dtype_val ||
        weight_tensor.dtype() != dtype_val || (has_bias_val && bias_tensor.dtype() != dtype_val)) {
        fprintf(stderr, "error: a linear layer needs output, input, weight and bias all float or all bfloat16\n");
        exit(EXIT_FAILURE);
    }
    // sizes: cuBLAS takes ints
    size_t out_features_val = weight_tensor.size(0);
    size_t in_features_val = weight_tensor.size(1);
    size_t input_rows_val = input_tensor.numel() / in_features_val;
    if (input_rows_val > INT_MAX || in_features_val > INT_MAX || out_features_val > INT_MAX) {
        fprintf(stderr, "error: a linear layer's sizes must each be below 2^31; got %zu rows, in_features %zu, "
                        "out_features %zu\n", input_rows_val, in_features_val, out_features_val);
        exit(EXIT_FAILURE);
    }
    LinearCublas& cublas_val = linear_cublas();
    cudaDataType_t number_type_val = linear_cublas_number_type(input_tensor);

    // 2. the product: float sums; weight turned, input as is; the bias added by the epilogue
    cublasLtMatmulDesc_t product_ptr;
    cublasCheck(cublasLtMatmulDescCreate(&product_ptr, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t turned_val = CUBLAS_OP_T;
    cublasOperation_t as_is_val = CUBLAS_OP_N;
    cublasCheck(cublasLtMatmulDescSetAttribute(product_ptr, CUBLASLT_MATMUL_DESC_TRANSA, &turned_val, sizeof(turned_val)));
    cublasCheck(cublasLtMatmulDescSetAttribute(product_ptr, CUBLASLT_MATMUL_DESC_TRANSB, &as_is_val, sizeof(as_is_val)));
    if (has_bias_val) {
        cublasLtEpilogue_t epilogue_val = CUBLASLT_EPILOGUE_BIAS;
        const void* bias_values_ptr = bias_tensor.data_ptr();
        cublasCheck(cublasLtMatmulDescSetAttribute(product_ptr, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue_val,
                                                   sizeof(epilogue_val)));
        cublasCheck(cublasLtMatmulDescSetAttribute(product_ptr, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_values_ptr,
                                                   sizeof(bias_values_ptr)));
    }

    // 3. each matrix as cuBLAS sees it: rows, columns, values per column
    cublasLtMatrixLayout_t weight_layout_ptr;
    cublasLtMatrixLayout_t input_layout_ptr;
    cublasLtMatrixLayout_t output_layout_ptr;
    cublasCheck(cublasLtMatrixLayoutCreate(&weight_layout_ptr, number_type_val, in_features_val, out_features_val, in_features_val));
    cublasCheck(cublasLtMatrixLayoutCreate(&input_layout_ptr, number_type_val, in_features_val, input_rows_val, in_features_val));
    cublasCheck(cublasLtMatrixLayoutCreate(&output_layout_ptr, number_type_val, out_features_val, input_rows_val, out_features_val));

    // 4. the workspace and each matrix's real alignment (cuBLASLt assumes LINEAR_CUBLAS_LARGEST_ALIGNMENT otherwise)
    cublasLtMatmulPreference_t preference_ptr;
    cublasCheck(cublasLtMatmulPreferenceCreate(&preference_ptr));
    uint64_t workspace_bytes_val = LINEAR_CUBLAS_WORKSPACE_BYTES;
    uint32_t weight_alignment_val = linear_cublas_alignment(weight_tensor.data_ptr());
    uint32_t input_alignment_val = linear_cublas_alignment(input_tensor.data_ptr());
    uint32_t output_alignment_val = linear_cublas_alignment(output_tensor.data_ptr());
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference_ptr, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                     &workspace_bytes_val, sizeof(workspace_bytes_val)));
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference_ptr, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                                                     &weight_alignment_val, sizeof(weight_alignment_val)));
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference_ptr, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                                                     &input_alignment_val, sizeof(input_alignment_val)));
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference_ptr, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                                                     &output_alignment_val, sizeof(output_alignment_val)));
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference_ptr, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
                                                     &output_alignment_val, sizeof(output_alignment_val)));

    // 5. the fastest kernel, then the call: beta 0, output written over
    cublasLtMatmulHeuristicResult_t best_val;
    int found_val = 0;
    cublasCheck(cublasLtMatmulAlgoGetHeuristic(cublas_val.lt_handle_ptr, product_ptr, weight_layout_ptr, input_layout_ptr,
                                               output_layout_ptr, output_layout_ptr, preference_ptr, 1, &best_val,
                                               &found_val));
    if (found_val == 0) {
        fprintf(stderr, "error: cuBLASLt has no kernel for a linear layer of %zu rows, in_features %zu, "
                        "out_features %zu%s\n", input_rows_val, in_features_val, out_features_val,
                has_bias_val ? ", with bias" : "");
        exit(EXIT_FAILURE);
    }
    float one_val = 1.0f;
    float zero_val = 0.0f;
    cublasCheck(cublasLtMatmul(cublas_val.lt_handle_ptr, product_ptr, &one_val, weight_tensor.data_ptr(),
                               weight_layout_ptr, input_tensor.data_ptr(), input_layout_ptr, &zero_val,
                               output_tensor.data_ptr(), output_layout_ptr, output_tensor.data_ptr(), output_layout_ptr,
                               &best_val.algo, cublas_val.workspace_ptr, LINEAR_CUBLAS_WORKSPACE_BYTES, 0));

    // 6. free the descriptions
    cublasCheck(cublasLtMatmulPreferenceDestroy(preference_ptr));
    cublasCheck(cublasLtMatrixLayoutDestroy(output_layout_ptr));
    cublasCheck(cublasLtMatrixLayoutDestroy(input_layout_ptr));
    cublasCheck(cublasLtMatrixLayoutDestroy(weight_layout_ptr));
    cublasCheck(cublasLtMatmulDescDestroy(product_ptr));
}

// ── backward ──────────────────────────────────────────────────────────────────

/// Works out the gradients of input, weight and bias with three cuBLAS calls. input_grad is written over;
/// weight_grad and bias_grad are added onto, so they must be 0 before a step's first backward.
///
///   input_grad_tensor               input's shape
///   weight_grad_tensor              weight's shape
///   bias_grad_tensor                {out_features}, or Tensor() for none
///   output_grad_tensor              the gradient from the layer above
///   input_tensor, weight_tensor     the forward's
inline void linear_backward(const Tensor& input_grad_tensor, const Tensor& weight_grad_tensor, const Tensor& bias_grad_tensor,
                            const Tensor& output_grad_tensor, const Tensor& input_tensor, const Tensor& weight_tensor) {
    // 1. check the tensors; any of them wrong: stop with an error
    // shapes: the same rows, the features matching weight, each grad its tensor's shape
    bool has_bias_val = bias_grad_tensor.data_ptr() != nullptr;
    bool fits_val = input_tensor.dim() >= 1 && weight_tensor.dim() == 2 && output_grad_tensor.dim() == input_tensor.dim() &&
                    input_tensor.size(input_tensor.dim() - 1) == weight_tensor.size(1) &&
                    output_grad_tensor.size(output_grad_tensor.dim() - 1) == weight_tensor.size(0) &&
                    input_grad_tensor.shape() == input_tensor.shape() && weight_grad_tensor.shape() == weight_tensor.shape() &&
                    (!has_bias_val || bias_grad_tensor.shape() == Shape({weight_tensor.size(0)}));
    for (int dim_val = 0; fits_val && dim_val < input_tensor.dim() - 1; dim_val++)
        fits_val = output_grad_tensor.size(dim_val) == input_tensor.size(dim_val);
    if (!fits_val) {
        fprintf(stderr, "error: a linear layer's backward needs input and input_grad {..., in_features}, output_grad "
                        "{..., out_features} with the same rows, weight and weight_grad {out_features, in_features}, "
                        "bias_grad {out_features} or Tensor()\n");
        exit(EXIT_FAILURE);
    }
    // sizes: cuBLAS takes ints
    size_t out_features_val = weight_tensor.size(0);
    size_t in_features_val = weight_tensor.size(1);
    size_t input_rows_val = input_tensor.numel() / in_features_val;
    if (input_rows_val > INT_MAX || in_features_val > INT_MAX || out_features_val > INT_MAX) {
        fprintf(stderr, "error: a linear layer's sizes must each be below 2^31; got %zu rows, in_features %zu, "
                        "out_features %zu\n", input_rows_val, in_features_val, out_features_val);
        exit(EXIT_FAILURE);
    }
    // types: all float or all bfloat16
    DType dtype_val = input_tensor.dtype();
    if ((dtype_val != DType::Float32 && dtype_val != DType::BFloat16) || input_grad_tensor.dtype() != dtype_val ||
        weight_grad_tensor.dtype() != dtype_val || output_grad_tensor.dtype() != dtype_val ||
        weight_tensor.dtype() != dtype_val || (has_bias_val && bias_grad_tensor.dtype() != dtype_val)) {
        fprintf(stderr, "error: a linear layer's backward needs its tensors all float or all bfloat16\n");
        exit(EXIT_FAILURE);
    }
    int rows_val = (int)input_rows_val;
    int in_val = (int)in_features_val;
    int out_val = (int)out_features_val;
    LinearCublas& cublas_val = linear_cublas();
    float one_val = 1.0f;
    float zero_val = 0.0f;

    // 2. bfloat16: the three products by cublasGemmEx, bfloat16 matrices, float sums
    if (dtype_val == DType::BFloat16) {
        cudaDataType_t bf16_val = CUDA_R_16BF;
        // dxᵀ = W · dyᵀ, written over input_grad
        cublasCheck(cublasGemmEx(cublas_val.handle_ptr, CUBLAS_OP_N, CUBLAS_OP_N, in_val, rows_val, out_val, &one_val,
                                 weight_tensor.data_ptr(), bf16_val, in_val, output_grad_tensor.data_ptr(), bf16_val, out_val,
                                 &zero_val, input_grad_tensor.data_ptr(), bf16_val, in_val, CUBLAS_COMPUTE_32F,
                                 CUBLAS_GEMM_DEFAULT));
        // dWᵀ = xᵀ · dy, added onto weight_grad
        cublasCheck(cublasGemmEx(cublas_val.handle_ptr, CUBLAS_OP_N, CUBLAS_OP_T, in_val, out_val, rows_val, &one_val,
                                 input_tensor.data_ptr(), bf16_val, in_val, output_grad_tensor.data_ptr(), bf16_val, out_val,
                                 &one_val, weight_grad_tensor.data_ptr(), bf16_val, in_val, CUBLAS_COMPUTE_32F,
                                 CUBLAS_GEMM_DEFAULT));
        // db = dyᵀ · 1s, added onto bias_grad
        if (has_bias_val)
            cublasCheck(cublasGemmEx(cublas_val.handle_ptr, CUBLAS_OP_N, CUBLAS_OP_N, out_val, 1, rows_val, &one_val,
                                     output_grad_tensor.data_ptr(), bf16_val, out_val,
                                     linear_cublas_bfloat16_ones(input_rows_val), bf16_val, rows_val, &one_val,
                                     bias_grad_tensor.data_ptr(), bf16_val, out_val, CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT));
        return;
    }

    // 3. float: cublasSgemm twice and cublasSgemv
    // dxᵀ = W · dyᵀ, written over input_grad
    cublasCheck(cublasSgemm(cublas_val.handle_ptr, CUBLAS_OP_N, CUBLAS_OP_N, in_val, rows_val, out_val, &one_val,
                            weight_tensor.data(), in_val, output_grad_tensor.data(), out_val, &zero_val,
                            input_grad_tensor.data(), in_val));
    // dWᵀ = xᵀ · dy, added onto weight_grad
    cublasCheck(cublasSgemm(cublas_val.handle_ptr, CUBLAS_OP_N, CUBLAS_OP_T, in_val, out_val, rows_val, &one_val,
                            input_tensor.data(), in_val, output_grad_tensor.data(), out_val, &one_val,
                            weight_grad_tensor.data(), in_val));
    // db = dyᵀ · 1s, added onto bias_grad
    if (has_bias_val)
        cublasCheck(cublasSgemv(cublas_val.handle_ptr, CUBLAS_OP_N, out_val, rows_val, &one_val, output_grad_tensor.data(),
                                out_val, linear_cublas_ones(input_rows_val), 1, &one_val, bias_grad_tensor.data(), 1));
}

// ── what both use ─────────────────────────────────────────────────────────────

/// Gives the handles and workspace, made on the first call. Both run on the default stream, in order with our
/// kernels.
inline LinearCublas& linear_cublas() {
    static LinearCublas cublas_val = [] {
        LinearCublas made_val = {};
        cublasCheck(cublasCreate(&made_val.handle_ptr));
        cublasCheck(cublasLtCreate(&made_val.lt_handle_ptr));
        cudaCheck(cudaMalloc(&made_val.workspace_ptr, LINEAR_CUBLAS_WORKSPACE_BYTES));
        return made_val;
    }();
    return cublas_val;
}

/// Gives a vector of at least `length_val` 1s on the GPU, for bias_grad's sums; made again, longer, when needed.
///
///   length_val   how many 1s
inline const float* linear_cublas_ones(size_t length_val) {
    LinearCublas& cublas_val = linear_cublas();
    if (cublas_val.ones_length_val < length_val) {
        if (cublas_val.ones_ptr != nullptr) cudaCheck(cudaFree(cublas_val.ones_ptr));  // waits for the GPU to finish with it
        std::vector<float> ones_list(length_val, 1.0f);
        cudaCheck(cudaMalloc(&cublas_val.ones_ptr, length_val * sizeof(float)));
        cudaCheck(cudaMemcpy(cublas_val.ones_ptr, ones_list.data(), length_val * sizeof(float), cudaMemcpyHostToDevice));
        cublas_val.ones_length_val = length_val;
    }
    return cublas_val.ones_ptr;
}

/// Gives a vector of at least `length_val` 1s as bfloat16s on the GPU, as linear_cublas_ones.
///
///   length_val   how many 1s
inline const nv_bfloat16* linear_cublas_bfloat16_ones(size_t length_val) {
    LinearCublas& cublas_val = linear_cublas();
    if (cublas_val.bfloat16_ones_length_val < length_val) {
        if (cublas_val.bfloat16_ones_ptr != nullptr) cudaCheck(cudaFree(cublas_val.bfloat16_ones_ptr));
        std::vector<nv_bfloat16> ones_list(length_val, from_float<nv_bfloat16>(1.0f));
        cudaCheck(cudaMalloc(&cublas_val.bfloat16_ones_ptr, length_val * sizeof(nv_bfloat16)));
        cudaCheck(cudaMemcpy(cublas_val.bfloat16_ones_ptr, ones_list.data(), length_val * sizeof(nv_bfloat16),
                             cudaMemcpyHostToDevice));
        cublas_val.bfloat16_ones_length_val = length_val;
    }
    return cublas_val.bfloat16_ones_ptr;
}

/// Gives the number type cuBLAS is told for a tensor.
///
///   tensor   float: CUDA_R_32F; bfloat16: CUDA_R_16BF
inline cudaDataType_t linear_cublas_number_type(const Tensor& tensor) {
    return tensor.dtype() == DType::BFloat16 ? CUDA_R_16BF : CUDA_R_32F;
}

/// Gives the largest power of 2, up to LINEAR_CUBLAS_LARGEST_ALIGNMENT, that the address is a multiple of: what
/// cuBLASLt may count on.
///
///   address_ptr   LINEAR_CUBLAS_LARGEST_ALIGNMENT for cudaMalloc's memory, sizeof(float) for a view one float in
inline uint32_t linear_cublas_alignment(const void* address_ptr) {
    uint64_t address_val = (uint64_t)address_ptr;
    uint32_t alignment_val = LINEAR_CUBLAS_LARGEST_ALIGNMENT;
    while (alignment_val > 1 && address_val % alignment_val != 0) alignment_val /= 2;
    return alignment_val;
}

#endif // LINEAR_CUBLAS_CUH
