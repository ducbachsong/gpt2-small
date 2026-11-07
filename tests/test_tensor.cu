// test_tensor.cu — tests for llmc/tensor.cuh. The expected numbers are worked out
// by hand in each test; no libtorch. Every check prints its test case, input, expected
// and actual value, and ok; the first FAIL stops the program.
//
//     nvcc -O3 -std=c++17 tests/test_tensor.cu -o test_tensor && ./test_tensor     # on a GPU (Colab)
#include "test_helpers.h"         // ASSERT_EQ, ASSERT_NEAR, sample_data
#include "../llmc/tensor.cuh"

#define HALF_BYTES 2              // a half's or a bfloat16's bytes
#define FLOAT_BYTES 4             // a float's or an int's bytes

// ── making tensors ────────────────────────────────────────────────────────────

static void zeros_makes_a_tensor_of_zeros(void) {
    // Expect: zeros({2, 3}) has 2 dims, 2 x 3 = 6 values, shape {2, 3}, and all 6 values are 0.
    Tensor t_tensor = Tensor::zeros({2, 3});
    ASSERT_EQ(t_tensor.dim(), 2);
    ASSERT_EQ(t_tensor.numel(), (size_t)6);                    // 2 x 3
    ASSERT_EQ(t_tensor.shape(), Shape({2, 3}));
    ASSERT_EQ(t_tensor.tolist(), std::vector<float>(6, 0.0f));
}

static void strides_follow_row_order(void) {
    // Expect: {2, 3, 4} has 24 values and strides {12, 4, 1}: the last dim moves fastest,
    // so one step along dim 0 skips 3 x 4 = 12 floats, along dim 1 skips 4, along dim 2 skips 1.
    Tensor t_tensor = Tensor::zeros({2, 3, 4});
    ASSERT_EQ(t_tensor.numel(), (size_t)24);                   // 2 x 3 x 4
    ASSERT_EQ(t_tensor.stride(), Shape({12, 4, 1}));
    ASSERT_EQ(t_tensor.size(0), (size_t)2);
    ASSERT_EQ(t_tensor.size(1), (size_t)3);
    ASSERT_EQ(t_tensor.size(2), (size_t)4);
    ASSERT_EQ(t_tensor.stride(0), (size_t)12);
    ASSERT_EQ(t_tensor.stride(1), (size_t)4);
    ASSERT_EQ(t_tensor.stride(2), (size_t)1);
}

static void an_empty_tensor_has_no_values(void) {
    // Expect: Tensor t has 0 dims, 0 values and no data;
    // after t = zeros({4}) it has 4 values and real GPU memory.
    // 1. a Tensor made with nothing
    Tensor t_tensor;
    ASSERT_EQ(t_tensor.dim(), 0);
    ASSERT_EQ(t_tensor.numel(), (size_t)0);
    ASSERT_EQ(t_tensor.data(), (float*)nullptr);
    // 2. given zeros({4})
    t_tensor = Tensor::zeros({4});
    ASSERT_EQ(t_tensor.numel(), (size_t)4);
    ASSERT_EQ(t_tensor.data() != nullptr, true);
}

// ── values in and out ─────────────────────────────────────────────────────────

static void values_come_back_as_they_went_in(void) {
    // Expect: 0..7 copied into a 2x4 comes back as 0..7.
    Tensor t_tensor = Tensor::zeros({2, 4});
    t_tensor.copy_from_cpu(sample_data(8).data());
    ASSERT_EQ(t_tensor.tolist(), sample_data(8));
}

static void value_i_j_sits_at_i_times_stride_0_plus_j(void) {
    // Expect: in a 2x4 holding 0..7, [i][j] is data[i * 4 + j], so its value is i * 4 + j: [1][2] is 6.
    // 1. 0..7 in a 2x4
    Tensor t_tensor = Tensor::zeros({2, 4});
    t_tensor.copy_from_cpu(sample_data(8).data());
    std::vector<float> values_on_cpu_list = t_tensor.tolist();
    // 2. each value at i x stride(0) + j x stride(1)
    for (size_t i_val = 0; i_val < 2; i_val++) {
        for (size_t j_val = 0; j_val < 4; j_val++)
            ASSERT_EQ(values_on_cpu_list[i_val * t_tensor.stride(0) + j_val * t_tensor.stride(1)], (float)(i_val * 4 + j_val));
    }
    ASSERT_EQ(values_on_cpu_list[1 * t_tensor.stride(0) + 2 * t_tensor.stride(1)], 6.0f);
}

// ── view: the same values, another shape ──────────────────────────────────────

static void view_changes_the_shape_not_the_values(void) {
    // Expect: 2x4 viewed as 4x2 has shape {4, 2}, strides {2, 1}, the same data pointer and
    // the same values 0..7; viewed as {8} it has shape {8}.
    // 1. 0..7 in a 2x4, viewed as 4x2
    Tensor t_tensor = Tensor::zeros({2, 4});
    t_tensor.copy_from_cpu(sample_data(8).data());
    Tensor view_tensor = t_tensor.view({4, 2});
    // 2. check
    ASSERT_EQ(view_tensor.shape(), Shape({4, 2}));
    ASSERT_EQ(view_tensor.stride(), Shape({2, 1}));
    ASSERT_EQ(view_tensor.data(), t_tensor.data());           // not a copy
    ASSERT_EQ(view_tensor.tolist(), sample_data(8));
    ASSERT_EQ(t_tensor.view({8}).shape(), Shape({8}));
}

static void a_view_shares_the_values(void) {
    // Expect: 0..7 written through the view {8} shows up in t: both use one piece of GPU memory.
    Tensor t_tensor = Tensor::zeros({2, 4});
    Tensor view_tensor = t_tensor.view({8});
    view_tensor.copy_from_cpu(sample_data(8).data());
    ASSERT_EQ(t_tensor.tolist(), sample_data(8));
}

static void a_view_keeps_the_memory_alive(void) {
    // Expect: v still reads 0..7 after t is gone: v shares t's memory, so the memory stays.
    // 1. a view of a tensor that then goes
    Tensor view_tensor;
    {
        Tensor t_tensor = Tensor::zeros({2, 4});
        t_tensor.copy_from_cpu(sample_data(8).data());
        view_tensor = t_tensor.view({8});
    }
    // 2. check
    ASSERT_EQ(view_tensor.tolist(), sample_data(8));
}

// ── narrow: some rows ─────────────────────────────────────────────────────────

static void narrow_takes_rows(void) {
    // Expect: rows 1 and 2 of a 4x3 holding 0..11 are a 2x3 that starts 1 x 3 = 3 floats in
    // and holds 3..8.
    // 1. rows 1 and 2
    Tensor t_tensor = Tensor::zeros({4, 3});
    t_tensor.copy_from_cpu(sample_data(12).data());
    Tensor rows_tensor = t_tensor.narrow(0, 1, 2);
    // 2. check
    ASSERT_EQ(rows_tensor.shape(), Shape({2, 3}));
    ASSERT_EQ(rows_tensor.data(), t_tensor.data() + 3);
    ASSERT_EQ(rows_tensor.tolist(), std::vector<float>({3, 4, 5, 6, 7, 8}));
}

static void narrow_then_view(void) {
    // Expect: values 8..19 of a flat 20, seen as 3x4, start 8 floats in and hold 8..19.
    // (How TensorBuffer makes its tensors.)
    // 1. a narrow, then a view
    Tensor whole_tensor = Tensor::zeros({20});
    whole_tensor.copy_from_cpu(sample_data(20).data());
    Tensor piece_tensor = whole_tensor.narrow(0, 8, 12).view({3, 4});
    // 2. check
    ASSERT_EQ(piece_tensor.shape(), Shape({3, 4}));
    ASSERT_EQ(piece_tensor.data(), whole_tensor.data() + 8);
    ASSERT_EQ(piece_tensor.tolist(), std::vector<float>({8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}));
}

static void writing_a_narrow_leaves_the_rest_alone(void) {
    // Expect: 1s written into rows 1 and 2 of a 4x3 of zeros; rows 0 and 3 stay 0.
    Tensor t_tensor = Tensor::zeros({4, 3});
    t_tensor.narrow(0, 1, 2).copy_from_cpu(std::vector<float>(6, 1.0f).data());
    ASSERT_EQ(t_tensor.tolist(), std::vector<float>({0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0}));
}

// ── from_blob: memory someone else owns ───────────────────────────────────────

static void from_blob_uses_the_memory_and_does_not_free_it(void) {
    // Expect: from_blob uses our GPU pointer and reads 0..7; after the tensor is gone, the memory
    // is still ours and still holds 0..7.
    // 1. our own GPU memory, 0..7
    float* gpu_ptr;
    cudaCheck(cudaMalloc(&gpu_ptr, 8 * sizeof(float)));
    cudaCheck(cudaMemcpy(gpu_ptr, sample_data(8).data(), 8 * sizeof(float), cudaMemcpyHostToDevice));
    // 2. a tensor on it, that then goes
    {
        Tensor t_tensor = Tensor::from_blob(gpu_ptr, {2, 4});
        ASSERT_EQ(t_tensor.data(), gpu_ptr);
        ASSERT_EQ(t_tensor.tolist(), sample_data(8));
    }
    // 3. t is gone; the memory must still be ours: copying from it must work
    std::vector<float> values_after_list(8);
    cudaCheck(cudaMemcpy(values_after_list.data(), gpu_ptr, 8 * sizeof(float), cudaMemcpyDeviceToHost));
    ASSERT_EQ(values_after_list, sample_data(8));
    cudaCheck(cudaFree(gpu_ptr));
}

// ── more ways to make tensors ─────────────────────────────────────────────────

static void empty_has_the_shape(void) {
    // Expect: empty({3, 4}) has shape {3, 4} and 12 values. Its values are unset, so they are not checked.
    Tensor t_tensor = Tensor::empty({3, 4});
    ASSERT_EQ(t_tensor.shape(), Shape({3, 4}));
    ASSERT_EQ(t_tensor.numel(), (size_t)12);
    ASSERT_EQ(t_tensor.data() != nullptr, true);
}

static void ones_makes_a_tensor_of_ones(void) {
    // Expect: ones({2, 2}) holds {1, 1, 1, 1}.
    Tensor t_tensor = Tensor::ones({2, 2});
    ASSERT_EQ(t_tensor.tolist(), std::vector<float>({1, 1, 1, 1}));
}

static void zeros_like_and_empty_like_copy_shape_and_dtype(void) {
    // Expect: zeros_like and empty_like of a 2x5 int tensor are 2x5 int tensors in memory of their
    // own; zeros_like holds ten 0s.
    // 1. zeros_like and empty_like of a 2x5 int tensor
    Tensor t_tensor = Tensor::ones({2, 5}, DType::Int32);
    Tensor zeros_like_t_tensor = Tensor::zeros_like(t_tensor);
    Tensor empty_like_t_tensor = Tensor::empty_like(t_tensor);
    // 2. check
    ASSERT_EQ(zeros_like_t_tensor.shape(), Shape({2, 5}));
    ASSERT_EQ(zeros_like_t_tensor.dtype() == DType::Int32, true);
    ASSERT_EQ(empty_like_t_tensor.shape(), Shape({2, 5}));
    ASSERT_EQ(empty_like_t_tensor.dtype() == DType::Int32, true);
    ASSERT_EQ(zeros_like_t_tensor.data_ptr<int>() != t_tensor.data_ptr<int>(), true);
    ASSERT_EQ(zeros_like_t_tensor.tolist<int>(), std::vector<int>(10, 0));
}

// ── int tensors ───────────────────────────────────────────────────────────────

static void int_tensors_hold_ints(void) {
    // Expect: token ids {3, 464, 50256, 0} go into an Int32 tensor and come back the same.
    Tensor tokens_tensor = Tensor::zeros({4}, DType::Int32);
    std::vector<int> ids_list = {3, 464, 50256, 0};
    tokens_tensor.copy_from_cpu(ids_list.data());
    ASSERT_EQ(tokens_tensor.dtype() == DType::Int32, true);
    ASSERT_EQ(tokens_tensor.tolist<int>(), ids_list);
}

// ── half tensors ──────────────────────────────────────────────────────────────

/// Reads halves as floats, to compare them: every half is a float exactly, so nothing is lost.
static std::vector<float> as_floats(const std::vector<half>& halves_list) {
    std::vector<float> floats_list(halves_list.size());
    for (size_t i_val = 0; i_val < halves_list.size(); i_val++) floats_list[i_val] = __half2float(halves_list[i_val]);
    return floats_list;
}

/// Rounds floats to halves on the CPU, to put values into a half tensor.
static std::vector<half> as_halves(const std::vector<float>& floats_list) {
    std::vector<half> halves_list(floats_list.size());
    for (size_t i_val = 0; i_val < floats_list.size(); i_val++) halves_list[i_val] = __float2half(floats_list[i_val]);
    return halves_list;
}

static void half_tensors_take_2_bytes_a_value(void) {
    // Expect: a {2, 4} half tensor has element_size 2, where float and int have 4. Its strides are {4, 1},
    // counted in values as for floats, so row 1 starts 4 values in, which is 4 x 2 = 8 bytes.
    Tensor t_tensor = Tensor::zeros({2, 4}, DType::Float16);
    ASSERT_EQ(t_tensor.element_size(), (size_t)HALF_BYTES);
    ASSERT_EQ(Tensor::zeros({1}).element_size(), (size_t)FLOAT_BYTES);
    ASSERT_EQ(Tensor::zeros({1}, DType::Int32).element_size(), (size_t)FLOAT_BYTES);
    ASSERT_EQ(t_tensor.stride(), Shape({4, 1}));
    ASSERT_EQ(t_tensor[1].data_ptr<half>(), t_tensor.data_ptr<half>() + 4);
    ASSERT_EQ((size_t)((char*)t_tensor[1].data_ptr<half>() - (char*)t_tensor.data_ptr<half>()), (size_t)(4 * HALF_BYTES));
    ASSERT_EQ(as_floats(t_tensor.tolist<half>()), std::vector<float>(8, 0.0f));
}

static void half_values_come_back_as_they_went_in(void) {
    // Expect: 0..7 as halves copied into a 2x4 half tensor come back as 0..7: small whole numbers are
    // halves exactly.
    Tensor t_tensor = Tensor::zeros({2, 4}, DType::Float16);
    t_tensor.copy_from_cpu(as_halves(sample_data(8)).data());
    ASSERT_EQ(as_floats(t_tensor.tolist<half>()), sample_data(8));
}

static void to_half_rounds_each_value_to_the_nearest_half(void) {
    // Expect: a half keeps 11 significant bits, so to(Float16) rounds:
    //     1/3   -> 0.333251953125      0.1 -> 0.0999755859375      1 and -2.5 stay
    //     2049  -> 2048: from 2048 to 4096 the halves are 2 apart, and 2049 is halfway from 2048 to 2050;
    //                    a tie goes to the half whose last bit is 0, 2048
    //     2051  -> 2052: halfway from 2050 to 2052, and 2052's last bit is 0
    //     65504 stays: the largest half;  70000 -> inf;  1e-8 -> 0: below half the smallest half, 6e-8
    // to(Float32) gives the same numbers back as floats, and the shape stays {9}.
    // 1. the values, and what they round to
    std::vector<float> values_list = {1.0f, 1.0f / 3.0f, 0.1f, -2.5f, 2049.0f, 2051.0f, 65504.0f, 70000.0f, 1e-8f};
    std::vector<float> rounded_list = {1.0f, 0.333251953125f, 0.0999755859375f, -2.5f, 2048.0f, 2052.0f, 65504.0f,
                                       INFINITY, 0.0f};
    // 2. to(Float16)
    Tensor floats_tensor = Tensor::zeros({9});
    floats_tensor.copy_from_cpu(values_list.data());
    Tensor halves_tensor = floats_tensor.to(DType::Float16);
    // 3. check
    ASSERT_EQ(halves_tensor.dtype() == DType::Float16, true);
    ASSERT_EQ(halves_tensor.shape(), Shape({9}));
    ASSERT_EQ(as_floats(halves_tensor.tolist<half>()), rounded_list);
    ASSERT_EQ(halves_tensor.to(DType::Float32).tolist(), rounded_list);
}

static void copy_converts_between_float_and_half(void) {
    // Expect: copy_ of floats {0.1, 0.5, 3} into a half {1, 3} rounds 0.1 to 0.0999755859375; copy_ of that
    // into a float {3} gives the halves back exactly.
    // 1. floats into halves
    Tensor floats_tensor = Tensor::zeros({3});
    floats_tensor.copy_from_cpu(std::vector<float>({0.1f, 0.5f, 3.0f}).data());
    Tensor halves_tensor = Tensor::zeros({1, 3}, DType::Float16);
    halves_tensor.copy_(floats_tensor);
    ASSERT_EQ(as_floats(halves_tensor.tolist<half>()), std::vector<float>({0.0999755859375f, 0.5f, 3.0f}));
    // 2. halves back into floats
    Tensor back_tensor = Tensor::zeros({3});
    back_tensor.copy_(halves_tensor);
    ASSERT_EQ(back_tensor.tolist(), std::vector<float>({0.0999755859375f, 0.5f, 3.0f}));
}

static void to_its_own_dtype_shares_the_tensor(void) {
    // Expect: to(Float32) of a float tensor and to(Float16) of a half one are the same tensors, not copies.
    Tensor floats_tensor = Tensor::zeros({4});
    Tensor halves_tensor = Tensor::zeros({4}, DType::Float16);
    ASSERT_EQ(floats_tensor.to(DType::Float32).data(), floats_tensor.data());
    ASSERT_EQ(halves_tensor.to(DType::Float16).data_ptr<half>(), halves_tensor.data_ptr<half>());
}

static void fill_ones_and_zero_for_halves(void) {
    // Expect: ones({3}, Float16) is {1, 1, 1}. fill_(0.1) on rows 1 and 2 of a 4x3 half of zeros gives them
    // 0.1 rounded, 0.0999755859375, and leaves rows 0 and 3 at 0; zero_() then makes all 12 values 0.
    // 1. ones
    ASSERT_EQ(as_floats(Tensor::ones({3}, DType::Float16).tolist<half>()), std::vector<float>({1, 1, 1}));
    // 2. fill_ on rows 1 and 2
    Tensor t_tensor = Tensor::zeros({4, 3}, DType::Float16);
    t_tensor.narrow(0, 1, 2).fill_(0.1);
    float r_val = 0.0999755859375f;
    ASSERT_EQ(as_floats(t_tensor.tolist<half>()), std::vector<float>({0, 0, 0, r_val, r_val, r_val, r_val, r_val, r_val, 0, 0, 0}));
    // 3. zero_
    t_tensor.zero_();
    ASSERT_EQ(as_floats(t_tensor.tolist<half>()), std::vector<float>(12, 0.0f));
}

static void narrow_then_view_on_halves(void) {
    // Expect: values 8..19 of a flat half 20, seen as 3x4, start 8 halves = 16 bytes in and hold 8..19, as
    // TensorBuffer makes its tensors.
    Tensor whole_tensor = Tensor::zeros({20}, DType::Float16);
    whole_tensor.copy_from_cpu(as_halves(sample_data(20)).data());
    Tensor piece_tensor = whole_tensor.narrow(0, 8, 12).view({3, 4});
    ASSERT_EQ(piece_tensor.data_ptr<half>(), whole_tensor.data_ptr<half>() + 8);
    ASSERT_EQ(as_floats(piece_tensor.tolist<half>()), std::vector<float>({8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}));
}

static void randn_in_half_is_randn_in_float_rounded(void) {
    // Expect: after manual_seed(7), randn({1000}, Float16) is randn({1000}) after the same seed, each value
    // rounded to a half: the numbers are made as floats either way.
    Tensor::manual_seed(7);
    std::vector<float> floats_rounded_list = as_floats(Tensor::randn({1000}).to(DType::Float16).tolist<half>());
    Tensor::manual_seed(7);
    std::vector<float> made_as_halves_list = as_floats(Tensor::randn({1000}, DType::Float16).tolist<half>());
    ASSERT_EQ(made_as_halves_list, floats_rounded_list);
}

static void item_and_indexing_on_halves(void) {
    // Expect: in a 2x4 half holding 0..7, t[1][2] is 6, read with item<half>(); a half {1} filled with 42.5
    // reads 42.5.
    // 1. t[1][2]
    Tensor t_tensor = Tensor::zeros({2, 4}, DType::Float16);
    t_tensor.copy_from_cpu(as_halves(sample_data(8)).data());
    ASSERT_EQ(__half2float(t_tensor[1][2].item<half>()), 6.0f);
    // 2. a half {1} filled
    Tensor loss_tensor = Tensor::zeros({1}, DType::Float16);
    loss_tensor.fill_(42.5);
    ASSERT_EQ(__half2float(loss_tensor.item<half>()), 42.5f);
}

// ── bfloat16 tensors ──────────────────────────────────────────────────────────

static void bfloat16_has_float_s_range_and_8_significant_bits(void) {
    // Expect: a bfloat16 is a float's first 16 bits: float's range, but 8 significant bits, so to(BFloat16)
    // rounds:
    //     1/3   -> 0.333984375         0.1 -> 0.10009765625        1 and -2.5 stay
    //     257   -> 256: from 256 to 512 the bfloat16s are 2 apart, and 257 is halfway from 256 to 258;
    //                   a tie goes to the one whose last bit is 0, 256
    //     259   -> 260: halfway from 258 to 260, and 260's last bit is 0
    //     65504 -> 65536 and 70000 -> 70144: no inf, where a half had 65504 and inf
    //     1e-8  -> 1.0011717677116394e-8: small values keep their 8 bits, where a half had 0
    // to(Float32) gives them back exactly; element_size is 2; fill_(0.1) gives 0.10009765625; to(Float16) of
    // 70144 is inf, past the largest half; tolist<nv_bfloat16>() reads the same values.
    // 1. the values, and what they round to
    std::vector<float> values_list = {1.0f, 1.0f / 3.0f, 0.1f, -2.5f, 257.0f, 259.0f, 65504.0f, 70000.0f, 1e-8f};
    std::vector<float> rounded_list = {1.0f, 0.333984375f, 0.10009765625f, -2.5f, 256.0f, 260.0f, 65536.0f, 70144.0f,
                                       1.0011717677116394e-8f};
    // 2. to(BFloat16), read back on the GPU and on the CPU
    Tensor floats_tensor = Tensor::zeros({9});
    floats_tensor.copy_from_cpu(values_list.data());
    Tensor bfloat16s_tensor = floats_tensor.to(DType::BFloat16);
    ASSERT_EQ(bfloat16s_tensor.dtype() == DType::BFloat16, true);
    ASSERT_EQ(bfloat16s_tensor.element_size(), (size_t)HALF_BYTES);
    ASSERT_EQ(bfloat16s_tensor.to(DType::Float32).tolist(), rounded_list);
    std::vector<float> read_on_cpu_list;
    for (nv_bfloat16 value_val : bfloat16s_tensor.tolist<nv_bfloat16>()) read_on_cpu_list.push_back(to_float(value_val));
    ASSERT_EQ(read_on_cpu_list, rounded_list);
    // 3. 70144 as a half: inf
    ASSERT_EQ(bfloat16s_tensor.to(DType::Float16)[7].to(DType::Float32).item(), INFINITY);
    // 4. fill_(0.1)
    Tensor filled_tensor = Tensor::zeros({3}, DType::BFloat16);
    filled_tensor.fill_(0.1);
    ASSERT_EQ(filled_tensor.to(DType::Float32).tolist(), std::vector<float>(3, 0.10009765625f));
}

// ── setting values ────────────────────────────────────────────────────────────

static void zero_sets_every_value_to_0(void) {
    // Expect: 0..5 in a 2x3, after zero_(), is six 0s.
    Tensor t_tensor = Tensor::zeros({2, 3});
    t_tensor.copy_from_cpu(sample_data(6).data());
    t_tensor.zero_();
    ASSERT_EQ(t_tensor.tolist(), std::vector<float>(6, 0.0f));
}

static void fill_sets_every_value(void) {
    // Expect: fill_(1.5) gives six 1.5s in a float 2x3; fill_(7) gives four 7s in an int {4}.
    // 1. a float 2x3
    Tensor t_tensor = Tensor::zeros({2, 3});
    t_tensor.fill_(1.5);
    ASSERT_EQ(t_tensor.tolist(), std::vector<float>(6, 1.5f));
    // 2. an int {4}
    Tensor ints_tensor = Tensor::zeros({4}, DType::Int32);
    ints_tensor.fill_(7);
    ASSERT_EQ(ints_tensor.tolist<int>(), std::vector<int>(4, 7));
}

static void fill_on_a_narrow_fills_only_those_rows(void) {
    // Expect: fill_(2) on rows 1 and 2 of a 4x3 of zeros; rows 0 and 3 stay 0.
    Tensor t_tensor = Tensor::zeros({4, 3});
    t_tensor.narrow(0, 1, 2).fill_(2);
    ASSERT_EQ(t_tensor.tolist(), std::vector<float>({0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0}));
}

// ── random values ─────────────────────────────────────────────────────────────

static void the_same_seed_gives_the_same_numbers(void) {
    // Expect: randn({1000}) after manual_seed(7) twice gives the same 1000 numbers; after seed 8, others.
    // 1. seed 7 twice, then seed 8
    Tensor::manual_seed(7);
    std::vector<float> seed_7_numbers_list = Tensor::randn({1000}).tolist();
    Tensor::manual_seed(7);
    std::vector<float> seed_7_again_list = Tensor::randn({1000}).tolist();
    Tensor::manual_seed(8);
    std::vector<float> seed_8_numbers_list = Tensor::randn({1000}).tolist();
    // 2. check
    ASSERT_EQ(seed_7_again_list, seed_7_numbers_list);
    ASSERT_EQ(seed_8_numbers_list != seed_7_numbers_list, true);
}

static void normal_has_the_mean_and_std_asked_for(void) {
    // Expect: 100000 values of normal_(0, 0.02) have a mean within 0.0005 of 0 and a std within
    // 2% of 0.02. (By chance they are off by about 0.00006 and 0.2%, so the limits are wide.)
    // 1. normal_(0, 0.02)
    Tensor::manual_seed(1);
    Tensor t_tensor = Tensor::zeros({100000});
    t_tensor.normal_(0.0f, 0.02f);
    // 2. the mean and std, in double
    std::vector<float> values_on_cpu_list = t_tensor.tolist();
    double sum_val = 0.0;
    double sum_squares_val = 0.0;
    for (float value_val : values_on_cpu_list) sum_val += value_val;
    double mean_val = sum_val / values_on_cpu_list.size();
    for (float value_val : values_on_cpu_list) sum_squares_val += (value_val - mean_val) * (value_val - mean_val);
    double deviation_val = sqrt(sum_squares_val / values_on_cpu_list.size());
    // 3. check
    ASSERT_NEAR(mean_val, 0.0, 0.0005);
    ASSERT_NEAR(deviation_val, 0.02, 0.02 * 0.02);             // 2% of 0.02
}

// ── copies on the GPU ─────────────────────────────────────────────────────────

static void copy_copies_gpu_to_gpu(void) {
    // Expect: copy_ of a 2x3 holding 0..5 into a 3x2 gives the 3x2 the values 0..5 and keeps its
    // shape {3, 2} and its own memory.
    // 1. copy_
    Tensor source_tensor = Tensor::zeros({2, 3});
    source_tensor.copy_from_cpu(sample_data(6).data());
    Tensor target_tensor = Tensor::zeros({3, 2});
    target_tensor.copy_(source_tensor);
    // 2. check
    ASSERT_EQ(target_tensor.tolist(), sample_data(6));
    ASSERT_EQ(target_tensor.shape(), Shape({3, 2}));
    ASSERT_EQ(target_tensor.data() != source_tensor.data(), true);
}

static void clone_is_a_separate_copy(void) {
    // Expect: a clone of 0..3 holds 0..3 and still does after the original is zeroed.
    // 1. a clone, then the original zeroed
    Tensor original_tensor = Tensor::zeros({4});
    original_tensor.copy_from_cpu(sample_data(4).data());
    Tensor copy_tensor = original_tensor.clone();
    original_tensor.zero_();
    // 2. check
    ASSERT_EQ(copy_tensor.tolist(), sample_data(4));
    ASSERT_EQ(copy_tensor.data() != original_tensor.data(), true);
}

// ── reading one value ─────────────────────────────────────────────────────────

static void item_reads_the_one_value(void) {
    // Expect: item() of a float {1} holding 42.5 is 42.5; item<int>() of an int {1} holding 50256 is 50256.
    // 1. a float {1}
    Tensor loss_tensor = Tensor::zeros({1});
    float value_val = 42.5f;
    loss_tensor.copy_from_cpu(&value_val);
    ASSERT_EQ(loss_tensor.item(), 42.5f);
    // 2. an int {1}
    Tensor token_tensor = Tensor::zeros({1}, DType::Int32);
    int id_val = 50256;
    token_tensor.copy_from_cpu(&id_val);
    ASSERT_EQ(token_tensor.item<int>(), 50256);
}

static void indexing_picks_rows_and_values(void) {
    // Expect: in a 2x4 holding 0..7, t[1] is row 1: shape {4}, values 4..7, starting 4 values in;
    // t[1][2] has no dims and 1 value, 6.
    // 1. 0..7 in a 2x4; row 1
    Tensor t_tensor = Tensor::zeros({2, 4});
    t_tensor.copy_from_cpu(sample_data(8).data());
    Tensor row_tensor = t_tensor[1];
    ASSERT_EQ(row_tensor.shape(), Shape({4}));
    ASSERT_EQ(row_tensor.tolist(), std::vector<float>({4, 5, 6, 7}));
    ASSERT_EQ(row_tensor.data(), t_tensor.data() + 4);
    // 2. one value
    ASSERT_EQ(t_tensor[1][2].dim(), 0);
    ASSERT_EQ(t_tensor[1][2].numel(), (size_t)1);
    ASSERT_EQ(t_tensor[1][2].item(), 6.0f);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void) {
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    zeros_makes_a_tensor_of_zeros();
    strides_follow_row_order();
    an_empty_tensor_has_no_values();
    values_come_back_as_they_went_in();
    value_i_j_sits_at_i_times_stride_0_plus_j();
    view_changes_the_shape_not_the_values();
    a_view_shares_the_values();
    a_view_keeps_the_memory_alive();
    narrow_takes_rows();
    narrow_then_view();
    writing_a_narrow_leaves_the_rest_alone();
    from_blob_uses_the_memory_and_does_not_free_it();
    empty_has_the_shape();
    ones_makes_a_tensor_of_ones();
    zeros_like_and_empty_like_copy_shape_and_dtype();
    int_tensors_hold_ints();
    half_tensors_take_2_bytes_a_value();
    half_values_come_back_as_they_went_in();
    to_half_rounds_each_value_to_the_nearest_half();
    copy_converts_between_float_and_half();
    to_its_own_dtype_shares_the_tensor();
    fill_ones_and_zero_for_halves();
    narrow_then_view_on_halves();
    randn_in_half_is_randn_in_float_rounded();
    item_and_indexing_on_halves();
    bfloat16_has_float_s_range_and_8_significant_bits();
    zero_sets_every_value_to_0();
    fill_sets_every_value();
    fill_on_a_narrow_fills_only_those_rows();
    the_same_seed_gives_the_same_numbers();
    normal_has_the_mean_and_std_asked_for();
    copy_copies_gpu_to_gpu();
    clone_is_a_separate_copy();
    item_reads_the_one_value();
    indexing_picks_rows_and_values();

    printf("\nall tests passed\n");
    return 0;
}
