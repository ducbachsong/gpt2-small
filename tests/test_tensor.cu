// test_tensor.cu — tests for llmc/tensor.cuh. The expected numbers are worked out
// by hand in each test; no libtorch. Every check prints its test case, input, expected
// and actual value, and ok; the first FAIL stops the program.
//
//     nvcc -O3 -std=c++17 tests/test_tensor.cu -o test_tensor && ./test_tensor     # on a GPU (Colab)
#include "check.h"                // ASSERT_EQ, ASSERT_NEAR, sample_data
#include "../llmc/tensor.cuh"

// ── making tensors ────────────────────────────────────────────────────────────

static void zeros_makes_a_tensor_of_zeros(void) {
    // Expect: zeros({2, 3}) has 2 dims, 2 x 3 = 6 values, shape {2, 3}, and all 6 values are 0.
    Tensor t = Tensor::zeros({2, 3});
    ASSERT_EQ(t.dim(), 2);
    ASSERT_EQ(t.numel(), (size_t)6);                           // 2 x 3
    ASSERT_EQ(t.shape(), Shape({2, 3}));
    ASSERT_EQ(t.tolist(), std::vector<float>(6, 0.0f));
}

static void strides_follow_row_order(void) {
    // Expect: {2, 3, 4} has 24 values and strides {12, 4, 1}: the last dim moves fastest,
    // so one step along dim 0 skips 3 x 4 = 12 floats, along dim 1 skips 4, along dim 2 skips 1.
    Tensor t = Tensor::zeros({2, 3, 4});
    ASSERT_EQ(t.numel(), (size_t)24);                          // 2 x 3 x 4
    ASSERT_EQ(t.stride(), Shape({12, 4, 1}));
    ASSERT_EQ(t.size(0), (size_t)2);
    ASSERT_EQ(t.size(1), (size_t)3);
    ASSERT_EQ(t.size(2), (size_t)4);
    ASSERT_EQ(t.stride(0), (size_t)12);
    ASSERT_EQ(t.stride(1), (size_t)4);
    ASSERT_EQ(t.stride(2), (size_t)1);
}

static void an_empty_tensor_has_no_values(void) {
    // Expect: Tensor t has 0 dims, 0 values and no data;
    // after t = zeros({4}) it has 4 values and real GPU memory.
    Tensor t;
    ASSERT_EQ(t.dim(), 0);
    ASSERT_EQ(t.numel(), (size_t)0);
    ASSERT_EQ(t.data(), (float*)nullptr);
    t = Tensor::zeros({4});
    ASSERT_EQ(t.numel(), (size_t)4);
    ASSERT_EQ(t.data() != nullptr, true);
}

// ── values in and out ─────────────────────────────────────────────────────────

static void values_come_back_as_they_went_in(void) {
    // Expect: 0..7 copied into a 2x4 comes back as 0..7.
    Tensor t = Tensor::zeros({2, 4});
    t.copy_from_cpu(sample_data(8).data());
    ASSERT_EQ(t.tolist(), sample_data(8));
}

static void value_i_j_sits_at_i_times_stride_0_plus_j(void) {
    // Expect: in a 2x4 holding 0..7, [i][j] is data[i * 4 + j], so its value is i * 4 + j: [1][2] is 6.
    Tensor t = Tensor::zeros({2, 4});
    t.copy_from_cpu(sample_data(8).data());
    std::vector<float> values = t.tolist();
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 4; j++) ASSERT_EQ(values[i * t.stride(0) + j * t.stride(1)], (float)(i * 4 + j));
    }
    ASSERT_EQ(values[1 * t.stride(0) + 2 * t.stride(1)], 6.0f);
}

// ── view: the same values, another shape ──────────────────────────────────────

static void view_changes_the_shape_not_the_values(void) {
    // Expect: 2x4 viewed as 4x2 has shape {4, 2}, strides {2, 1}, the same data pointer and
    // the same values 0..7; viewed as {8} it has shape {8}.
    Tensor t = Tensor::zeros({2, 4});
    t.copy_from_cpu(sample_data(8).data());
    Tensor v = t.view({4, 2});
    ASSERT_EQ(v.shape(), Shape({4, 2}));
    ASSERT_EQ(v.stride(), Shape({2, 1}));
    ASSERT_EQ(v.data(), t.data());                         // not a copy
    ASSERT_EQ(v.tolist(), sample_data(8));
    ASSERT_EQ(t.view({8}).shape(), Shape({8}));
}

static void a_view_shares_the_values(void) {
    // Expect: 0..7 written through the view {8} shows up in t: both use one piece of GPU memory.
    Tensor t = Tensor::zeros({2, 4});
    Tensor v = t.view({8});
    v.copy_from_cpu(sample_data(8).data());
    ASSERT_EQ(t.tolist(), sample_data(8));
}

static void a_view_keeps_the_memory_alive(void) {
    // Expect: v still reads 0..7 after t is gone: v shares t's memory, so the memory stays.
    Tensor v;
    {
        Tensor t = Tensor::zeros({2, 4});
        t.copy_from_cpu(sample_data(8).data());
        v = t.view({8});
    }
    ASSERT_EQ(v.tolist(), sample_data(8));
}

// ── narrow: some rows ─────────────────────────────────────────────────────────

static void narrow_takes_rows(void) {
    // Expect: rows 1 and 2 of a 4x3 holding 0..11 are a 2x3 that starts 1 x 3 = 3 floats in
    // and holds 3..8.
    Tensor t = Tensor::zeros({4, 3});
    t.copy_from_cpu(sample_data(12).data());
    Tensor rows = t.narrow(0, 1, 2);
    ASSERT_EQ(rows.shape(), Shape({2, 3}));
    ASSERT_EQ(rows.data(), t.data() + 3);
    ASSERT_EQ(rows.tolist(), std::vector<float>({3, 4, 5, 6, 7, 8}));
}

static void narrow_then_view(void) {
    // Expect: values 8..19 of a flat 20, seen as 3x4, start 8 floats in and hold 8..19.
    // (How TensorBuffer makes its tensors.)
    Tensor all = Tensor::zeros({20});
    all.copy_from_cpu(sample_data(20).data());
    Tensor piece = all.narrow(0, 8, 12).view({3, 4});
    ASSERT_EQ(piece.shape(), Shape({3, 4}));
    ASSERT_EQ(piece.data(), all.data() + 8);
    ASSERT_EQ(piece.tolist(), std::vector<float>({8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}));
}

static void writing_a_narrow_leaves_the_rest_alone(void) {
    // Expect: 1s written into rows 1 and 2 of a 4x3 of zeros; rows 0 and 3 stay 0.
    Tensor t = Tensor::zeros({4, 3});
    t.narrow(0, 1, 2).copy_from_cpu(std::vector<float>(6, 1.0f).data());
    ASSERT_EQ(t.tolist(), std::vector<float>({0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0}));
}

// ── from_blob: memory someone else owns ───────────────────────────────────────

static void from_blob_uses_the_memory_and_does_not_free_it(void) {
    // Expect: from_blob uses our GPU pointer and reads 0..7; after the tensor is gone, the memory
    // is still ours and still holds 0..7.
    float* gpu;
    cudaCheck(cudaMalloc(&gpu, 8 * sizeof(float)));
    cudaCheck(cudaMemcpy(gpu, sample_data(8).data(), 8 * sizeof(float), cudaMemcpyHostToDevice));
    {
        Tensor t = Tensor::from_blob(gpu, {2, 4});
        ASSERT_EQ(t.data(), gpu);
        ASSERT_EQ(t.tolist(), sample_data(8));
    }
    // t is gone; the memory must still be ours: copying from it must work.
    std::vector<float> after(8);
    cudaCheck(cudaMemcpy(after.data(), gpu, 8 * sizeof(float), cudaMemcpyDeviceToHost));
    ASSERT_EQ(after, sample_data(8));
    cudaCheck(cudaFree(gpu));
}

// ── more ways to make tensors ─────────────────────────────────────────────────

static void empty_has_the_shape(void) {
    // Expect: empty({3, 4}) has shape {3, 4} and 12 values. Its values are unset, so they are not checked.
    Tensor t = Tensor::empty({3, 4});
    ASSERT_EQ(t.shape(), Shape({3, 4}));
    ASSERT_EQ(t.numel(), (size_t)12);
    ASSERT_EQ(t.data() != nullptr, true);
}

static void ones_makes_a_tensor_of_ones(void) {
    // Expect: ones({2, 2}) holds {1, 1, 1, 1}.
    Tensor t = Tensor::ones({2, 2});
    ASSERT_EQ(t.tolist(), std::vector<float>({1, 1, 1, 1}));
}

static void zeros_like_and_empty_like_copy_shape_and_dtype(void) {
    // Expect: zeros_like and empty_like of a 2x5 int tensor are 2x5 int tensors in memory of their
    // own; zeros_like holds ten 0s.
    Tensor t = Tensor::ones({2, 5}, DType::Int32);
    Tensor z = Tensor::zeros_like(t);
    Tensor e = Tensor::empty_like(t);
    ASSERT_EQ(z.shape(), Shape({2, 5}));
    ASSERT_EQ(z.dtype() == DType::Int32, true);
    ASSERT_EQ(e.shape(), Shape({2, 5}));
    ASSERT_EQ(e.dtype() == DType::Int32, true);
    ASSERT_EQ(z.data_ptr<int>() != t.data_ptr<int>(), true);
    ASSERT_EQ(z.tolist<int>(), std::vector<int>(10, 0));
}

// ── int tensors ───────────────────────────────────────────────────────────────

static void int_tensors_hold_ints(void) {
    // Expect: token ids {3, 464, 50256, 0} go into an Int32 tensor and come back the same.
    Tensor tokens = Tensor::zeros({4}, DType::Int32);
    std::vector<int> ids = {3, 464, 50256, 0};
    tokens.copy_from_cpu(ids.data());
    ASSERT_EQ(tokens.dtype() == DType::Int32, true);
    ASSERT_EQ(tokens.tolist<int>(), ids);
}

// ── setting values ────────────────────────────────────────────────────────────

static void zero_sets_every_value_to_0(void) {
    // Expect: 0..5 in a 2x3, after zero_(), is six 0s.
    Tensor t = Tensor::zeros({2, 3});
    t.copy_from_cpu(sample_data(6).data());
    t.zero_();
    ASSERT_EQ(t.tolist(), std::vector<float>(6, 0.0f));
}

static void fill_sets_every_value(void) {
    // Expect: fill_(1.5) gives six 1.5s in a float 2x3; fill_(7) gives four 7s in an int {4}.
    Tensor t = Tensor::zeros({2, 3});
    t.fill_(1.5);
    ASSERT_EQ(t.tolist(), std::vector<float>(6, 1.5f));
    Tensor ints = Tensor::zeros({4}, DType::Int32);
    ints.fill_(7);
    ASSERT_EQ(ints.tolist<int>(), std::vector<int>(4, 7));
}

static void fill_on_a_narrow_fills_only_those_rows(void) {
    // Expect: fill_(2) on rows 1 and 2 of a 4x3 of zeros; rows 0 and 3 stay 0.
    Tensor t = Tensor::zeros({4, 3});
    t.narrow(0, 1, 2).fill_(2);
    ASSERT_EQ(t.tolist(), std::vector<float>({0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0}));
}

// ── random values ─────────────────────────────────────────────────────────────

static void the_same_seed_gives_the_same_numbers(void) {
    // Expect: randn({1000}) after manual_seed(7) twice gives the same 1000 numbers; after seed 8, others.
    Tensor::manual_seed(7);
    std::vector<float> first = Tensor::randn({1000}).tolist();
    Tensor::manual_seed(7);
    std::vector<float> again = Tensor::randn({1000}).tolist();
    Tensor::manual_seed(8);
    std::vector<float> other = Tensor::randn({1000}).tolist();
    ASSERT_EQ(again, first);
    ASSERT_EQ(other != first, true);
}

static void normal_has_the_mean_and_std_asked_for(void) {
    // Expect: 100000 values of normal_(0, 0.02) have a mean within 0.0005 of 0 and a std within
    // 2% of 0.02. (By chance they are off by about 0.00006 and 0.2%, so the limits are wide.)
    Tensor::manual_seed(1);
    Tensor t = Tensor::zeros({100000});
    t.normal_(0.0f, 0.02f);
    std::vector<float> values = t.tolist();
    double sum = 0.0, sum_squares = 0.0;
    for (float value : values) sum += value;
    double mean = sum / values.size();
    for (float value : values) sum_squares += (value - mean) * (value - mean);
    double deviation = sqrt(sum_squares / values.size());
    ASSERT_NEAR(mean, 0.0, 0.0005);
    ASSERT_NEAR(deviation, 0.02, 0.02 * 0.02);                 // 2% of 0.02
}

// ── copies on the GPU ─────────────────────────────────────────────────────────

static void copy_copies_gpu_to_gpu(void) {
    // Expect: copy_ of a 2x3 holding 0..5 into a 3x2 gives the 3x2 the values 0..5 and keeps its
    // shape {3, 2} and its own memory.
    Tensor source = Tensor::zeros({2, 3});
    source.copy_from_cpu(sample_data(6).data());
    Tensor target = Tensor::zeros({3, 2});
    target.copy_(source);
    ASSERT_EQ(target.tolist(), sample_data(6));
    ASSERT_EQ(target.shape(), Shape({3, 2}));
    ASSERT_EQ(target.data() != source.data(), true);
}

static void clone_is_a_separate_copy(void) {
    // Expect: a clone of 0..3 holds 0..3 and still does after the original is zeroed.
    Tensor original = Tensor::zeros({4});
    original.copy_from_cpu(sample_data(4).data());
    Tensor copy = original.clone();
    original.zero_();
    ASSERT_EQ(copy.tolist(), sample_data(4));
    ASSERT_EQ(copy.data() != original.data(), true);
}

// ── reading one value ─────────────────────────────────────────────────────────

static void item_reads_the_one_value(void) {
    // Expect: item() of a float {1} holding 42.5 is 42.5; item<int>() of an int {1} holding 50256 is 50256.
    Tensor loss = Tensor::zeros({1});
    float value = 42.5f;
    loss.copy_from_cpu(&value);
    ASSERT_EQ(loss.item(), 42.5f);
    Tensor token = Tensor::zeros({1}, DType::Int32);
    int id = 50256;
    token.copy_from_cpu(&id);
    ASSERT_EQ(token.item<int>(), 50256);
}

static void indexing_picks_rows_and_values(void) {
    // Expect: in a 2x4 holding 0..7, t[1] is row 1: shape {4}, values 4..7, starting 4 values in;
    // t[1][2] has no dims and 1 value, 6.
    Tensor t = Tensor::zeros({2, 4});
    t.copy_from_cpu(sample_data(8).data());
    Tensor row = t[1];
    ASSERT_EQ(row.shape(), Shape({4}));
    ASSERT_EQ(row.tolist(), std::vector<float>({4, 5, 6, 7}));
    ASSERT_EQ(row.data(), t.data() + 4);
    ASSERT_EQ(t[1][2].dim(), 0);
    ASSERT_EQ(t[1][2].numel(), (size_t)1);
    ASSERT_EQ(t[1][2].item(), 6.0f);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void) {
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
    zero_sets_every_value_to_0();
    fill_sets_every_value();
    fill_on_a_narrow_fills_only_those_rows();
    the_same_seed_gives_the_same_numbers();
    normal_has_the_mean_and_std_asked_for();
    copy_copies_gpu_to_gpu();
    clone_is_a_separate_copy();
    item_reads_the_one_value();
    indexing_picks_rows_and_values();

    printf("all tests passed\n");
    return 0;
}
