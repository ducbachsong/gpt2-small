// test_tensor_buffer.cu — tests for llmc/tensor_buffer.cuh. The expected numbers are
// worked out by hand in each test; no libtorch. Every check prints its test case,
// input, expected and actual value, and ok; the first FAIL stops the program.
//
//     nvcc -O3 -std=c++17 tests/test_tensor_buffer.cu -o test_tensor_buffer && ./test_tensor_buffer   # on a GPU
//
// Part 1 tests the buffer itself. Most of its tests use the same three tensors, 32
// values in all:
//
//     tensor 0: {2, 4}    8 values    starts at value 0
//     tensor 1: {12}     12 values    starts at value 8
//     tensor 2: {2, 2, 3} 12 values   starts at value 20
//
// Part 2 runs every test of tests/test_tensor.cu again, on a tensor inside a buffer.
#include "test_helpers.h"         // ASSERT_EQ, ASSERT_NEAR, sample_data
#include "../llmc/tensor_buffer.cuh"

#define BFLOAT16_BYTES 2          // a bfloat16's bytes
#define NEIGHBOUR_VALUES 4        // part 2: the values of each neighbour around the tensor under test

// ── adding tensors ────────────────────────────────────────────────────────────

static void add_returns_each_tensors_index(void) {
    // Expect: the first add gives index 0, the second 1; the buffer then has 2 tensors.
    TensorBuffer buffer_val;
    ASSERT_EQ(buffer_val.add({2, 4}), 0);
    ASSERT_EQ(buffer_val.add({12}), 1);
    ASSERT_EQ(buffer_val.num_tensors(), 2);
}

static void add_many_returns_its_group(void) {
    // Expect: after one add (a tensor, not a group), add_many of two shapes gives group 0
    // and add_many of one shape group 1. The buffer then has 2 groups of 2 and 1 tensors,
    // and 1 + 2 + 1 = 4 tensors in all.
    // 1. one add, then two add_many
    TensorBuffer buffer_val;
    buffer_val.add({4});
    ASSERT_EQ(buffer_val.add_many({{2, 4}, {12}}), 0);
    ASSERT_EQ(buffer_val.add_many({{2, 2, 3}}), 1);
    // 2. the groups and tensors
    ASSERT_EQ(buffer_val.num_groups(), 2);
    ASSERT_EQ(buffer_val.group_size(0), 2);
    ASSERT_EQ(buffer_val.group_size(1), 1);
    ASSERT_EQ(buffer_val.num_tensors(), 4);
}

static void numel_is_known_before_allocate(void) {
    // Expect: before allocate(), numel() already adds up the shapes: 8 + 12 + 12 = 32.
    TensorBuffer buffer_val;
    buffer_val.add_many({{2, 4}, {12}, {2, 2, 3}});
    ASSERT_EQ(buffer_val.numel(), (size_t)32);
}

// ── one allocation ────────────────────────────────────────────────────────────

static void allocate_makes_one_flat_tensor_of_zeros(void) {
    // Expect: flat() is one flat tensor of shape {32}, and all 32 values are 0.
    TensorBuffer buffer_val;
    buffer_val.add_many({{2, 4}, {12}, {2, 2, 3}});
    buffer_val.allocate();
    ASSERT_EQ(buffer_val.flat().shape(), Shape({32}));
    ASSERT_EQ(buffer_val.flat().tolist(), std::vector<float>(32, 0.0f));
    ASSERT_EQ(buffer_val.numel(), (size_t)32);
}

static void each_tensor_has_its_own_shape(void) {
    // Expect: after allocate(), tensor(0), tensor(1), tensor(2) have shapes {2, 4}, {12}, {2, 2, 3}.
    TensorBuffer buffer_val;
    buffer_val.add_many({{2, 4}, {12}, {2, 2, 3}});
    buffer_val.allocate();
    ASSERT_EQ(buffer_val.tensor(0).shape(), Shape({2, 4}));
    ASSERT_EQ(buffer_val.tensor(1).shape(), Shape({12}));
    ASSERT_EQ(buffer_val.tensor(2).shape(), Shape({2, 2, 3}));
}

static void tensors_sit_one_after_another(void) {
    // Expect: tensor 0 starts at the start of flat(), tensor 1 at 8 values in (after
    // tensor 0's 8), tensor 2 at 20 values in (after 8 + 12).
    TensorBuffer buffer_val;
    buffer_val.add_many({{2, 4}, {12}, {2, 2, 3}});
    buffer_val.allocate();
    ASSERT_EQ(buffer_val.tensor(0).data(), buffer_val.flat().data());
    ASSERT_EQ(buffer_val.tensor(1).data(), buffer_val.flat().data() + 8);
    ASSERT_EQ(buffer_val.tensor(2).data(), buffer_val.flat().data() + 20);
}

// ── the tensors share the buffer's memory ─────────────────────────────────────

static void each_tensor_sees_its_part_of_the_memory(void) {
    // Expect: with 0..31 written into flat(), tensor 0 holds 0..7, tensor 1 holds 8..19
    // and tensor 2 holds 20..31.
    // 1. 0..31 into flat()
    TensorBuffer buffer_val;
    buffer_val.add_many({{2, 4}, {12}, {2, 2, 3}});
    buffer_val.allocate();
    buffer_val.flat().copy_from_cpu(sample_data(32).data());
    // 2. each tensor's part
    ASSERT_EQ(buffer_val.tensor(0).tolist(), std::vector<float>({0, 1, 2, 3, 4, 5, 6, 7}));
    ASSERT_EQ(buffer_val.tensor(1).tolist(), std::vector<float>({8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}));
    ASSERT_EQ(buffer_val.tensor(2).tolist(), std::vector<float>({20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}));
}

static void writing_a_tensor_writes_only_its_part(void) {
    // Expect: fill_(1) on tensor 1 turns values 8..19 of flat() into 1; values 0..7 and
    // 20..31 stay 0.
    // 1. fill_(1) on tensor 1
    TensorBuffer buffer_val;
    buffer_val.add_many({{2, 4}, {12}, {2, 2, 3}});
    buffer_val.allocate();
    buffer_val.tensor(1).fill_(1);
    // 2. what flat() should hold
    std::vector<float> expected_list(32, 0.0f);
    for (size_t i_val = 8; i_val < 20; i_val++) expected_list[i_val] = 1.0f;
    // 3. check
    ASSERT_EQ(buffer_val.flat().tolist(), expected_list);
}

static void zeroing_all_zeroes_every_tensor(void) {
    // Expect: with 0..31 in the buffer, flat().zero_() makes every tensor all 0s: one call
    // for the whole buffer, as the trainer zeroes its gradients.
    // 1. 0..31 in, then zero_()
    TensorBuffer buffer_val;
    buffer_val.add_many({{2, 4}, {12}, {2, 2, 3}});
    buffer_val.allocate();
    buffer_val.flat().copy_from_cpu(sample_data(32).data());
    buffer_val.flat().zero_();
    // 2. check
    ASSERT_EQ(buffer_val.tensor(0).tolist(), std::vector<float>(8, 0.0f));
    ASSERT_EQ(buffer_val.tensor(1).tolist(), std::vector<float>(12, 0.0f));
    ASSERT_EQ(buffer_val.tensor(2).tolist(), std::vector<float>(12, 0.0f));
}

static void a_tensor_keeps_the_memory_alive(void) {
    // Expect: a tensor taken from a buffer still reads 0..3 after the buffer is gone: it
    // shares the buffer's memory, so the memory stays.
    // 1. a tensor of a buffer that then goes
    Tensor kept_tensor;
    {
        TensorBuffer buffer_val;
        buffer_val.add({4});
        buffer_val.allocate();
        buffer_val.flat().copy_from_cpu(sample_data(4).data());
        kept_tensor = buffer_val.tensor(0);
    }
    // 2. check
    ASSERT_EQ(kept_tensor.tolist(), sample_data(4));
}

// ── matrices ──────────────────────────────────────────────────────────────────

static void matrices_are_the_tensors_with_2_or_more_dims(void) {
    // Expect: {2, 4}, {1, 4} and {2, 2, 3} are matrices (dim() >= 2, they get weight decay);
    // {12} is not (dim() is 1). The trainer asks tensor(k).dim() >= 2, as nanoGPT does.
    TensorBuffer buffer_val;
    buffer_val.add_many({{2, 4}, {12}, {1, 4}, {2, 2, 3}});
    buffer_val.allocate();
    ASSERT_EQ(buffer_val.tensor(0).dim() >= 2, true);
    ASSERT_EQ(buffer_val.tensor(1).dim() >= 2, false);
    ASSERT_EQ(buffer_val.tensor(2).dim() >= 2, true);
    ASSERT_EQ(buffer_val.tensor(3).dim() >= 2, true);
}

// ── groups ────────────────────────────────────────────────────────────────────
// Two groups of the same three tensors, as AdamW adds its params and then their grads:
//
//     flat()   [ group 0: tensors 0, 1, 2 = values 0..31 | group 1: tensors 3, 4, 5 = values 32..63 ]

/// Makes the buffer the group tests share: the two groups above, allocated, with 0..63 written into flat().
static TensorBuffer two_groups_of_three(void) {
    TensorBuffer buffer_val;
    buffer_val.add_many({{2, 4}, {12}, {2, 2, 3}});
    buffer_val.add_many({{2, 4}, {12}, {2, 2, 3}});
    buffer_val.allocate();
    buffer_val.flat().copy_from_cpu(sample_data(64).data());
    return buffer_val;
}

static void a_group_is_one_flat_tensor_of_its_values(void) {
    // Expect: group(0) is one flat tensor of its 8 + 12 + 12 = 32 values, 0..31; group(1)
    // starts 32 values into flat() and holds 32..63.
    // 1. group 0
    TensorBuffer buffer_val = two_groups_of_three();
    ASSERT_EQ(buffer_val.group(0).shape(), Shape({32}));
    ASSERT_EQ(buffer_val.group(0).data(), buffer_val.flat().data());
    ASSERT_EQ(buffer_val.group(0).tolist(), sample_data(32));
    // 2. group 1
    std::vector<float> values_32_to_63_list;
    for (int i_val = 32; i_val < 64; i_val++) values_32_to_63_list.push_back((float)i_val);
    ASSERT_EQ(buffer_val.group(1).data(), buffer_val.flat().data() + 32);
    ASSERT_EQ(buffer_val.group(1).tolist(), values_32_to_63_list);
}

static void tensor_g_k_is_tensor_k_of_group_g(void) {
    // Expect: with groups of 3, tensor(0, 1) is tensor(1), and tensor(1, 0) is tensor(3),
    // the first of group 1: the same memory, the same shape. tensor(1, 2) is {2, 2, 3}.
    TensorBuffer buffer_val = two_groups_of_three();
    ASSERT_EQ(buffer_val.tensor(0, 1).data(), buffer_val.tensor(1).data());
    ASSERT_EQ(buffer_val.tensor(1, 0).data(), buffer_val.tensor(3).data());
    ASSERT_EQ(buffer_val.tensor(1, 0).shape(), Shape({2, 4}));
    ASSERT_EQ(buffer_val.tensor(1, 2).shape(), Shape({2, 2, 3}));
}

static void writing_a_group_writes_only_its_part(void) {
    // Expect: group(1).zero_() makes values 32..63 of flat() 0, in one call (as the trainer
    // zeroes all its gradients); group 0 keeps 0..31.
    // 1. zero_() on group 1
    TensorBuffer buffer_val = two_groups_of_three();
    buffer_val.group(1).zero_();
    // 2. what flat() should hold, and check
    std::vector<float> expected_list = sample_data(32);
    expected_list.resize(64, 0.0f);                            // 0..31, then 32 zeros
    ASSERT_EQ(buffer_val.flat().tolist(), expected_list);
}

// ── zeros_like ────────────────────────────────────────────────────────────────

static void zeros_like_copies_the_groups(void) {
    // Expect: zeros_like of the two groups of three has the same groups: 2 of them, 3
    // tensors each, group(1) 32 values starting 32 in, all 0.
    TensorBuffer buffer_val = two_groups_of_three();
    TensorBuffer copy_val = TensorBuffer::zeros_like(buffer_val);
    ASSERT_EQ(copy_val.num_groups(), 2);
    ASSERT_EQ(copy_val.group_size(0), 3);
    ASSERT_EQ(copy_val.group_size(1), 3);
    ASSERT_EQ(copy_val.group(1).data(), copy_val.flat().data() + 32);
    ASSERT_EQ(copy_val.group(1).tolist(), std::vector<float>(32, 0.0f));
}

static void zeros_like_copies_the_layout_not_the_memory(void) {
    // Expect: zeros_like(params) has the same 3 tensors with the same shapes and 32
    // values, all 0, in memory of its own: writing into it leaves params' 0..31 alone.
    // 1. params holding 0..31, and zeros_like of it
    TensorBuffer params_val;
    params_val.add_many({{2, 4}, {12}, {2, 2, 3}});
    params_val.allocate();
    params_val.flat().copy_from_cpu(sample_data(32).data());
    TensorBuffer grads_val = TensorBuffer::zeros_like(params_val);
    // 2. the same layout, zeros, memory of its own
    ASSERT_EQ(grads_val.num_tensors(), 3);
    ASSERT_EQ(grads_val.tensor(0).shape(), Shape({2, 4}));
    ASSERT_EQ(grads_val.tensor(1).shape(), Shape({12}));
    ASSERT_EQ(grads_val.tensor(2).shape(), Shape({2, 2, 3}));
    ASSERT_EQ(grads_val.flat().tolist(), std::vector<float>(32, 0.0f));
    ASSERT_EQ(grads_val.flat().data() != params_val.flat().data(), true);
    // 3. writing into it leaves params alone
    grads_val.flat().fill_(5);
    ASSERT_EQ(params_val.flat().tolist(), sample_data(32));
}

// ── int buffers ───────────────────────────────────────────────────────────────

static void an_int_buffer_holds_ints(void) {
    // Expect: allocate(DType::Int32) gives Int32 tensors of zeros; zeros_like keeps Int32.
    // 1. an Int32 buffer
    TensorBuffer tokens_val;
    tokens_val.add_many({{2, 4}, {4}});
    tokens_val.allocate(DType::Int32);
    ASSERT_EQ(tokens_val.flat().dtype() == DType::Int32, true);
    ASSERT_EQ(tokens_val.tensor(0).tolist<int>(), std::vector<int>(8, 0));
    ASSERT_EQ(tokens_val.tensor(1).tolist<int>(), std::vector<int>(4, 0));
    // 2. zeros_like of it
    TensorBuffer copy_val = TensorBuffer::zeros_like(tokens_val);
    ASSERT_EQ(copy_val.flat().dtype() == DType::Int32, true);
}

// ── bfloat16 buffers ──────────────────────────────────────────────────────────

static void a_bfloat16_buffer_takes_2_bytes_a_value(void) {
    // Expect: allocate(DType::BFloat16) of {2, 4}, {12}, {2, 2, 3} gives BFloat16 tensors of zeros, each
    // starting 2 bytes per value after the last: tensor(1) 8 values = 16 bytes in, tensor(2) 20 values = 40
    // bytes in. 0..11 written into tensor(1) reads back as 0..11 (whole numbers below 256 are bfloat16s
    // exactly), and its neighbours stay 0. zeros_like keeps BFloat16.
    // 1. a BFloat16 buffer: where each tensor starts
    TensorBuffer buffer_val;
    buffer_val.add_many({{2, 4}, {12}, {2, 2, 3}});
    buffer_val.allocate(DType::BFloat16);
    const char* start_ptr = (const char*)buffer_val.flat().data_ptr<nv_bfloat16>();
    ASSERT_EQ(buffer_val.flat().dtype() == DType::BFloat16, true);
    ASSERT_EQ((size_t)((const char*)buffer_val.tensor(1).data_ptr<nv_bfloat16>() - start_ptr), (size_t)(8 * BFLOAT16_BYTES));
    ASSERT_EQ((size_t)((const char*)buffer_val.tensor(2).data_ptr<nv_bfloat16>() - start_ptr), (size_t)(20 * BFLOAT16_BYTES));
    // 2. 0..11 into tensor(1)
    Tensor values_tensor = Tensor::zeros({12});
    values_tensor.copy_from_cpu(sample_data(12).data());
    buffer_val.tensor(1).copy_(values_tensor);                         // float into bfloat16: copy_ rounds
    ASSERT_EQ(buffer_val.tensor(1).to(DType::Float32).tolist(), sample_data(12));
    ASSERT_EQ(buffer_val.tensor(0).to(DType::Float32).tolist(), std::vector<float>(8, 0.0f));
    ASSERT_EQ(buffer_val.tensor(2).to(DType::Float32).tolist(), std::vector<float>(12, 0.0f));
    // 3. zeros_like of it
    TensorBuffer copy_val = TensorBuffer::zeros_like(buffer_val);
    ASSERT_EQ(copy_val.flat().dtype() == DType::BFloat16, true);
}

// ══ every Tensor test again, on a tensor inside a TensorBuffer ════════════════
// The cases of tests/test_tensor.cu, with the same numbers, but the tensor under test
// is tensor(1) of a buffer, between two neighbours of NEIGHBOUR_VALUES (4) values:
//
//     flat()   [ neighbour {4} | the tensor under test | neighbour {4} ]
//                tensor(0)       tensor(1), 4 values in  tensor(2)
//
// So it does not start at the start of the memory, and a write that goes past its ends
// would show in a neighbour: every test that writes checks both neighbours after.

/// Makes the buffer part 2 tests in: three tensors, {4}, `shape_list`, {4}, all zeros. tensor(1) is the one
/// under test.
static TensorBuffer buffer_with_neighbours(const Shape& shape_list, DType dtype_val = DType::Float32) {
    TensorBuffer buffer_val;
    buffer_val.add_many({{NEIGHBOUR_VALUES}, shape_list, {NEIGHBOUR_VALUES}});
    buffer_val.allocate(dtype_val);
    return buffer_val;
}

// ── making tensors ────────────────────────────────────────────────────────────

static void buffer_tensor_starts_as_zeros(void) {
    // Expect: tensor(1) of shape {2, 3} has 2 dims, 2 x 3 = 6 values, shape {2, 3}, all 0,
    // and starts 4 values into flat(), right after its neighbour.
    TensorBuffer buffer_val = buffer_with_neighbours({2, 3});
    Tensor t_tensor = buffer_val.tensor(1);
    ASSERT_EQ(t_tensor.dim(), 2);
    ASSERT_EQ(t_tensor.numel(), (size_t)6);
    ASSERT_EQ(t_tensor.shape(), Shape({2, 3}));
    ASSERT_EQ(t_tensor.tolist(), std::vector<float>(6, 0.0f));
    ASSERT_EQ(t_tensor.data(), buffer_val.flat().data() + NEIGHBOUR_VALUES);
}

static void buffer_tensor_strides_follow_row_order(void) {
    // Expect: {2, 3, 4} inside a buffer has 24 values and strides {12, 4, 1}, the same as a
    // tensor with memory of its own: one step along dim 0 skips 3 x 4 = 12 values.
    TensorBuffer buffer_val = buffer_with_neighbours({2, 3, 4});
    Tensor t_tensor = buffer_val.tensor(1);
    ASSERT_EQ(t_tensor.numel(), (size_t)24);
    ASSERT_EQ(t_tensor.stride(), Shape({12, 4, 1}));
    ASSERT_EQ(t_tensor.size(0), (size_t)2);
    ASSERT_EQ(t_tensor.size(1), (size_t)3);
    ASSERT_EQ(t_tensor.size(2), (size_t)4);
    ASSERT_EQ(t_tensor.stride(0), (size_t)12);
    ASSERT_EQ(t_tensor.stride(1), (size_t)4);
    ASSERT_EQ(t_tensor.stride(2), (size_t)1);
}

static void an_empty_tensor_can_take_a_buffer_tensor(void) {
    // Expect: Tensor t; has 0 dims, 0 values and no data; after t = buffer.tensor(1) it is
    // the {4} tensor: 4 values, starting 4 values into flat().
    // 1. a Tensor made with nothing
    TensorBuffer buffer_val = buffer_with_neighbours({4});
    Tensor t_tensor;
    ASSERT_EQ(t_tensor.dim(), 0);
    ASSERT_EQ(t_tensor.numel(), (size_t)0);
    ASSERT_EQ(t_tensor.data(), (float*)nullptr);
    // 2. given tensor(1)
    t_tensor = buffer_val.tensor(1);
    ASSERT_EQ(t_tensor.numel(), (size_t)4);
    ASSERT_EQ(t_tensor.data(), buffer_val.flat().data() + NEIGHBOUR_VALUES);
}

// ── values in and out ─────────────────────────────────────────────────────────

static void buffer_tensor_values_come_back_as_they_went_in(void) {
    // Expect: 0..7 copied into tensor(1), a 2x4, comes back as 0..7; the neighbours stay 0.
    TensorBuffer buffer_val = buffer_with_neighbours({2, 4});
    Tensor t_tensor = buffer_val.tensor(1);
    t_tensor.copy_from_cpu(sample_data(8).data());
    ASSERT_EQ(t_tensor.tolist(), sample_data(8));
    ASSERT_EQ(buffer_val.tensor(0).tolist(), std::vector<float>(NEIGHBOUR_VALUES, 0.0f));
    ASSERT_EQ(buffer_val.tensor(2).tolist(), std::vector<float>(NEIGHBOUR_VALUES, 0.0f));
}

static void buffer_tensor_value_i_j_sits_at_i_times_stride_0_plus_j(void) {
    // Expect: in a 2x4 holding 0..7, [i][j] is data[i * 4 + j], so its value is i * 4 + j:
    // [1][2] is 6. The same rule inside a buffer, counted from the tensor's own start.
    // 1. 0..7 in tensor(1), a 2x4
    TensorBuffer buffer_val = buffer_with_neighbours({2, 4});
    Tensor t_tensor = buffer_val.tensor(1);
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

static void buffer_tensor_view_changes_the_shape_not_the_values(void) {
    // Expect: tensor(1), a 2x4, viewed as 4x2 has shape {4, 2}, strides {2, 1}, the same data
    // pointer and the same values 0..7; viewed as {8} it has shape {8}.
    // 1. 0..7 in tensor(1), viewed as 4x2
    TensorBuffer buffer_val = buffer_with_neighbours({2, 4});
    Tensor t_tensor = buffer_val.tensor(1);
    t_tensor.copy_from_cpu(sample_data(8).data());
    Tensor view_tensor = t_tensor.view({4, 2});
    // 2. check
    ASSERT_EQ(view_tensor.shape(), Shape({4, 2}));
    ASSERT_EQ(view_tensor.stride(), Shape({2, 1}));
    ASSERT_EQ(view_tensor.data(), t_tensor.data());           // not a copy
    ASSERT_EQ(view_tensor.tolist(), sample_data(8));
    ASSERT_EQ(t_tensor.view({8}).shape(), Shape({8}));
}

static void buffer_tensor_view_shares_the_values(void) {
    // Expect: 0..7 written through a view {8} of tensor(1) shows up in tensor(1); the
    // neighbours stay 0.
    TensorBuffer buffer_val = buffer_with_neighbours({2, 4});
    Tensor t_tensor = buffer_val.tensor(1);
    Tensor view_tensor = t_tensor.view({8});
    view_tensor.copy_from_cpu(sample_data(8).data());
    ASSERT_EQ(t_tensor.tolist(), sample_data(8));
    ASSERT_EQ(buffer_val.tensor(0).tolist(), std::vector<float>(NEIGHBOUR_VALUES, 0.0f));
    ASSERT_EQ(buffer_val.tensor(2).tolist(), std::vector<float>(NEIGHBOUR_VALUES, 0.0f));
}

static void buffer_tensor_view_keeps_the_memory_alive(void) {
    // Expect: a view of tensor(1) still reads 0..7 after the buffer is gone: the view shares
    // the buffer's memory, so the memory stays.
    // 1. a view of a buffer's tensor, then the buffer goes
    Tensor view_tensor;
    {
        TensorBuffer buffer_val = buffer_with_neighbours({2, 4});
        Tensor t_tensor = buffer_val.tensor(1);
        t_tensor.copy_from_cpu(sample_data(8).data());
        view_tensor = t_tensor.view({8});
    }
    // 2. check
    ASSERT_EQ(view_tensor.tolist(), sample_data(8));
}

// ── narrow: some rows ─────────────────────────────────────────────────────────

static void buffer_tensor_narrow_takes_rows(void) {
    // Expect: rows 1 and 2 of tensor(1), a 4x3 holding 0..11, are a 2x3 that starts 1 x 3 = 3
    // values after tensor(1) (so 4 + 3 = 7 into flat()) and holds 3..8.
    // 1. rows 1 and 2
    TensorBuffer buffer_val = buffer_with_neighbours({4, 3});
    Tensor t_tensor = buffer_val.tensor(1);
    t_tensor.copy_from_cpu(sample_data(12).data());
    Tensor rows_tensor = t_tensor.narrow(0, 1, 2);
    // 2. check
    ASSERT_EQ(rows_tensor.shape(), Shape({2, 3}));
    ASSERT_EQ(rows_tensor.data(), t_tensor.data() + 3);
    ASSERT_EQ(rows_tensor.data(), buffer_val.flat().data() + NEIGHBOUR_VALUES + 3);
    ASSERT_EQ(rows_tensor.tolist(), std::vector<float>({3, 4, 5, 6, 7, 8}));
}

static void buffer_tensor_narrow_then_view(void) {
    // Expect: values 8..19 of tensor(1), a flat 20 holding 0..19, seen as 3x4, start 8 values
    // after tensor(1) and hold 8..19.
    // 1. a narrow, then a view
    TensorBuffer buffer_val = buffer_with_neighbours({20});
    Tensor t_tensor = buffer_val.tensor(1);
    t_tensor.copy_from_cpu(sample_data(20).data());
    Tensor piece_tensor = t_tensor.narrow(0, 8, 12).view({3, 4});
    // 2. check
    ASSERT_EQ(piece_tensor.shape(), Shape({3, 4}));
    ASSERT_EQ(piece_tensor.data(), t_tensor.data() + 8);
    ASSERT_EQ(piece_tensor.tolist(), std::vector<float>({8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}));
}

static void buffer_tensor_writing_a_narrow_leaves_the_rest_alone(void) {
    // Expect: 1s written into rows 1 and 2 of tensor(1), a 4x3 of zeros; rows 0 and 3 stay
    // 0, and so do the neighbours.
    TensorBuffer buffer_val = buffer_with_neighbours({4, 3});
    Tensor t_tensor = buffer_val.tensor(1);
    t_tensor.narrow(0, 1, 2).copy_from_cpu(std::vector<float>(6, 1.0f).data());
    ASSERT_EQ(t_tensor.tolist(), std::vector<float>({0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0}));
    ASSERT_EQ(buffer_val.tensor(0).tolist(), std::vector<float>(NEIGHBOUR_VALUES, 0.0f));
    ASSERT_EQ(buffer_val.tensor(2).tolist(), std::vector<float>(NEIGHBOUR_VALUES, 0.0f));
}

// ── from_blob: memory someone else owns ───────────────────────────────────────

static void from_blob_over_a_buffer_tensor_does_not_free_it(void) {
    // Expect: from_blob over the data of tensor(1) uses that pointer and reads 0..7; after the
    // from_blob tensor is gone, tensor(1) still holds 0..7: the buffer's memory was not freed.
    // 1. 0..7 in tensor(1)
    TensorBuffer buffer_val = buffer_with_neighbours({2, 4});
    Tensor t_tensor = buffer_val.tensor(1);
    t_tensor.copy_from_cpu(sample_data(8).data());
    // 2. a from_blob tensor on it, that then goes
    {
        Tensor blob_tensor = Tensor::from_blob(t_tensor.data(), {2, 4});
        ASSERT_EQ(blob_tensor.data(), t_tensor.data());
        ASSERT_EQ(blob_tensor.tolist(), sample_data(8));
    }
    // 3. the buffer's memory is still there
    ASSERT_EQ(t_tensor.tolist(), sample_data(8));
}

// ── more ways to make tensors ─────────────────────────────────────────────────

static void buffer_tensor_zeros_like_and_empty_like_copy_shape_and_dtype(void) {
    // Expect: zeros_like and empty_like of tensor(1), a 2x5 int tensor of ones, are 2x5 int
    // tensors in memory of their own, outside the buffer; zeros_like holds ten 0s.
    // 1. zeros_like and empty_like of tensor(1)
    TensorBuffer buffer_val = buffer_with_neighbours({2, 5}, DType::Int32);
    Tensor t_tensor = buffer_val.tensor(1);
    t_tensor.fill_(1);
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

static void buffer_tensor_fill_1_makes_ones(void) {
    // Expect: fill_(1) on tensor(1), a 2x2, gives {1, 1, 1, 1}, like Tensor::ones; the
    // neighbours stay 0.
    TensorBuffer buffer_val = buffer_with_neighbours({2, 2});
    Tensor t_tensor = buffer_val.tensor(1);
    t_tensor.fill_(1);
    ASSERT_EQ(t_tensor.tolist(), std::vector<float>({1, 1, 1, 1}));
    ASSERT_EQ(buffer_val.tensor(0).tolist(), std::vector<float>(NEIGHBOUR_VALUES, 0.0f));
    ASSERT_EQ(buffer_val.tensor(2).tolist(), std::vector<float>(NEIGHBOUR_VALUES, 0.0f));
}

// ── int tensors ───────────────────────────────────────────────────────────────

static void buffer_tensor_holds_ints(void) {
    // Expect: token ids {3, 464, 50256, 0} go into tensor(1) of an Int32 buffer and come back
    // the same; the neighbours stay 0.
    TensorBuffer buffer_val = buffer_with_neighbours({4}, DType::Int32);
    Tensor tokens_tensor = buffer_val.tensor(1);
    std::vector<int> ids_list = {3, 464, 50256, 0};
    tokens_tensor.copy_from_cpu(ids_list.data());
    ASSERT_EQ(tokens_tensor.dtype() == DType::Int32, true);
    ASSERT_EQ(tokens_tensor.tolist<int>(), ids_list);
    ASSERT_EQ(buffer_val.tensor(0).tolist<int>(), std::vector<int>(NEIGHBOUR_VALUES, 0));
    ASSERT_EQ(buffer_val.tensor(2).tolist<int>(), std::vector<int>(NEIGHBOUR_VALUES, 0));
}

// ── setting values ────────────────────────────────────────────────────────────

static void buffer_tensor_zero_leaves_the_neighbours_alone(void) {
    // Expect: with every value of the buffer 9 and 0..5 in tensor(1), a 2x3, zero_() on
    // tensor(1) gives six 0s; the neighbours keep their 9s.
    // 1. every value 9, then 0..5 in tensor(1)
    TensorBuffer buffer_val = buffer_with_neighbours({2, 3});
    buffer_val.flat().fill_(9);
    Tensor t_tensor = buffer_val.tensor(1);
    t_tensor.copy_from_cpu(sample_data(6).data());
    // 2. zero_() on tensor(1)
    t_tensor.zero_();
    // 3. check
    ASSERT_EQ(t_tensor.tolist(), std::vector<float>(6, 0.0f));
    ASSERT_EQ(buffer_val.tensor(0).tolist(), std::vector<float>(NEIGHBOUR_VALUES, 9.0f));
    ASSERT_EQ(buffer_val.tensor(2).tolist(), std::vector<float>(NEIGHBOUR_VALUES, 9.0f));
}

static void buffer_tensor_fill_sets_every_value(void) {
    // Expect: fill_(1.5) gives six 1.5s in tensor(1) of a float buffer, a 2x3; fill_(7) gives
    // four 7s in tensor(1) of an int buffer, a {4}; the neighbours stay 0 in both.
    // 1. a float buffer
    TensorBuffer floats_val = buffer_with_neighbours({2, 3});
    floats_val.tensor(1).fill_(1.5);
    ASSERT_EQ(floats_val.tensor(1).tolist(), std::vector<float>(6, 1.5f));
    ASSERT_EQ(floats_val.tensor(0).tolist(), std::vector<float>(NEIGHBOUR_VALUES, 0.0f));
    ASSERT_EQ(floats_val.tensor(2).tolist(), std::vector<float>(NEIGHBOUR_VALUES, 0.0f));
    // 2. an int buffer
    TensorBuffer ints_val = buffer_with_neighbours({4}, DType::Int32);
    ints_val.tensor(1).fill_(7);
    ASSERT_EQ(ints_val.tensor(1).tolist<int>(), std::vector<int>(4, 7));
    ASSERT_EQ(ints_val.tensor(0).tolist<int>(), std::vector<int>(NEIGHBOUR_VALUES, 0));
    ASSERT_EQ(ints_val.tensor(2).tolist<int>(), std::vector<int>(NEIGHBOUR_VALUES, 0));
}

static void buffer_tensor_fill_on_a_narrow_fills_only_those_rows(void) {
    // Expect: fill_(2) on rows 1 and 2 of tensor(1), a 4x3 of zeros; rows 0 and 3 stay 0, and
    // so do the neighbours.
    TensorBuffer buffer_val = buffer_with_neighbours({4, 3});
    Tensor t_tensor = buffer_val.tensor(1);
    t_tensor.narrow(0, 1, 2).fill_(2);
    ASSERT_EQ(t_tensor.tolist(), std::vector<float>({0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0}));
    ASSERT_EQ(buffer_val.tensor(0).tolist(), std::vector<float>(NEIGHBOUR_VALUES, 0.0f));
    ASSERT_EQ(buffer_val.tensor(2).tolist(), std::vector<float>(NEIGHBOUR_VALUES, 0.0f));
}

// ── random values ─────────────────────────────────────────────────────────────

static void buffer_tensor_the_same_seed_gives_the_same_numbers(void) {
    // Expect: after manual_seed(7), normal_(0, 1) on tensor(1), a {1000}, gives the same 1000
    // numbers as randn({1000}) after manual_seed(7); the neighbours stay 0.
    // 1. randn({1000}) after seed 7
    Tensor::manual_seed(7);
    std::vector<float> expected_list = Tensor::randn({1000}).tolist();
    // 2. normal_(0, 1) on tensor(1) after seed 7
    TensorBuffer buffer_val = buffer_with_neighbours({1000});
    Tensor::manual_seed(7);
    buffer_val.tensor(1).normal_(0.0f, 1.0f);
    // 3. check
    ASSERT_EQ(buffer_val.tensor(1).tolist(), expected_list);
    ASSERT_EQ(buffer_val.tensor(0).tolist(), std::vector<float>(NEIGHBOUR_VALUES, 0.0f));
    ASSERT_EQ(buffer_val.tensor(2).tolist(), std::vector<float>(NEIGHBOUR_VALUES, 0.0f));
}

static void buffer_tensor_normal_has_the_mean_and_std_asked_for(void) {
    // Expect: 100000 values of normal_(0, 0.02) in tensor(1) have a mean within 0.0005 of 0
    // and a std within 2% of 0.02; the neighbours stay 0.
    // 1. normal_(0, 0.02) on tensor(1)
    TensorBuffer buffer_val = buffer_with_neighbours({100000});
    Tensor::manual_seed(1);
    buffer_val.tensor(1).normal_(0.0f, 0.02f);
    // 2. the mean and std, in double
    std::vector<float> values_on_cpu_list = buffer_val.tensor(1).tolist();
    double sum_val = 0.0;
    double sum_squares_val = 0.0;
    for (float value_val : values_on_cpu_list) sum_val += value_val;
    double mean_val = sum_val / values_on_cpu_list.size();
    for (float value_val : values_on_cpu_list) sum_squares_val += (value_val - mean_val) * (value_val - mean_val);
    double deviation_val = sqrt(sum_squares_val / values_on_cpu_list.size());
    // 3. check
    ASSERT_NEAR(mean_val, 0.0, 0.0005);
    ASSERT_NEAR(deviation_val, 0.02, 0.02 * 0.02);             // 2% of 0.02
    ASSERT_EQ(buffer_val.tensor(0).tolist(), std::vector<float>(NEIGHBOUR_VALUES, 0.0f));
    ASSERT_EQ(buffer_val.tensor(2).tolist(), std::vector<float>(NEIGHBOUR_VALUES, 0.0f));
}

// ── copies on the GPU ─────────────────────────────────────────────────────────

static void buffer_tensor_copy_copies_gpu_to_gpu(void) {
    // Expect: copy_ of a 2x3 holding 0..5 into tensor(1), a 3x2, gives tensor(1) the values
    // 0..5 and keeps its shape {3, 2}; the neighbours stay 0. Copying tensor(1) out into a
    // tensor of its own gives that tensor 0..5 too.
    // 1. copy_ into tensor(1)
    TensorBuffer buffer_val = buffer_with_neighbours({3, 2});
    Tensor source_tensor = Tensor::zeros({2, 3});
    source_tensor.copy_from_cpu(sample_data(6).data());
    Tensor t_tensor = buffer_val.tensor(1);
    t_tensor.copy_(source_tensor);
    ASSERT_EQ(t_tensor.tolist(), sample_data(6));
    ASSERT_EQ(t_tensor.shape(), Shape({3, 2}));
    ASSERT_EQ(buffer_val.tensor(0).tolist(), std::vector<float>(NEIGHBOUR_VALUES, 0.0f));
    ASSERT_EQ(buffer_val.tensor(2).tolist(), std::vector<float>(NEIGHBOUR_VALUES, 0.0f));
    // 2. copy_ out of tensor(1)
    Tensor copied_out_tensor = Tensor::zeros({6});
    copied_out_tensor.copy_(t_tensor);
    ASSERT_EQ(copied_out_tensor.tolist(), sample_data(6));
}

static void buffer_tensor_clone_is_outside_the_buffer(void) {
    // Expect: a clone of tensor(1) holding 0..3 holds 0..3, in memory of its own: it still
    // does after every value of the buffer is zeroed.
    // 1. a clone, then the whole buffer zeroed
    TensorBuffer buffer_val = buffer_with_neighbours({4});
    Tensor t_tensor = buffer_val.tensor(1);
    t_tensor.copy_from_cpu(sample_data(4).data());
    Tensor copy_tensor = t_tensor.clone();
    buffer_val.flat().zero_();
    // 2. check
    ASSERT_EQ(copy_tensor.tolist(), sample_data(4));
    ASSERT_EQ(copy_tensor.data() != t_tensor.data(), true);
}

// ── reading one value ─────────────────────────────────────────────────────────

static void buffer_tensor_item_reads_the_one_value(void) {
    // Expect: item() of tensor(1), a float {1} holding 42.5, is 42.5; item<int>() of tensor(1)
    // of an int buffer, a {1} holding 50256, is 50256.
    // 1. a float buffer
    TensorBuffer floats_val = buffer_with_neighbours({1});
    float value_val = 42.5f;
    floats_val.tensor(1).copy_from_cpu(&value_val);
    ASSERT_EQ(floats_val.tensor(1).item(), 42.5f);
    // 2. an int buffer
    TensorBuffer ints_val = buffer_with_neighbours({1}, DType::Int32);
    int id_val = 50256;
    ints_val.tensor(1).copy_from_cpu(&id_val);
    ASSERT_EQ(ints_val.tensor(1).item<int>(), 50256);
}

static void buffer_tensor_indexing_picks_rows_and_values(void) {
    // Expect: in tensor(1), a 2x4 holding 0..7, t[1] is row 1: shape {4}, values 4..7,
    // starting 4 values after tensor(1) (8 into flat()); t[1][2] has no dims and 1 value, 6.
    // 1. 0..7 in tensor(1); row 1
    TensorBuffer buffer_val = buffer_with_neighbours({2, 4});
    Tensor t_tensor = buffer_val.tensor(1);
    t_tensor.copy_from_cpu(sample_data(8).data());
    Tensor row_tensor = t_tensor[1];
    ASSERT_EQ(row_tensor.shape(), Shape({4}));
    ASSERT_EQ(row_tensor.tolist(), std::vector<float>({4, 5, 6, 7}));
    ASSERT_EQ(row_tensor.data(), t_tensor.data() + 4);
    ASSERT_EQ(row_tensor.data(), buffer_val.flat().data() + NEIGHBOUR_VALUES + 4);
    // 2. one value
    ASSERT_EQ(t_tensor[1][2].dim(), 0);
    ASSERT_EQ(t_tensor[1][2].numel(), (size_t)1);
    ASSERT_EQ(t_tensor[1][2].item(), 6.0f);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void) {
    start_run_log(TESTS_LOG_FOLDER, __FILE__);    // the console, and logs/tests/<test>_<date>_<time>.log
    add_returns_each_tensors_index();
    add_many_returns_its_group();
    numel_is_known_before_allocate();
    allocate_makes_one_flat_tensor_of_zeros();
    each_tensor_has_its_own_shape();
    tensors_sit_one_after_another();
    each_tensor_sees_its_part_of_the_memory();
    writing_a_tensor_writes_only_its_part();
    zeroing_all_zeroes_every_tensor();
    a_tensor_keeps_the_memory_alive();
    matrices_are_the_tensors_with_2_or_more_dims();
    a_group_is_one_flat_tensor_of_its_values();
    tensor_g_k_is_tensor_k_of_group_g();
    writing_a_group_writes_only_its_part();
    zeros_like_copies_the_groups();
    zeros_like_copies_the_layout_not_the_memory();
    an_int_buffer_holds_ints();
    a_bfloat16_buffer_takes_2_bytes_a_value();

    buffer_tensor_starts_as_zeros();
    buffer_tensor_strides_follow_row_order();
    an_empty_tensor_can_take_a_buffer_tensor();
    buffer_tensor_values_come_back_as_they_went_in();
    buffer_tensor_value_i_j_sits_at_i_times_stride_0_plus_j();
    buffer_tensor_view_changes_the_shape_not_the_values();
    buffer_tensor_view_shares_the_values();
    buffer_tensor_view_keeps_the_memory_alive();
    buffer_tensor_narrow_takes_rows();
    buffer_tensor_narrow_then_view();
    buffer_tensor_writing_a_narrow_leaves_the_rest_alone();
    from_blob_over_a_buffer_tensor_does_not_free_it();
    buffer_tensor_zeros_like_and_empty_like_copy_shape_and_dtype();
    buffer_tensor_fill_1_makes_ones();
    buffer_tensor_holds_ints();
    buffer_tensor_zero_leaves_the_neighbours_alone();
    buffer_tensor_fill_sets_every_value();
    buffer_tensor_fill_on_a_narrow_fills_only_those_rows();
    buffer_tensor_the_same_seed_gives_the_same_numbers();
    buffer_tensor_normal_has_the_mean_and_std_asked_for();
    buffer_tensor_copy_copies_gpu_to_gpu();
    buffer_tensor_clone_is_outside_the_buffer();
    buffer_tensor_item_reads_the_one_value();
    buffer_tensor_indexing_picks_rows_and_values();

    printf("\nall tests passed\n");
    return 0;
}
