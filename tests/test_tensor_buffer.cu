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
#include "check.h"                // ASSERT_EQ, ASSERT_NEAR, sample_data
#include "../llmc/tensor_buffer.cuh"

// ── adding tensors ────────────────────────────────────────────────────────────

static void add_returns_each_tensors_index(void) {
    // Expect: the first add gives index 0, the second 1; the buffer then has 2 tensors.
    TensorBuffer buffer;
    ASSERT_EQ(buffer.add({2, 4}), 0);
    ASSERT_EQ(buffer.add({12}), 1);
    ASSERT_EQ(buffer.num_tensors(), 2);
}

static void add_many_returns_the_first_index(void) {
    // Expect: after one add, add_many of three shapes gives 1, the index of the first of
    // them; the buffer then has 4 tensors.
    TensorBuffer buffer;
    buffer.add({4});
    ASSERT_EQ(buffer.add_many({{2, 4}, {12}, {2, 2, 3}}), 1);
    ASSERT_EQ(buffer.num_tensors(), 4);
}

static void numel_is_known_before_allocate(void) {
    // Expect: before allocate(), numel() already adds up the shapes: 8 + 12 + 12 = 32.
    TensorBuffer buffer;
    buffer.add_many({{2, 4}, {12}, {2, 2, 3}});
    ASSERT_EQ(buffer.numel(), (size_t)32);
}

// ── one allocation ────────────────────────────────────────────────────────────

static void allocate_makes_one_flat_tensor_of_zeros(void) {
    // Expect: flat() is one flat tensor of shape {32}, and all 32 values are 0.
    TensorBuffer buffer;
    buffer.add_many({{2, 4}, {12}, {2, 2, 3}});
    buffer.allocate();
    ASSERT_EQ(buffer.flat().shape(), Shape({32}));
    ASSERT_EQ(buffer.flat().tolist(), std::vector<float>(32, 0.0f));
    ASSERT_EQ(buffer.numel(), (size_t)32);
}

static void each_tensor_has_its_own_shape(void) {
    // Expect: after allocate(), tensor(0), tensor(1), tensor(2) have shapes {2, 4}, {12}, {2, 2, 3}.
    TensorBuffer buffer;
    buffer.add_many({{2, 4}, {12}, {2, 2, 3}});
    buffer.allocate();
    ASSERT_EQ(buffer.tensor(0).shape(), Shape({2, 4}));
    ASSERT_EQ(buffer.tensor(1).shape(), Shape({12}));
    ASSERT_EQ(buffer.tensor(2).shape(), Shape({2, 2, 3}));
}

static void tensors_sit_one_after_another(void) {
    // Expect: tensor 0 starts at the start of flat(), tensor 1 at 8 values in (after
    // tensor 0's 8), tensor 2 at 20 values in (after 8 + 12).
    TensorBuffer buffer;
    buffer.add_many({{2, 4}, {12}, {2, 2, 3}});
    buffer.allocate();
    ASSERT_EQ(buffer.tensor(0).data(), buffer.flat().data());
    ASSERT_EQ(buffer.tensor(1).data(), buffer.flat().data() + 8);
    ASSERT_EQ(buffer.tensor(2).data(), buffer.flat().data() + 20);
}

// ── the tensors share the buffer's memory ─────────────────────────────────────

static void each_tensor_sees_its_part_of_the_memory(void) {
    // Expect: with 0..31 written into flat(), tensor 0 holds 0..7, tensor 1 holds 8..19
    // and tensor 2 holds 20..31.
    TensorBuffer buffer;
    buffer.add_many({{2, 4}, {12}, {2, 2, 3}});
    buffer.allocate();
    buffer.flat().copy_from_cpu(sample_data(32).data());
    ASSERT_EQ(buffer.tensor(0).tolist(), std::vector<float>({0, 1, 2, 3, 4, 5, 6, 7}));
    ASSERT_EQ(buffer.tensor(1).tolist(), std::vector<float>({8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}));
    ASSERT_EQ(buffer.tensor(2).tolist(), std::vector<float>({20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}));
}

static void writing_a_tensor_writes_only_its_part(void) {
    // Expect: fill_(1) on tensor 1 turns values 8..19 of flat() into 1; values 0..7 and
    // 20..31 stay 0.
    TensorBuffer buffer;
    buffer.add_many({{2, 4}, {12}, {2, 2, 3}});
    buffer.allocate();
    buffer.tensor(1).fill_(1);
    std::vector<float> expected(32, 0.0f);
    for (size_t i = 8; i < 20; i++) expected[i] = 1.0f;
    ASSERT_EQ(buffer.flat().tolist(), expected);
}

static void zeroing_all_zeroes_every_tensor(void) {
    // Expect: with 0..31 in the buffer, flat().zero_() makes every tensor all 0s: one call
    // for the whole buffer, as the trainer zeroes its gradients.
    TensorBuffer buffer;
    buffer.add_many({{2, 4}, {12}, {2, 2, 3}});
    buffer.allocate();
    buffer.flat().copy_from_cpu(sample_data(32).data());
    buffer.flat().zero_();
    ASSERT_EQ(buffer.tensor(0).tolist(), std::vector<float>(8, 0.0f));
    ASSERT_EQ(buffer.tensor(1).tolist(), std::vector<float>(12, 0.0f));
    ASSERT_EQ(buffer.tensor(2).tolist(), std::vector<float>(12, 0.0f));
}

static void a_tensor_keeps_the_memory_alive(void) {
    // Expect: a tensor taken from a buffer still reads 0..3 after the buffer is gone: it
    // shares the buffer's memory, so the memory stays.
    Tensor kept;
    {
        TensorBuffer buffer;
        buffer.add({4});
        buffer.allocate();
        buffer.flat().copy_from_cpu(sample_data(4).data());
        kept = buffer.tensor(0);
    }
    ASSERT_EQ(kept.tolist(), sample_data(4));
}

// ── matrices ──────────────────────────────────────────────────────────────────

static void matrices_are_the_tensors_with_2_or_more_dims(void) {
    // Expect: {2, 4}, {1, 4} and {2, 2, 3} are matrices (dim() >= 2, they get weight decay);
    // {12} is not (dim() is 1). The trainer asks tensor(k).dim() >= 2, as nanoGPT does.
    TensorBuffer buffer;
    buffer.add_many({{2, 4}, {12}, {1, 4}, {2, 2, 3}});
    buffer.allocate();
    ASSERT_EQ(buffer.tensor(0).dim() >= 2, true);
    ASSERT_EQ(buffer.tensor(1).dim() >= 2, false);
    ASSERT_EQ(buffer.tensor(2).dim() >= 2, true);
    ASSERT_EQ(buffer.tensor(3).dim() >= 2, true);
}

// ── zeros_like ────────────────────────────────────────────────────────────────

static void zeros_like_copies_the_layout_not_the_memory(void) {
    // Expect: zeros_like(params) has the same 3 tensors with the same shapes and 32
    // values, all 0, in memory of its own: writing into it leaves params' 0..31 alone.
    TensorBuffer params;
    params.add_many({{2, 4}, {12}, {2, 2, 3}});
    params.allocate();
    params.flat().copy_from_cpu(sample_data(32).data());
    TensorBuffer grads = TensorBuffer::zeros_like(params);
    ASSERT_EQ(grads.num_tensors(), 3);
    ASSERT_EQ(grads.tensor(0).shape(), Shape({2, 4}));
    ASSERT_EQ(grads.tensor(1).shape(), Shape({12}));
    ASSERT_EQ(grads.tensor(2).shape(), Shape({2, 2, 3}));
    ASSERT_EQ(grads.flat().tolist(), std::vector<float>(32, 0.0f));
    ASSERT_EQ(grads.flat().data() != params.flat().data(), true);
    grads.flat().fill_(5);
    ASSERT_EQ(params.flat().tolist(), sample_data(32));
}

// ── int buffers ───────────────────────────────────────────────────────────────

static void an_int_buffer_holds_ints(void) {
    // Expect: allocate(DType::Int32) gives Int32 tensors of zeros; zeros_like keeps Int32.
    TensorBuffer tokens;
    tokens.add_many({{2, 4}, {4}});
    tokens.allocate(DType::Int32);
    ASSERT_EQ(tokens.flat().dtype() == DType::Int32, true);
    ASSERT_EQ(tokens.tensor(0).tolist<int>(), std::vector<int>(8, 0));
    ASSERT_EQ(tokens.tensor(1).tolist<int>(), std::vector<int>(4, 0));
    TensorBuffer copy = TensorBuffer::zeros_like(tokens);
    ASSERT_EQ(copy.flat().dtype() == DType::Int32, true);
}

// ══ every Tensor test again, on a tensor inside a TensorBuffer ════════════════
// The cases of tests/test_tensor.cu, with the same numbers, but the tensor under test
// is tensor(1) of a buffer, between two neighbours of 4 values:
//
//     flat()   [ neighbour {4} | the tensor under test | neighbour {4} ]
//                tensor(0)       tensor(1), 4 values in  tensor(2)
//
// So it does not start at the start of the memory, and a write that goes past its ends
// would show in a neighbour: every test that writes checks both neighbours after.

/// A buffer of three tensors, {4}, `shape`, {4}, all zeros. tensor(1) is the one under test.
static TensorBuffer buffer_with_neighbours(const Shape& shape, DType dtype = DType::Float32) {
    TensorBuffer buffer;
    buffer.add_many({{4}, shape, {4}});
    buffer.allocate(dtype);
    return buffer;
}

// ── making tensors ────────────────────────────────────────────────────────────

static void buffer_tensor_starts_as_zeros(void) {
    // Expect: tensor(1) of shape {2, 3} has 2 dims, 2 x 3 = 6 values, shape {2, 3}, all 0,
    // and starts 4 values into flat(), right after its neighbour.
    TensorBuffer buffer = buffer_with_neighbours({2, 3});
    Tensor t = buffer.tensor(1);
    ASSERT_EQ(t.dim(), 2);
    ASSERT_EQ(t.numel(), (size_t)6);
    ASSERT_EQ(t.shape(), Shape({2, 3}));
    ASSERT_EQ(t.tolist(), std::vector<float>(6, 0.0f));
    ASSERT_EQ(t.data(), buffer.flat().data() + 4);
}

static void buffer_tensor_strides_follow_row_order(void) {
    // Expect: {2, 3, 4} inside a buffer has 24 values and strides {12, 4, 1}, the same as a
    // tensor with memory of its own: one step along dim 0 skips 3 x 4 = 12 values.
    TensorBuffer buffer = buffer_with_neighbours({2, 3, 4});
    Tensor t = buffer.tensor(1);
    ASSERT_EQ(t.numel(), (size_t)24);
    ASSERT_EQ(t.stride(), Shape({12, 4, 1}));
    ASSERT_EQ(t.size(0), (size_t)2);
    ASSERT_EQ(t.size(1), (size_t)3);
    ASSERT_EQ(t.size(2), (size_t)4);
    ASSERT_EQ(t.stride(0), (size_t)12);
    ASSERT_EQ(t.stride(1), (size_t)4);
    ASSERT_EQ(t.stride(2), (size_t)1);
}

static void an_empty_tensor_can_take_a_buffer_tensor(void) {
    // Expect: Tensor t; has 0 dims, 0 values and no data; after t = buffer.tensor(1) it is
    // the {4} tensor: 4 values, starting 4 values into flat().
    TensorBuffer buffer = buffer_with_neighbours({4});
    Tensor t;
    ASSERT_EQ(t.dim(), 0);
    ASSERT_EQ(t.numel(), (size_t)0);
    ASSERT_EQ(t.data(), (float*)nullptr);
    t = buffer.tensor(1);
    ASSERT_EQ(t.numel(), (size_t)4);
    ASSERT_EQ(t.data(), buffer.flat().data() + 4);
}

// ── values in and out ─────────────────────────────────────────────────────────

static void buffer_tensor_values_come_back_as_they_went_in(void) {
    // Expect: 0..7 copied into tensor(1), a 2x4, comes back as 0..7; the neighbours stay 0.
    TensorBuffer buffer = buffer_with_neighbours({2, 4});
    Tensor t = buffer.tensor(1);
    t.copy_from_cpu(sample_data(8).data());
    ASSERT_EQ(t.tolist(), sample_data(8));
    ASSERT_EQ(buffer.tensor(0).tolist(), std::vector<float>(4, 0.0f));
    ASSERT_EQ(buffer.tensor(2).tolist(), std::vector<float>(4, 0.0f));
}

static void buffer_tensor_value_i_j_sits_at_i_times_stride_0_plus_j(void) {
    // Expect: in a 2x4 holding 0..7, [i][j] is data[i * 4 + j], so its value is i * 4 + j:
    // [1][2] is 6. The same rule inside a buffer, counted from the tensor's own start.
    TensorBuffer buffer = buffer_with_neighbours({2, 4});
    Tensor t = buffer.tensor(1);
    t.copy_from_cpu(sample_data(8).data());
    std::vector<float> values = t.tolist();
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 4; j++) ASSERT_EQ(values[i * t.stride(0) + j * t.stride(1)], (float)(i * 4 + j));
    }
    ASSERT_EQ(values[1 * t.stride(0) + 2 * t.stride(1)], 6.0f);
}

// ── view: the same values, another shape ──────────────────────────────────────

static void buffer_tensor_view_changes_the_shape_not_the_values(void) {
    // Expect: tensor(1), a 2x4, viewed as 4x2 has shape {4, 2}, strides {2, 1}, the same data
    // pointer and the same values 0..7; viewed as {8} it has shape {8}.
    TensorBuffer buffer = buffer_with_neighbours({2, 4});
    Tensor t = buffer.tensor(1);
    t.copy_from_cpu(sample_data(8).data());
    Tensor v = t.view({4, 2});
    ASSERT_EQ(v.shape(), Shape({4, 2}));
    ASSERT_EQ(v.stride(), Shape({2, 1}));
    ASSERT_EQ(v.data(), t.data());                         // not a copy
    ASSERT_EQ(v.tolist(), sample_data(8));
    ASSERT_EQ(t.view({8}).shape(), Shape({8}));
}

static void buffer_tensor_view_shares_the_values(void) {
    // Expect: 0..7 written through a view {8} of tensor(1) shows up in tensor(1); the
    // neighbours stay 0.
    TensorBuffer buffer = buffer_with_neighbours({2, 4});
    Tensor t = buffer.tensor(1);
    Tensor v = t.view({8});
    v.copy_from_cpu(sample_data(8).data());
    ASSERT_EQ(t.tolist(), sample_data(8));
    ASSERT_EQ(buffer.tensor(0).tolist(), std::vector<float>(4, 0.0f));
    ASSERT_EQ(buffer.tensor(2).tolist(), std::vector<float>(4, 0.0f));
}

static void buffer_tensor_view_keeps_the_memory_alive(void) {
    // Expect: a view of tensor(1) still reads 0..7 after the buffer is gone: the view shares
    // the buffer's memory, so the memory stays.
    Tensor v;
    {
        TensorBuffer buffer = buffer_with_neighbours({2, 4});
        Tensor t = buffer.tensor(1);
        t.copy_from_cpu(sample_data(8).data());
        v = t.view({8});
    }
    ASSERT_EQ(v.tolist(), sample_data(8));
}

// ── narrow: some rows ─────────────────────────────────────────────────────────

static void buffer_tensor_narrow_takes_rows(void) {
    // Expect: rows 1 and 2 of tensor(1), a 4x3 holding 0..11, are a 2x3 that starts 1 x 3 = 3
    // values after tensor(1) (so 4 + 3 = 7 into flat()) and holds 3..8.
    TensorBuffer buffer = buffer_with_neighbours({4, 3});
    Tensor t = buffer.tensor(1);
    t.copy_from_cpu(sample_data(12).data());
    Tensor rows = t.narrow(0, 1, 2);
    ASSERT_EQ(rows.shape(), Shape({2, 3}));
    ASSERT_EQ(rows.data(), t.data() + 3);
    ASSERT_EQ(rows.data(), buffer.flat().data() + 7);
    ASSERT_EQ(rows.tolist(), std::vector<float>({3, 4, 5, 6, 7, 8}));
}

static void buffer_tensor_narrow_then_view(void) {
    // Expect: values 8..19 of tensor(1), a flat 20 holding 0..19, seen as 3x4, start 8 values
    // after tensor(1) and hold 8..19.
    TensorBuffer buffer = buffer_with_neighbours({20});
    Tensor t = buffer.tensor(1);
    t.copy_from_cpu(sample_data(20).data());
    Tensor piece = t.narrow(0, 8, 12).view({3, 4});
    ASSERT_EQ(piece.shape(), Shape({3, 4}));
    ASSERT_EQ(piece.data(), t.data() + 8);
    ASSERT_EQ(piece.tolist(), std::vector<float>({8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}));
}

static void buffer_tensor_writing_a_narrow_leaves_the_rest_alone(void) {
    // Expect: 1s written into rows 1 and 2 of tensor(1), a 4x3 of zeros; rows 0 and 3 stay
    // 0, and so do the neighbours.
    TensorBuffer buffer = buffer_with_neighbours({4, 3});
    Tensor t = buffer.tensor(1);
    t.narrow(0, 1, 2).copy_from_cpu(std::vector<float>(6, 1.0f).data());
    ASSERT_EQ(t.tolist(), std::vector<float>({0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0}));
    ASSERT_EQ(buffer.tensor(0).tolist(), std::vector<float>(4, 0.0f));
    ASSERT_EQ(buffer.tensor(2).tolist(), std::vector<float>(4, 0.0f));
}

// ── from_blob: memory someone else owns ───────────────────────────────────────

static void from_blob_over_a_buffer_tensor_does_not_free_it(void) {
    // Expect: from_blob over the data of tensor(1) uses that pointer and reads 0..7; after the
    // from_blob tensor is gone, tensor(1) still holds 0..7: the buffer's memory was not freed.
    TensorBuffer buffer = buffer_with_neighbours({2, 4});
    Tensor t = buffer.tensor(1);
    t.copy_from_cpu(sample_data(8).data());
    {
        Tensor blob = Tensor::from_blob(t.data(), {2, 4});
        ASSERT_EQ(blob.data(), t.data());
        ASSERT_EQ(blob.tolist(), sample_data(8));
    }
    ASSERT_EQ(t.tolist(), sample_data(8));
}

// ── more ways to make tensors ─────────────────────────────────────────────────

static void buffer_tensor_zeros_like_and_empty_like_copy_shape_and_dtype(void) {
    // Expect: zeros_like and empty_like of tensor(1), a 2x5 int tensor of ones, are 2x5 int
    // tensors in memory of their own, outside the buffer; zeros_like holds ten 0s.
    TensorBuffer buffer = buffer_with_neighbours({2, 5}, DType::Int32);
    Tensor t = buffer.tensor(1);
    t.fill_(1);
    Tensor z = Tensor::zeros_like(t);
    Tensor e = Tensor::empty_like(t);
    ASSERT_EQ(z.shape(), Shape({2, 5}));
    ASSERT_EQ(z.dtype() == DType::Int32, true);
    ASSERT_EQ(e.shape(), Shape({2, 5}));
    ASSERT_EQ(e.dtype() == DType::Int32, true);
    ASSERT_EQ(z.data_ptr<int>() != t.data_ptr<int>(), true);
    ASSERT_EQ(z.tolist<int>(), std::vector<int>(10, 0));
}

static void buffer_tensor_fill_1_makes_ones(void) {
    // Expect: fill_(1) on tensor(1), a 2x2, gives {1, 1, 1, 1}, like Tensor::ones; the
    // neighbours stay 0.
    TensorBuffer buffer = buffer_with_neighbours({2, 2});
    Tensor t = buffer.tensor(1);
    t.fill_(1);
    ASSERT_EQ(t.tolist(), std::vector<float>({1, 1, 1, 1}));
    ASSERT_EQ(buffer.tensor(0).tolist(), std::vector<float>(4, 0.0f));
    ASSERT_EQ(buffer.tensor(2).tolist(), std::vector<float>(4, 0.0f));
}

// ── int tensors ───────────────────────────────────────────────────────────────

static void buffer_tensor_holds_ints(void) {
    // Expect: token ids {3, 464, 50256, 0} go into tensor(1) of an Int32 buffer and come back
    // the same; the neighbours stay 0.
    TensorBuffer buffer = buffer_with_neighbours({4}, DType::Int32);
    Tensor tokens = buffer.tensor(1);
    std::vector<int> ids = {3, 464, 50256, 0};
    tokens.copy_from_cpu(ids.data());
    ASSERT_EQ(tokens.dtype() == DType::Int32, true);
    ASSERT_EQ(tokens.tolist<int>(), ids);
    ASSERT_EQ(buffer.tensor(0).tolist<int>(), std::vector<int>(4, 0));
    ASSERT_EQ(buffer.tensor(2).tolist<int>(), std::vector<int>(4, 0));
}

// ── setting values ────────────────────────────────────────────────────────────

static void buffer_tensor_zero_leaves_the_neighbours_alone(void) {
    // Expect: with every value of the buffer 9 and 0..5 in tensor(1), a 2x3, zero_() on
    // tensor(1) gives six 0s; the neighbours keep their 9s.
    TensorBuffer buffer = buffer_with_neighbours({2, 3});
    buffer.flat().fill_(9);
    Tensor t = buffer.tensor(1);
    t.copy_from_cpu(sample_data(6).data());
    t.zero_();
    ASSERT_EQ(t.tolist(), std::vector<float>(6, 0.0f));
    ASSERT_EQ(buffer.tensor(0).tolist(), std::vector<float>(4, 9.0f));
    ASSERT_EQ(buffer.tensor(2).tolist(), std::vector<float>(4, 9.0f));
}

static void buffer_tensor_fill_sets_every_value(void) {
    // Expect: fill_(1.5) gives six 1.5s in tensor(1) of a float buffer, a 2x3; fill_(7) gives
    // four 7s in tensor(1) of an int buffer, a {4}; the neighbours stay 0 in both.
    TensorBuffer floats = buffer_with_neighbours({2, 3});
    floats.tensor(1).fill_(1.5);
    ASSERT_EQ(floats.tensor(1).tolist(), std::vector<float>(6, 1.5f));
    ASSERT_EQ(floats.tensor(0).tolist(), std::vector<float>(4, 0.0f));
    ASSERT_EQ(floats.tensor(2).tolist(), std::vector<float>(4, 0.0f));
    TensorBuffer ints = buffer_with_neighbours({4}, DType::Int32);
    ints.tensor(1).fill_(7);
    ASSERT_EQ(ints.tensor(1).tolist<int>(), std::vector<int>(4, 7));
    ASSERT_EQ(ints.tensor(0).tolist<int>(), std::vector<int>(4, 0));
    ASSERT_EQ(ints.tensor(2).tolist<int>(), std::vector<int>(4, 0));
}

static void buffer_tensor_fill_on_a_narrow_fills_only_those_rows(void) {
    // Expect: fill_(2) on rows 1 and 2 of tensor(1), a 4x3 of zeros; rows 0 and 3 stay 0, and
    // so do the neighbours.
    TensorBuffer buffer = buffer_with_neighbours({4, 3});
    Tensor t = buffer.tensor(1);
    t.narrow(0, 1, 2).fill_(2);
    ASSERT_EQ(t.tolist(), std::vector<float>({0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0}));
    ASSERT_EQ(buffer.tensor(0).tolist(), std::vector<float>(4, 0.0f));
    ASSERT_EQ(buffer.tensor(2).tolist(), std::vector<float>(4, 0.0f));
}

// ── random values ─────────────────────────────────────────────────────────────

static void buffer_tensor_the_same_seed_gives_the_same_numbers(void) {
    // Expect: after manual_seed(7), normal_(0, 1) on tensor(1), a {1000}, gives the same 1000
    // numbers as randn({1000}) after manual_seed(7); the neighbours stay 0.
    Tensor::manual_seed(7);
    std::vector<float> expected = Tensor::randn({1000}).tolist();
    TensorBuffer buffer = buffer_with_neighbours({1000});
    Tensor::manual_seed(7);
    buffer.tensor(1).normal_(0.0f, 1.0f);
    ASSERT_EQ(buffer.tensor(1).tolist(), expected);
    ASSERT_EQ(buffer.tensor(0).tolist(), std::vector<float>(4, 0.0f));
    ASSERT_EQ(buffer.tensor(2).tolist(), std::vector<float>(4, 0.0f));
}

static void buffer_tensor_normal_has_the_mean_and_std_asked_for(void) {
    // Expect: 100000 values of normal_(0, 0.02) in tensor(1) have a mean within 0.0005 of 0
    // and a std within 2% of 0.02; the neighbours stay 0.
    TensorBuffer buffer = buffer_with_neighbours({100000});
    Tensor::manual_seed(1);
    buffer.tensor(1).normal_(0.0f, 0.02f);
    std::vector<float> values = buffer.tensor(1).tolist();
    double sum = 0.0, sum_squares = 0.0;
    for (float value : values) sum += value;
    double mean = sum / values.size();
    for (float value : values) sum_squares += (value - mean) * (value - mean);
    double deviation = sqrt(sum_squares / values.size());
    ASSERT_NEAR(mean, 0.0, 0.0005);
    ASSERT_NEAR(deviation, 0.02, 0.02 * 0.02);                 // 2% of 0.02
    ASSERT_EQ(buffer.tensor(0).tolist(), std::vector<float>(4, 0.0f));
    ASSERT_EQ(buffer.tensor(2).tolist(), std::vector<float>(4, 0.0f));
}

// ── copies on the GPU ─────────────────────────────────────────────────────────

static void buffer_tensor_copy_copies_gpu_to_gpu(void) {
    // Expect: copy_ of a 2x3 holding 0..5 into tensor(1), a 3x2, gives tensor(1) the values
    // 0..5 and keeps its shape {3, 2}; the neighbours stay 0. Copying tensor(1) out into a
    // tensor of its own gives that tensor 0..5 too.
    TensorBuffer buffer = buffer_with_neighbours({3, 2});
    Tensor source = Tensor::zeros({2, 3});
    source.copy_from_cpu(sample_data(6).data());
    Tensor t = buffer.tensor(1);
    t.copy_(source);
    ASSERT_EQ(t.tolist(), sample_data(6));
    ASSERT_EQ(t.shape(), Shape({3, 2}));
    ASSERT_EQ(buffer.tensor(0).tolist(), std::vector<float>(4, 0.0f));
    ASSERT_EQ(buffer.tensor(2).tolist(), std::vector<float>(4, 0.0f));
    Tensor out = Tensor::zeros({6});
    out.copy_(t);
    ASSERT_EQ(out.tolist(), sample_data(6));
}

static void buffer_tensor_clone_is_outside_the_buffer(void) {
    // Expect: a clone of tensor(1) holding 0..3 holds 0..3, in memory of its own: it still
    // does after every value of the buffer is zeroed.
    TensorBuffer buffer = buffer_with_neighbours({4});
    Tensor t = buffer.tensor(1);
    t.copy_from_cpu(sample_data(4).data());
    Tensor copy = t.clone();
    buffer.flat().zero_();
    ASSERT_EQ(copy.tolist(), sample_data(4));
    ASSERT_EQ(copy.data() != t.data(), true);
}

// ── reading one value ─────────────────────────────────────────────────────────

static void buffer_tensor_item_reads_the_one_value(void) {
    // Expect: item() of tensor(1), a float {1} holding 42.5, is 42.5; item<int>() of tensor(1)
    // of an int buffer, a {1} holding 50256, is 50256.
    TensorBuffer floats = buffer_with_neighbours({1});
    float value = 42.5f;
    floats.tensor(1).copy_from_cpu(&value);
    ASSERT_EQ(floats.tensor(1).item(), 42.5f);
    TensorBuffer ints = buffer_with_neighbours({1}, DType::Int32);
    int id = 50256;
    ints.tensor(1).copy_from_cpu(&id);
    ASSERT_EQ(ints.tensor(1).item<int>(), 50256);
}

static void buffer_tensor_indexing_picks_rows_and_values(void) {
    // Expect: in tensor(1), a 2x4 holding 0..7, t[1] is row 1: shape {4}, values 4..7,
    // starting 4 values after tensor(1) (8 into flat()); t[1][2] has no dims and 1 value, 6.
    TensorBuffer buffer = buffer_with_neighbours({2, 4});
    Tensor t = buffer.tensor(1);
    t.copy_from_cpu(sample_data(8).data());
    Tensor row = t[1];
    ASSERT_EQ(row.shape(), Shape({4}));
    ASSERT_EQ(row.tolist(), std::vector<float>({4, 5, 6, 7}));
    ASSERT_EQ(row.data(), t.data() + 4);
    ASSERT_EQ(row.data(), buffer.flat().data() + 8);
    ASSERT_EQ(t[1][2].dim(), 0);
    ASSERT_EQ(t[1][2].numel(), (size_t)1);
    ASSERT_EQ(t[1][2].item(), 6.0f);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void) {
    add_returns_each_tensors_index();
    add_many_returns_the_first_index();
    numel_is_known_before_allocate();
    allocate_makes_one_flat_tensor_of_zeros();
    each_tensor_has_its_own_shape();
    tensors_sit_one_after_another();
    each_tensor_sees_its_part_of_the_memory();
    writing_a_tensor_writes_only_its_part();
    zeroing_all_zeroes_every_tensor();
    a_tensor_keeps_the_memory_alive();
    matrices_are_the_tensors_with_2_or_more_dims();
    zeros_like_copies_the_layout_not_the_memory();
    an_int_buffer_holds_ints();

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
