// check.h — what every test file here uses: ASSERT_EQ and ASSERT_NEAR, which print
// every check and stop the program on the first FAIL, and sample_data.
//
//     normal_has_the_mean_and_std_asked_for                            ← the test case
//         input: mean      expected: 0 +/- 0.0005   actual: 6.1e-05   ok  (line 293)
#ifndef TESTS_CHECK_H
#define TESTS_CHECK_H

#undef NDEBUG                     // asserts on, whatever the build flags
#include <cassert>
#include <cmath>
#include <stdio.h>
#include <vector>

/// Sample data for the tests: 0, 1, 2, ..., n - 1 on the CPU. Each value is its own position,
/// so after a view or a narrow you can see where every value came from.
static std::vector<float> sample_data(size_t n) {
    std::vector<float> values(n);
    for (size_t i = 0; i < n; i++) values[i] = (float)i;
    return values;
}

// ── printing values ───────────────────────────────────────────────────────────

static void print_value(bool value) { printf("%s", value ? "true" : "false"); }
static void print_value(int value) { printf("%d", value); }
static void print_value(size_t value) { printf("%zu", value); }
static void print_value(float value) { printf("%g", value); }
static void print_value(const void* pointer) { printf("%p", pointer); }

/// {a, b, c}; a long vector shows its first 16 values and how many there are.
template <typename T>
static void print_value(const std::vector<T>& values) {
    printf("{");
    for (size_t i = 0; i < values.size() && i < 16; i++) {
        if (i > 0) printf(", ");
        print_value(values[i]);
    }
    if (values.size() > 16) printf(", ... (%zu values)", values.size());
    printf("}");
}

// ── ASSERT_EQ, ASSERT_NEAR: print every check, stop on a FAIL ─────────────────

/// The test case's name, once, before its first check.
static void print_test_name(const char* test) {
    static const char* last_test = nullptr;
    if (test != last_test) printf("\n%s\n", test);
    last_test = test;
}

/// One check: the input (what is checked), the expected and the actual value, ok or FAIL.
template <typename A, typename B>
static void check_equal(const char* test, const char* input, const A& actual, const B& expected, int line) {
    print_test_name(test);
    bool equal = actual == expected;
    printf("    input: %-36s expected: ", input);
    print_value(expected);
    printf("   actual: ");
    print_value(actual);
    printf("   %s  (line %d)\n", equal ? "ok" : "FAIL", line);
    fflush(stdout);                       // printed before assert stops the program
    assert(equal && "the values are printed above");
}

/// The same for a number that only has to be close: expected +/- tolerance.
static void check_near(const char* test, const char* input, double actual, double expected, double tolerance,
                       int line) {
    print_test_name(test);
    bool near = fabs(actual - expected) <= tolerance;
    printf("    input: %-36s expected: %g +/- %g   actual: %g   %s  (line %d)\n", input, expected, tolerance, actual,
           near ? "ok" : "FAIL", line);
    fflush(stdout);
    assert(near && "the values are printed above");
}

// __func__ is the name of the function the check is in: the test case.
// Lists inside need ( ) around { }: ASSERT_EQ(v.shape(), Shape({4, 2})).
#define ASSERT_EQ(actual, expected) check_equal(__func__, #actual, (actual), (expected), __LINE__)
#define ASSERT_NEAR(actual, expected, tolerance) \
    check_near(__func__, #actual, (actual), (expected), (tolerance), __LINE__)

#endif // TESTS_CHECK_H
