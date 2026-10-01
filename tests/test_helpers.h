// test_helpers.h — what every test file here uses: ASSERT_EQ, ASSERT_NEAR and ASSERT_FAR,
// which print every check and stop the program on the first FAIL, and sample_data.
//
//     normal_has_the_mean_and_std_asked_for                            ← the test case
//         input: mean      expected: 0 +/- 0.0005   actual: 6.1e-05   ok  (line 293)
//
// Two lists print both, then how far apart they are:
//
//         input: weight.tolist()   expected: {0.79, ...}   actual: {0.79, ...}   difference: 2.6e-08 (at most 1e-05)   ok
#ifndef TESTS_TEST_HELPERS_H
#define TESTS_TEST_HELPERS_H

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

// ── ASSERT_EQ, ASSERT_NEAR, ASSERT_FAR: print every check, stop on a FAIL ─────

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

/// The largest difference between two lists of values, as a fraction of the largest
/// expected value: 1e-5 means they agree to about 5 digits. NaN if `actual` has a NaN.
static double relative_difference(const std::vector<float>& actual, const std::vector<float>& expected) {
    double largest_difference = 0.0, largest_value = 0.0;
    for (size_t i = 0; i < expected.size(); i++) {
        double difference = fabs((double)actual[i] - expected[i]);
        if (std::isnan(difference) || difference > largest_difference) largest_difference = difference;
        largest_value = fmax(largest_value, fabs((double)expected[i]));
    }
    return largest_value > 0 ? largest_difference / largest_value : largest_difference;
}

/// Two lists, both printed, then their relative_difference. close: it must be at most
/// `limit` (the same values, up to rounding); not close: more than `limit` (far apart).
static void check_lists(const char* test, const char* input, const std::vector<float>& actual,
                        const std::vector<float>& expected, double limit, bool close, int line) {
    print_test_name(test);
    bool same_size = actual.size() == expected.size();
    double difference = same_size ? relative_difference(actual, expected) : NAN;
    bool passed = same_size && (close ? difference <= limit : difference > limit);   // NaN fails both
    printf("    input: %-36s expected: ", input);
    print_value(expected);
    printf("   actual: ");
    print_value(actual);
    printf("   difference: %g (%s %g)   %s  (line %d)\n", difference, close ? "at most" : "more than", limit,
           passed ? "ok" : "FAIL", line);
    fflush(stdout);
    assert(passed && "the values are printed above");
}

/// Two lists that must be close: expected +/- tolerance, as a fraction of the largest value.
static void check_near(const char* test, const char* input, const std::vector<float>& actual,
                       const std::vector<float>& expected, double tolerance, int line) {
    check_lists(test, input, actual, expected, tolerance, true, line);
}

// __func__ is the name of the function the check is in: the test case.
// Lists inside need ( ) around { }: ASSERT_EQ(v.shape(), Shape({4, 2})).
// ASSERT_NEAR takes two numbers or two lists of floats; ASSERT_FAR takes two lists.
#define ASSERT_EQ(actual, expected) check_equal(__func__, #actual, (actual), (expected), __LINE__)
#define ASSERT_NEAR(actual, expected, tolerance) \
    check_near(__func__, #actual, (actual), (expected), (tolerance), __LINE__)
#define ASSERT_FAR(actual, expected, bound) \
    check_lists(__func__, #actual, (actual), (expected), (bound), false, __LINE__)

#endif // TESTS_TEST_HELPERS_H
