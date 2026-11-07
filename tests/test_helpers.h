// test_helpers.h — what every test file here uses: ASSERT_EQ, ASSERT_NEAR and ASSERT_FAR,
// which print every check and stop the program on the first FAIL, sample_data, and start_run_log, which
// copies everything a run prints into a new log file.
//
//     normal_has_the_mean_and_std_asked_for                            ← the test case
//         input: mean_val      expected: 0 +/- 0.0005   actual: 6.1e-05   ok  (line 293)
//
// Two lists print both, then how far apart they are:
//
//         input: weight_tensor.tolist()   expected: {0.79, ...}   actual: {0.79, ...}   difference: 2.6e-08 (at most 1e-05)   ok
#ifndef TESTS_TEST_HELPERS_H
#define TESTS_TEST_HELPERS_H

#undef NDEBUG                     // asserts on, whatever the build flags
#include <cassert>
#include <cmath>
#include <ctime>                  // the log file's date and time
#include <filesystem>             // std::filesystem::create_directories, for the log's folder
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#ifndef _WIN32
#include <unistd.h>               // dup, dup2, STDOUT_FILENO: the console and the log both get every line
#endif

#define PRINTED_LIST_VALUES 16    // a long list prints only its first this many values
#define TESTS_LOG_FOLDER "logs/tests"   // where start_run_log puts a test's logs (git ignores logs/)

// ── a log file for each run ───────────────────────────────────────────────────

static FILE* run_log_tee_ptr = nullptr;  // the `tee` that copies this program's output into its log
static int console_stdout_val = -1;      // the console's stdout and stderr, kept to give them back at the end
static int console_stderr_val = -1;

/// Ends the log at exit: gives stdout and stderr back to the console, so `tee` sees the end of the output, and
/// waits for it to write its last lines.
static void end_run_log(void) {
#ifndef _WIN32
    // 1. everything printed so far goes out, then the console gets stdout and stderr back
    fflush(stdout);
    fflush(stderr);
    dup2(console_stdout_val, STDOUT_FILENO);
    dup2(console_stderr_val, STDERR_FILENO);
    // 2. tee has the whole output: wait for it to finish
    pclose(run_log_tee_ptr);
    run_log_tee_ptr = nullptr;
#endif
}

/// Starts a log file for this run, so its output need not be copied from the console: from here on everything
/// the program prints, stdout and stderr, still shows on the console and also goes into a new file
///
///     <folder_ptr>/<name>_<date>_<time>.<extension_ptr>      e.g. logs/tests/test_gelu_2026-11-07_153012.log
///
/// where name is the program's file name without folder and ".cu" (from __FILE__), less `drop_prefix_ptr` if it
/// starts with it, plus _rank<r> under torchrun (each rank its own file). A FAIL's assert message and a CUDA
/// error land in it too, and stdout is flushed every line, so a run that stops still has its log up to there.
/// It needs `tee` (Linux, Colab); on Windows the output only goes to the console. Gives the log's path.
///
///   folder_ptr        the folder, made if missing
///   file_ptr          __FILE__ of the program's main
///   drop_prefix_ptr   taken off the front of the name: "tune_" gives gelu, not tune_gelu
///   extension_ptr     "log" for tests, "txt" for tunes' results
static std::string start_run_log(const char* folder_ptr, const char* file_ptr, const char* drop_prefix_ptr = "",
                                 const char* extension_ptr = "log") {
    // 1. the name: the program's file name, without folder, ".cu" or the prefix; the rank under torchrun
    std::string name_val = std::filesystem::path(file_ptr).stem().string();
    std::string prefix_val = drop_prefix_ptr;
    if (!prefix_val.empty() && name_val.rfind(prefix_val, 0) == 0) name_val = name_val.substr(prefix_val.size());
    const char* rank_ptr = getenv("RANK");
    if (rank_ptr != nullptr) name_val += std::string("_rank") + rank_ptr;
    // 2. the date and time
    char stamp_list[32];
    time_t now_val = time(nullptr);
    strftime(stamp_list, sizeof(stamp_list), "%Y-%m-%d_%H%M%S", localtime(&now_val));
    std::filesystem::create_directories(folder_ptr);
    std::string path_val = std::string(folder_ptr) + "/" + name_val + "_" + stamp_list + "." + extension_ptr;
#ifdef _WIN32
    printf("log: none on Windows (needs tee); the output is only on the console\n");
#else
    // 3. `tee` writes whatever it is given to the console and to the file; stdout and stderr now go to it
    fflush(stdout);
    fflush(stderr);
    run_log_tee_ptr = popen(("tee '" + path_val + "'").c_str(), "w");
    if (run_log_tee_ptr == nullptr) {
        printf("log: could not start tee for %s; the output is only on the console\n", path_val.c_str());
        return path_val;
    }
    console_stdout_val = dup(STDOUT_FILENO);
    console_stderr_val = dup(STDERR_FILENO);
    dup2(fileno(run_log_tee_ptr), STDOUT_FILENO);
    dup2(fileno(run_log_tee_ptr), STDERR_FILENO);
    setvbuf(stdout, nullptr, _IOLBF, 0);    // a pipe is buffered in blocks: by lines, a FAIL loses nothing
    atexit(end_run_log);
    printf("log: %s\n", path_val.c_str());
#endif
    return path_val;
}

/// Sample data for the tests: 0, 1, 2, ..., n - 1 on the CPU. Each value is its own position,
/// so after a view or a narrow you can see where every value came from.
///
///   n_val   how many values
static std::vector<float> sample_data(size_t n_val) {
    std::vector<float> values_list(n_val);
    for (size_t i_val = 0; i_val < n_val; i_val++) values_list[i_val] = (float)i_val;
    return values_list;
}

// ── printing values ───────────────────────────────────────────────────────────

static void print_value(bool value_val) { printf("%s", value_val ? "true" : "false"); }
static void print_value(int value_val) { printf("%d", value_val); }
static void print_value(size_t value_val) { printf("%zu", value_val); }
static void print_value(long value_val) { printf("%ld", value_val); }             // int64_t on Linux
static void print_value(long long value_val) { printf("%lld", value_val); }       // int64_t on Windows
static void print_value(float value_val) { printf("%g", value_val); }
static void print_value(const void* value_ptr) { printf("%p", value_ptr); }

/// Prints {a, b, c}; a long list shows its first PRINTED_LIST_VALUES values and how many there are.
///
///   values_list   the list
template <typename T>
static void print_value(const std::vector<T>& values_list) {
    printf("{");
    for (size_t i_val = 0; i_val < values_list.size() && i_val < PRINTED_LIST_VALUES; i_val++) {
        if (i_val > 0) printf(", ");
        print_value(values_list[i_val]);
    }
    if (values_list.size() > PRINTED_LIST_VALUES) printf(", ... (%zu values)", values_list.size());
    printf("}");
}

// ── ASSERT_EQ, ASSERT_NEAR, ASSERT_FAR: print every check, stop on a FAIL ─────

/// Prints the test case's name, once, before its first check.
///
///   test_ptr   the test case's name: __func__
static void print_test_name(const char* test_ptr) {
    static const char* last_test_ptr = nullptr;
    if (test_ptr != last_test_ptr) printf("\n%s\n", test_ptr);
    last_test_ptr = test_ptr;
}

/// Checks one value: prints the input (what is checked), the expected and the actual value, ok or FAIL.
///
///   test_ptr, input_ptr      the test case's name, and the checked expression as text
///   actual_val, expected_val what came out, and what should have
///   line_val                 the check's line
template <typename A, typename B>
static void check_equal(const char* test_ptr, const char* input_ptr, const A& actual_val, const B& expected_val,
                        int line_val) {
    print_test_name(test_ptr);
    bool equal_val = actual_val == expected_val;
    printf("    input: %-36s expected: ", input_ptr);
    print_value(expected_val);
    printf("   actual: ");
    print_value(actual_val);
    printf("   %s  (line %d)\n", equal_val ? "ok" : "FAIL", line_val);
    fflush(stdout);                       // printed before assert stops the program
    assert(equal_val && "the values are printed above");
}

/// Checks a number that only has to be close: expected +/- tolerance.
///
///   test_ptr, input_ptr                        as in check_equal
///   actual_val, expected_val, tolerance_val    what came out, what should have, and how far off it may be
///   line_val                                   the check's line
static void check_near(const char* test_ptr, const char* input_ptr, double actual_val, double expected_val,
                       double tolerance_val, int line_val) {
    print_test_name(test_ptr);
    bool near_val = fabs(actual_val - expected_val) <= tolerance_val;
    printf("    input: %-36s expected: %g +/- %g   actual: %g   %s  (line %d)\n", input_ptr, expected_val, tolerance_val,
           actual_val, near_val ? "ok" : "FAIL", line_val);
    fflush(stdout);
    assert(near_val && "the values are printed above");
}

/// Gives the largest difference between two lists of values, as a fraction of the largest expected value:
/// 1e-5 means they agree to about 5 digits. NaN if `actual_list` has a NaN.
///
///   actual_list, expected_list   the two lists, the same length
static double relative_difference(const std::vector<float>& actual_list, const std::vector<float>& expected_list) {
    double largest_difference_val = 0.0;
    double largest_value_val = 0.0;
    for (size_t i_val = 0; i_val < expected_list.size(); i_val++) {
        double difference_val = fabs((double)actual_list[i_val] - expected_list[i_val]);
        if (std::isnan(difference_val) || difference_val > largest_difference_val) largest_difference_val = difference_val;
        largest_value_val = fmax(largest_value_val, fabs((double)expected_list[i_val]));
    }
    return largest_value_val > 0 ? largest_difference_val / largest_value_val : largest_difference_val;
}

/// Checks two lists: prints both, then their relative_difference. close: it must be at most `limit_val` (the
/// same values, up to rounding); not close: more than `limit_val` (far apart).
///
///   test_ptr, input_ptr            as in check_equal
///   actual_list, expected_list     the two lists
///   limit_val, close_val           the bound, and which side of it passes
///   line_val                       the check's line
static void check_lists(const char* test_ptr, const char* input_ptr, const std::vector<float>& actual_list,
                        const std::vector<float>& expected_list, double limit_val, bool close_val, int line_val) {
    print_test_name(test_ptr);
    bool same_size_val = actual_list.size() == expected_list.size();
    double difference_val = same_size_val ? relative_difference(actual_list, expected_list) : NAN;
    bool passed_val = same_size_val && (close_val ? difference_val <= limit_val : difference_val > limit_val);   // NaN fails both
    printf("    input: %-36s expected: ", input_ptr);
    print_value(expected_list);
    printf("   actual: ");
    print_value(actual_list);
    printf("   difference: %g (%s %g)   %s  (line %d)\n", difference_val, close_val ? "at most" : "more than", limit_val,
           passed_val ? "ok" : "FAIL", line_val);
    fflush(stdout);
    assert(passed_val && "the values are printed above");
}

/// Checks two lists that must be close: expected +/- tolerance, as a fraction of the largest value.
///
///   test_ptr, input_ptr                          as in check_equal
///   actual_list, expected_list, tolerance_val    the two lists, and how far apart they may be
///   line_val                                     the check's line
static void check_near(const char* test_ptr, const char* input_ptr, const std::vector<float>& actual_list,
                       const std::vector<float>& expected_list, double tolerance_val, int line_val) {
    check_lists(test_ptr, input_ptr, actual_list, expected_list, tolerance_val, true, line_val);
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
