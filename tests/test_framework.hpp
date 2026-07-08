#ifndef NEUROFLOW_TEST_FRAMEWORK_HPP
#define NEUROFLOW_TEST_FRAMEWORK_HPP

#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace neuroflow_test {

struct TestCase {
    std::string name;
    std::function<void()> func;
};

static std::vector<TestCase>& get_tests() {
    static std::vector<TestCase> tests;
    return tests;
}

static int& fail_count() {
    static int count = 0;
    return count;
}

struct TestRegistrar {
    TestRegistrar(const char* name, std::function<void()> func) {
        get_tests().push_back({name, func});
    }
};

#define TEST(group, name) \
    void test_##group##_##name(); \
    static neuroflow_test::TestRegistrar reg_##group##_##name(#group "." #name, test_##group##_##name); \
    void test_##group##_##name()

#define EXPECT_EQ(a, b) do { \
    auto _a = (a); auto _b = (b); \
    if (!(_a == _b)) { \
        std::cerr << "  FAIL: " << #a << " == " << #b \
                  << " (got " << _a << " vs " << _b << ")" << std::endl; \
        neuroflow_test::fail_count()++; \
    } \
} while(0)

#define EXPECT_NE(a, b) do { \
    auto _a = (a); auto _b = (b); \
    if (_a == _b) { \
        std::cerr << "  FAIL: " << #a << " != " << #b << std::endl; \
        neuroflow_test::fail_count()++; \
    } \
} while(0)

#define EXPECT_GT(a, b) do { \
    auto _a = (a); auto _b = (b); \
    if (!(_a > _b)) { \
        std::cerr << "  FAIL: " << #a << " > " << #b \
                  << " (got " << _a << " vs " << _b << ")" << std::endl; \
        neuroflow_test::fail_count()++; \
    } \
} while(0)

#define EXPECT_LT(a, b) do { \
    auto _a = (a); auto _b = (b); \
    if (!(_a < _b)) { \
        std::cerr << "  FAIL: " << #a << " < " << #b \
                  << " (got " << _a << " vs " << _b << ")" << std::endl; \
        neuroflow_test::fail_count()++; \
    } \
} while(0)

#define EXPECT_NEAR(a, b, tol) do { \
    auto _a = static_cast<double>(a); auto _b = static_cast<double>(b); auto _t = static_cast<double>(tol); \
    if (std::abs(_a - _b) > _t) { \
        std::cerr << "  FAIL: " << #a << " ~= " << #b \
                  << " (got " << _a << " vs " << _b << ", tol=" << _t << ")" << std::endl; \
        neuroflow_test::fail_count()++; \
    } \
} while(0)

#define EXPECT_TRUE(cond) do { \
    if (!(cond)) { \
        std::cerr << "  FAIL: " << #cond << std::endl; \
        neuroflow_test::fail_count()++; \
    } \
} while(0)

#define EXPECT_FALSE(cond) do { \
    if (cond) { \
        std::cerr << "  FAIL: NOT(" << #cond << ")" << std::endl; \
        neuroflow_test::fail_count()++; \
    } \
} while(0)

#define EXPECT_THROW(expr, exc_type) do { \
    bool _caught = false; \
    try { expr; } catch (const exc_type&) { _caught = true; } \
    if (!_caught) { \
        std::cerr << "  FAIL: " << #expr << " did not throw " << #exc_type << std::endl; \
        neuroflow_test::fail_count()++; \
    } \
} while(0)

#define RUN_ALL_TESTS() do { \
    int _passed = 0; int _failed = 0; \
    for (auto& _tc : neuroflow_test::get_tests()) { \
        neuroflow_test::fail_count() = 0; \
        std::cout << "[ RUN    ] " << _tc.name << std::endl; \
        _tc.func(); \
        if (neuroflow_test::fail_count() == 0) { \
            std::cout << "[     OK ] " << _tc.name << std::endl; \
            _passed++; \
        } else { \
            std::cout << "[ FAILED ] " << _tc.name << std::endl; \
            _failed++; \
        } \
    } \
    std::cout << "\n=== " << _passed << " passed, " << _failed << " failed ===" << std::endl; \
    return _failed > 0 ? 1 : 0; \
} while(0)

} // namespace neuroflow_test

#endif