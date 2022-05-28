#include <gtest/gtest.h>
#include <ghostbasil/util/macros.hpp>
#include <ghostbasil/util/algorithm.hpp>

namespace ghostbasil {
namespace util {
namespace {

struct KImaxFixture
    : ::testing::Test
{
    template <class VType>
    void test(
            VType actual,
            size_t s,
            VType expected)
    {
        std::sort(expected.begin(), expected.end());
        std::sort(actual.begin(), std::next(actual.begin(), s));
        
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(expected[i], actual[i]);
        }
    }
};

TEST_F(KImaxFixture, k_imax_less_than_k)
{
    size_t k = 6;
    auto skip = [&](auto i) {
        return i % 2 == 0;   
    };
    std::vector<int> v({1,3,8,-3,2});
    std::vector<uint32_t> actual(k);
    auto s = k_imax(v, skip, k, actual.begin());
    EXPECT_EQ(s, 2);
    
    std::vector<uint32_t> expected(2);
    expected[0] = 1;
    expected[1] = 3;
    
    test(actual, s, expected);
}

TEST_F(KImaxFixture, k_imax_eq_k)
{
    size_t k = 2;
    auto skip = [&](auto i) {
        return i % 2 == 0;   
    };
    std::vector<int> v({1,3,8,-3,2});
    std::vector<uint32_t> actual(k);
    auto s = k_imax(v, skip, k, actual.begin());
    EXPECT_EQ(s, 2);
    
    std::vector<uint32_t> expected(2);
    expected[0] = 1;
    expected[1] = 3;
    
    test(actual, s, expected);
}

TEST_F(KImaxFixture, k_imax_ge_k)
{
    size_t k = 3;
    auto skip = [&](auto i) {
        return i == 0;
    };
    std::vector<int> v({1,3,-1,8,-3,2});
    std::vector<uint32_t> actual(k);
    auto s = k_imax(v, skip, k, actual.begin());
    EXPECT_EQ(s, k);
    
    std::vector<uint32_t> expected(3);
    expected[0] = 3;
    expected[1] = 1;
    expected[2] = 5;
    
    test(actual, s, expected);
}


TEST_F(KImaxFixture, k_imax_ge_k_larger)
{
    size_t k = 4;
    auto skip = [&](auto i) {
        return (i == 0) || (i == 3);
    };
    std::vector<int> v({1,3,-1,8,-3,2,-2,-5,-10,30,2,35});
    std::vector<uint32_t> actual(k);
    auto s = k_imax(v, skip, k, actual.begin());
    EXPECT_EQ(s, k);
    
    std::vector<uint32_t> expected(k);
    expected[0] = v.size()-1;
    expected[1] = v.size()-3;
    expected[2] = 1;
    expected[3] = v.size()-2;
    
    test(actual, s, expected);
}

}
} // namespace util
} // namespace ghostbasil
