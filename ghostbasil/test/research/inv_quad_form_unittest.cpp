// Assesses the accuracy of the estimate of 
//      v^T A^{-1} v
// using the e_{-1} in Eq (8) of the following paper:
// https://www.mdpi.com/2227-7390/9/12/1432/pdf
//
// The following isn't really a test.
// It simply outputs the results of the estimation method
// and the ground truth via taking an inverse.
// We don't necessarily care for any measure of accuracy,
// but just to see the general trend.
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <ghostbasil/util/macros.hpp>

namespace ghostbasil {
namespace {

struct InvQuadFormFixture
    : ::testing::Test,
      testing::WithParamInterface<
        std::tuple<size_t, size_t, double> >
{
    using mat_t = Eigen::MatrixXd;
    using vec_t = Eigen::VectorXd;

    auto make_input(
            size_t seed,
            size_t p,
            double eps)
    {
        srand(seed);
        mat_t A = mat_t::Random(p, p);
        A = (A.transpose() * A) / p;
        A.diagonal().array() += eps; // make PD
        vec_t v = vec_t::Random(p);
        return std::make_tuple(A, v);
    }

    void test(
            const mat_t& A,
            const vec_t& v)
    {
        // Estimation 
        auto v_norm_sq = v.squaredNorm();
        vec_t Av = A * v;
        auto Av_norm_sq = Av.squaredNorm();
        auto vTAv = v.dot(Av);
        double actual = 0;
        if (v_norm_sq > 0) {
            if (vTAv <= 0) actual = std::numeric_limits<double>::infinity();
            else {
                auto v_norm_sq_div_vTAv = v_norm_sq / vTAv;
                auto v_norm_sq_div_vTAv_pow3 = 
                    v_norm_sq_div_vTAv * v_norm_sq_div_vTAv * v_norm_sq_div_vTAv;
                actual = v_norm_sq_div_vTAv_pow3 * Av_norm_sq;
            }
        }
       
        // Expected
        Eigen::FullPivLU<mat_t> lu(A);
        auto inv_A_v = lu.solve(v);
        double expected = v.dot(inv_A_v);

        PRINT(actual);
        PRINT(expected);
    }
};

TEST_P(InvQuadFormFixture, inv_quad_form_test)
{
    size_t seed;
    size_t p;
    double eps;
    std::tie(seed, p, eps) = GetParam();
    mat_t A;
    vec_t v;
    std::tie(A, v) = make_input(seed, p, eps);
    
    test(A, v);
}

INSTANTIATE_TEST_SUITE_P(
        InvQuadFormSuite,
        InvQuadFormFixture,
        testing::Values(
            std::make_tuple(0, 10, 0.1),
            std::make_tuple(321, 10, 0.5),
            std::make_tuple(3214, 10, 0.9),
            std::make_tuple(0, 100, 0.1),
            std::make_tuple(321, 100, 0.5),
            std::make_tuple(3214, 100, 0.9)
            )
    );

}
} // namespace ghostbasil
