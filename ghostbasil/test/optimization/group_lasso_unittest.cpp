#include <gtest/gtest.h>
#include <ghostbasil/optimization/group_lasso.hpp>
#include <tools/data_util.hpp>
#include <tools/macros.hpp>

namespace ghostbasil {
namespace {

struct GroupLassoFixture
    : ::testing::Test
{
    const double tol = 1e-8;
    const double thr = 1e-24;
    const size_t max_cds = 10000;
    const double newton_tol = 1e-10;
    const size_t newton_max_iters = 10000;

    template <class F>
    auto make_input(F generate_dataset)
    {
        auto dataset = generate_dataset();
        auto&& A = std::get<0>(dataset);
        auto&& r = std::get<1>(dataset);
        auto&& groups = std::get<2>(dataset);
        auto&& s = std::get<3>(dataset);
        auto&& strong_set = std::get<4>(dataset);
        auto&& lmdas = std::get<5>(dataset);
        auto&& expected_betas = std::get<6>(dataset);
        auto&& expected_objs = std::get<7>(dataset);

        std::vector<int> strong_begins(strong_set.size());
        int strong_values_size = 0;
        for (size_t i = 0; i < strong_begins.size(); ++i) {
            strong_begins[i] = strong_values_size;
            const auto group = strong_set[i];
            const auto group_size = groups[group+1] - groups[group];
            strong_values_size += group_size;
        }

        std::vector<double> strong_grad(strong_values_size);
        for (size_t i = 0; i < strong_begins.size(); ++i) {
            const auto begin = strong_begins[i];
            const auto group = strong_set[i];
            const auto group_size = groups[group+1] - groups[group];
            Eigen::Map<util::vec_type<double>> sg_map(
                strong_grad.data(), strong_grad.size()
            );
            sg_map.segment(begin, group_size) = 
                r.segment(groups[group], group_size);
        }

        util::vec_type<double> strong_beta(strong_values_size);
        strong_beta.setZero();
        
        util::vec_type<double> strong_A_diag(strong_values_size);
        for (size_t i = 0; i < strong_begins.size(); ++i) {
            const auto begin = strong_begins[i];
            const auto group = strong_set[i];
            const auto group_size = groups[group+1] - groups[group];
            Eigen::Map<util::vec_type<double>> sad_map(
                strong_A_diag.data(), strong_A_diag.size()
            );
            sad_map.segment(begin, group_size) = 
                A.block(groups[group], groups[group], group_size, group_size).diagonal();
        }

        using sp_vec_t = Eigen::SparseVector<double>;
        std::vector<int> active_set;
        std::vector<int> active_begins;
        std::vector<int> active_order;
        std::vector<bool> is_active(strong_set.size(), false);
        std::vector<sp_vec_t> betas(lmdas.size());
        std::vector<double> rsqs(lmdas.size());
        size_t n_cds = 0;
        size_t n_lmdas = 0;

        double rsq = 0;

        return std::make_tuple(
                std::move(A), 
                std::move(r), 
                std::move(groups),
                std::move(s),
                std::move(strong_set), 
                std::move(strong_begins),
                std::move(strong_A_diag), 
                std::move(lmdas), 
                rsq, 
                std::move(strong_beta),
                std::move(strong_grad),
                std::move(active_set),
                std::move(active_begins),
                std::move(active_order),
                std::move(is_active),
                std::move(betas),
                std::move(rsqs),
                n_cds, 
                n_lmdas,
                std::move(expected_betas),
                std::move(expected_objs));
    }

    template <class InputType>
    auto run(InputType&& input)
    {
        auto& A = std::get<0>(input);
        auto& groups = std::get<2>(input);
        auto& s = std::get<3>(input);
        auto& strong_set = std::get<4>(input);
        auto& strong_begins = std::get<5>(input);
        auto& strong_A_diag = std::get<6>(input);
        auto& lmdas = std::get<7>(input);
        auto& rsq = std::get<8>(input);
        auto& strong_beta = std::get<9>(input);
        auto& strong_grad = std::get<10>(input);
        auto& active_set = std::get<11>(input);
        auto& active_begins = std::get<12>(input);
        auto& active_order = std::get<13>(input);
        auto& is_active = std::get<14>(input);
        auto& betas = std::get<15>(input);
        auto& rsqs = std::get<16>(input);
        auto& n_cds = std::get<17>(input);
        auto& n_lmdas = std::get<18>(input);
        
        GroupLasso<double> gl(A.cols(), A.cols());
        
        gl.group_lasso(
            A, groups, s, strong_set, strong_begins, 
            strong_A_diag, lmdas, max_cds, thr, newton_tol, newton_max_iters,
            rsq, strong_beta, 
            strong_grad, active_set, active_begins, active_order, 
            is_active, 
            betas, rsqs, n_cds, n_lmdas);

        return std::make_tuple(betas, rsqs, n_cds, n_lmdas);
    }

    template <class GenerateFType>
    void test(GenerateFType generate_dataset)
    {
        auto&& input = make_input(generate_dataset);
        auto&& output = run(input);

        auto&& A = std::get<0>(input);
        auto&& r = std::get<1>(input);
        auto&& groups = std::get<2>(input);
        auto&& s = std::get<3>(input);
        auto&& lmdas = std::get<7>(input);
        auto&& expected_betas = std::get<19>(input);
        auto&& expected_objs = std::get<20>(input);

        auto&& betas = std::get<0>(output);
        auto&& n_cds = std::get<2>(output);
        auto&& n_lmdas = std::get<3>(output);

        EXPECT_LE(n_cds, max_cds);

        EXPECT_EQ(betas.size(), lmdas.size());
        EXPECT_EQ(expected_betas.cols(), lmdas.size());
        EXPECT_EQ(expected_objs.size(), lmdas.size());
        EXPECT_LE(0, n_lmdas);
        EXPECT_LE(n_lmdas, lmdas.size());
        
        GroupLasso<double> gl(A.cols(), A.cols());

        for (size_t i = 0; i < n_lmdas; ++i) {
            const auto& actual = betas[i];
            auto expected = expected_betas.col(i);

            EXPECT_NEAR(expected_objs[i], gl.objective(A, r, groups, s, lmdas[i], actual), tol);
            EXPECT_EQ(actual.size(), expected.size());
            for (size_t i = 0; i < expected.size(); ++i) {
                EXPECT_NEAR(expected[i], actual.coeff(i), tol);
            }
        }
    }
};

#ifndef GENERATE_DATASET
#define GENERATE_DATASET(n) \
    []() { \
        return tools::generate_group_dataset("group_lasso_" STRINGIFY(n));\
    }
#endif

TEST_F(GroupLassoFixture, group_lasso_n_ge_p_full)
{
    test(GENERATE_DATASET(1));
}

TEST_F(GroupLassoFixture, group_lasso_n_le_p_partial)
{
    test(GENERATE_DATASET(2));
}

TEST_F(GroupLassoFixture, group_lasso_n_ge_p_partial)
{
    test(GENERATE_DATASET(3));
}

} // namespace
} // namespace ghostbasil

#undef GENERATE_DATASET