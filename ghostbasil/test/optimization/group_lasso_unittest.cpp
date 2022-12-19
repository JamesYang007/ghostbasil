#include <gtest/gtest.h>
#include <ghostbasil/optimization/group_lasso.hpp>
#include <tools/data_util.hpp>
#include <tools/macros.hpp>

namespace ghostbasil {
namespace group_lasso {
namespace {

struct GroupLassoFixture
    : ::testing::Test
{
    const double tol = 1e-8;
    const double thr = 1e-24;
    const size_t max_cds = 10000;
    const double newton_tol = 1e-10;
    const size_t newton_max_iters = 10000;

    template <class GenerateFType>
    void run(GenerateFType generate_dataset)
    {
        // Generate dataset
        auto dataset = generate_dataset();
        auto&& A = std::get<0>(dataset);
        auto&& r = std::get<1>(dataset);
        auto&& groups = std::get<2>(dataset);
        auto&& s = std::get<3>(dataset);
        auto&& strong_set = std::get<4>(dataset);
        auto&& lmdas = std::get<5>(dataset);
        auto&& expected_betas = std::get<6>(dataset);
        auto&& expected_objs = std::get<7>(dataset);
        
        const auto n_groups = groups.size() - 1;
        util::vec_type<int> group_sizes =
            groups.tail(n_groups) - groups.head(n_groups);
            
        std::vector<int> strong_g1;
        std::vector<int> strong_g2;
        strong_g1.reserve(strong_set.size());
        strong_g2.reserve(strong_set.size());
        for (size_t i = 0; i < strong_set.size(); ++i) {
            const auto group = strong_set[i];
            if (group_sizes[group] == 1) {
                strong_g1.push_back(i);
            } else {
                strong_g2.push_back(i);
            }
        }

        std::vector<int> strong_begins(strong_set.size());
        int strong_values_size = 0;
        for (size_t i = 0; i < strong_begins.size(); ++i) {
            strong_begins[i] = strong_values_size;
            const auto group = strong_set[i];
            const auto group_size = group_sizes[group];
            strong_values_size += group_size;
        }

        std::vector<double> strong_grad(strong_values_size);
        for (size_t i = 0; i < strong_begins.size(); ++i) {
            const auto begin = strong_begins[i];
            const auto group = strong_set[i];
            const auto group_size = group_sizes[group];
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
            const auto group_size = group_sizes[group];
            Eigen::Map<util::vec_type<double>> sad_map(
                strong_A_diag.data(), strong_A_diag.size()
            );
            sad_map.segment(begin, group_size) = 
                A.block(groups[group], groups[group], group_size, group_size).diagonal();
        }

        using sp_vec_t = Eigen::SparseVector<double>;
        std::vector<int> active_set;
        std::vector<int> active_g1;
        std::vector<int> active_g2;
        std::vector<int> active_begins;
        std::vector<int> active_order;
        std::vector<int> is_active(strong_set.size(), false);
        std::vector<sp_vec_t> betas(lmdas.size());
        std::vector<double> rsqs(lmdas.size());
        size_t n_cds = 0;
        size_t n_lmdas = 0;
        double rsq = 0;
        
        // run group lasso fitting procedure
        GroupLassoParamPack<
            util::mat_type<double>,
            double,
            int,
            int
        > pack(
            A, groups, group_sizes, s, strong_set, strong_g1, strong_g2, strong_begins,
            strong_A_diag, lmdas, max_cds, thr, newton_tol, newton_max_iters,
            rsq, strong_beta, strong_grad, active_set, 
            active_g1, active_g2, active_begins, active_order,
            is_active, betas, rsqs, n_cds, n_lmdas
        );
        fit(pack);
        test(pack, r, expected_betas, expected_objs);
    }
    
    template <class PackType, class RType, class EBType, class EOType>
    void test(const PackType& pack,
              const RType& r,
              const EBType& expected_betas,
              const EOType& expected_objs)
    {
        const auto& A = *pack.A;
        const auto& groups = pack.groups;
        const auto& group_sizes = pack.group_sizes;
        const auto& lmdas = pack.lmdas;
        const auto& betas = *pack.betas;
        const auto s = pack.s;
        const auto n_cds = pack.n_cds;
        const auto n_lmdas = pack.n_lmdas;

        EXPECT_LE(n_cds, max_cds);

        EXPECT_EQ(betas.size(), lmdas.size());
        EXPECT_EQ(expected_betas.cols(), lmdas.size());
        EXPECT_EQ(expected_objs.size(), lmdas.size());
        EXPECT_LE(0, n_lmdas);
        EXPECT_LE(n_lmdas, lmdas.size());
        
        for (size_t i = 0; i < n_lmdas; ++i) {
            const auto& actual = betas[i];
            auto expected = expected_betas.col(i);

            const auto obj = objective(A, r, groups, group_sizes, s, lmdas[i], actual);
            EXPECT_NEAR(expected_objs[i], obj, tol);
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
    run(GENERATE_DATASET(1));
}

TEST_F(GroupLassoFixture, group_lasso_n_le_p_partial)
{
    run(GENERATE_DATASET(2));
}

TEST_F(GroupLassoFixture, group_lasso_n_ge_p_partial)
{
    run(GENERATE_DATASET(3));
}

} // namespace
} // namespace group_lasso
} // namespace ghostbasil

#undef GENERATE_DATASET