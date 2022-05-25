#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <ghostbasil/macros.hpp>
#include <testutil/data_util.hpp>

#ifndef GENERATE_DATASET_F
#define GENERATE_DATASET_F(n)\
    inline auto generate_dataset_##n() \
    { \
        Eigen::VectorXd s = load_csv(DATA_PATH + std::string("s_basil_" STRINGIFY(n) ".csv")); \
        Eigen::VectorXd expected_lmda = load_csv(DATA_PATH + std::string("lmda_basil_" STRINGIFY(n) ".csv"));\
        Eigen::MatrixXd A = load_csv(DATA_PATH + std::string("A_basil_" STRINGIFY(n) ".csv"));\
        Eigen::VectorXd r = load_csv(DATA_PATH + std::string("r_basil_" STRINGIFY(n) ".csv"));\
        Eigen::MatrixXd expected_betas = load_csv(DATA_PATH + std::string("beta_basil_" STRINGIFY(n) ".csv"));\
        Eigen::VectorXd expected_objs = load_csv(DATA_PATH + std::string("obj_basil_" STRINGIFY(n) ".csv"));\
        return std::make_tuple(A, r, s[0], expected_lmda, expected_betas, expected_objs);\
    }
#endif

namespace ghostbasil {
namespace fit_basil_util {

/*
 * Generate data and expected output for optimizing:
 * \[   
 *      (1-s)/2 \beta^\top A \beta - \beta^\top r + s/2 ||\beta||_2^2 + \lambda ||\beta||_1
 * \]
 */

GENERATE_DATASET_F(1); 
GENERATE_DATASET_F(2);
GENERATE_DATASET_F(3);

auto make_basil_output()
{
    return std::make_tuple(
            std::vector<Eigen::SparseMatrix<double>>(),
            std::vector<Eigen::VectorXd>());
}

} // namespace fit_basil_util
} // namespace ghostbasil

#undef GENERATE_DATASET_F
