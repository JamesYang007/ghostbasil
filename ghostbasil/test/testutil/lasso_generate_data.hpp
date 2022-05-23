#pragma once
#include <Eigen/Dense>
#include <vector>
#include <ghostbasil/macros.hpp>
#include <testutil/data_util.hpp>

#ifndef GENERATE_DATASET_F
#define GENERATE_DATASET_F(n)\
    inline auto generate_dataset_##n() \
    { \
        Eigen::VectorXd s = load_csv(DATA_PATH + std::string("s_lasso_" STRINGIFY(n) ".csv")); \
        Eigen::VectorXd expected_lmda = load_csv(DATA_PATH + std::string("lmda_lasso_" STRINGIFY(n) ".csv"));\
        Eigen::MatrixXd A = load_csv(DATA_PATH + std::string("A_lasso_" STRINGIFY(n) ".csv"));\
        Eigen::VectorXd r = load_csv(DATA_PATH + std::string("r_lasso_" STRINGIFY(n) ".csv"));\
        Eigen::MatrixXd expected_betas = load_csv(DATA_PATH + std::string("beta_lasso_" STRINGIFY(n) ".csv"));\
        Eigen::VectorXd expected_objs = load_csv(DATA_PATH + std::string("obj_lasso_" STRINGIFY(n) ".csv"));\
        Eigen::VectorXi strong_set = load_csv(DATA_PATH + std::string("strong_set_lasso_" STRINGIFY(n) ".csv")).template cast<int>();\
        return std::make_tuple(A, r, s[0], strong_set, expected_lmda, expected_betas, expected_objs);\
    }
#endif

namespace ghostbasil {

/*
 * Generate data and expected output for optimizing:
 * \[   
 *      (1-s)/2 \beta^\top A \beta - \beta^\top r + s/2 ||\beta||_2^2 + \lambda ||\beta||_1
 * \]
 * with the additional constraint that $\beta_k = 0$ for all $k$ not in the strong set.
 */

GENERATE_DATASET_F(1);
GENERATE_DATASET_F(2);
GENERATE_DATASET_F(3);
GENERATE_DATASET_F(4);
GENERATE_DATASET_F(5);

} // namespace ghostbasil

#undef GENERATE_DATASET_F
