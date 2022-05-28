#pragma once
#include <Eigen/Dense>
#include <string>

namespace ghostbasil {
namespace tools {

Eigen::MatrixXd load_csv_direct(const std::string& path);
Eigen::MatrixXd load_csv(const std::string& path);
Eigen::MatrixXd load_data_path_csv(const std::string& path);

/*
 * Generate data and expected output for optimizing:
 * \[   
 *      (1-s)/2 \beta^\top A \beta - \beta^\top r + s/2 ||\beta||_2^2 + \lambda ||\beta||_1
 * \]
 */
inline auto generate_dataset(const std::string& suffix)
{
    Eigen::MatrixXd A = load_data_path_csv("A_" + suffix + ".csv");
    assert((A.size() > 0) && (A.rows() == A.cols()));
    Eigen::VectorXd r = load_data_path_csv("r_" + suffix + ".csv");
    assert(r.size() == A.cols());
    Eigen::VectorXd s = load_data_path_csv("s_" + suffix + ".csv");
    Eigen::VectorXi strong_set = load_data_path_csv("strong_set_" + suffix + ".csv").template cast<int>();
    assert((strong_set.minCoeff() >= 0) && (strong_set.maxCoeff() < r.size()));
    Eigen::VectorXd expected_lmdas = load_data_path_csv("lmda_" + suffix + ".csv");
    Eigen::MatrixXd expected_betas = load_data_path_csv("beta_" + suffix + ".csv");
    Eigen::VectorXd expected_objs = load_data_path_csv("obj_" + suffix + ".csv");
    assert(expected_lmdas.size() == expected_betas.cols());
    assert(expected_lmdas.size() == expected_objs.size());
    return std::make_tuple(A, r, s[0], strong_set, expected_lmdas, expected_betas, expected_objs);
}

} // namespace tools    
} // namespace ghostbasil
