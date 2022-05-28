#pragma once
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include "tools/cpp/runfiles/runfiles.h"

namespace ghostbasil {
namespace testutil {

static constexpr const char* DATA_PATH = "ghostbasil/test/testutil/reference/data/";

inline Eigen::MatrixXd load_csv_direct(const std::string& path) 
{
    using namespace Eigen;
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    size_t rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            values.data(), rows, (rows == 0) ? 0 : values.size()/rows);
}

// Loads csv by modifying the path to be bazel's path.
inline auto load_csv(const std::string& path)
{
    using bazel::tools::cpp::runfiles::Runfiles;
    std::string error;
    std::unique_ptr<Runfiles> runfiles(Runfiles::CreateForTest(&error));

    // Important:
    //   If this is a test, use Runfiles::CreateForTest(&error).
    //   Otherwise, if you don't have the value for argv[0] for whatever
    //   reason, then use Runfiles::Create(&error).

    if (runfiles == nullptr) {
        throw std::runtime_error("Runfiles failed.");
    }

    std::string new_path = runfiles->Rlocation("__main__/" + path);
    return load_csv_direct(new_path);
}

// Loads csv in bazel_path/DATA_PATH/path.
inline auto load_data_path_csv(const std::string& path)
{
    return load_csv(DATA_PATH + path); 
}

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

} // namespace testutil
} // namespace ghostbasil
