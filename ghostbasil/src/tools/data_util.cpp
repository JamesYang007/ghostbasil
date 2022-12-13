#include <vector>
#include <fstream>
#include <iostream>
#include "tools/cpp/runfiles/runfiles.h" // must be quotes!
#include <tools/data_util.hpp>

namespace ghostbasil {
namespace tools {

static constexpr const char* DATA_PATH = "ghostbasil/reference/data/";

Eigen::MatrixXd load_csv_direct(const std::string& path) 
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
Eigen::MatrixXd load_csv(const std::string& path)
{
    using bazel::tools::cpp::runfiles::Runfiles;
    std::string error;
    std::unique_ptr<Runfiles> runfiles(Runfiles::CreateForTest(&error));

    // Important:
    //   If this is a test, use Runfiles::CreateForTest(&error).
    //   Otherwise, if you don't have the value for argv[0] for whatever
    //   reason, then use Runfiles::Create(&error).
    std::string new_path;

    // if runfiles has an error, it is because the user is not
    // calling the binary under "bazel run" or "bazel test" call.
    // In this case, we will look relative to current working directory.
    if (runfiles == nullptr) {
        std::cerr << error << std::endl;
        new_path = path;
    } else {
        new_path = runfiles->Rlocation("__main__/" + path);
    }

    return load_csv_direct(new_path);
}

// Loads csv in bazel_path/DATA_PATH/path.
Eigen::MatrixXd load_data_path_csv(const std::string& path)
{
    return load_csv(DATA_PATH + path); 
}

} // namespace tools
} // namespace ghostbasil
