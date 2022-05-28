#pragma once
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <vector>
#include <thread>

namespace ghostbasil {
namespace basil_util {

template <class F>
inline auto make_basil_input(
        F generate_dataset,
        size_t max_strong_size,
        bool do_user)
{
    auto dataset = generate_dataset();
    auto& A = std::get<0>(dataset);
    auto& r = std::get<1>(dataset);
    auto& s = std::get<2>(dataset);
    auto& expected_lmdas = std::get<4>(dataset);
    auto& expected_betas = std::get<5>(dataset);
    auto& expected_objs = std::get<6>(dataset);

    std::vector<double> user_lmdas;
    size_t max_strong_size_local = max_strong_size;
    if (do_user) {
        user_lmdas.resize(expected_lmdas.size());
        Eigen::Map<Eigen::VectorXd>(user_lmdas.data(), user_lmdas.size())
            = expected_lmdas;
        max_strong_size_local = r.size(); // force full strong set
    }
    
    std::vector<Eigen::SparseVector<double>> betas;
    std::vector<double> lmdas;
    std::vector<double> rsqs;
    return std::make_tuple(
            A, r, s, user_lmdas, max_strong_size_local,
            betas, lmdas, rsqs,
            expected_betas, expected_lmdas, expected_objs);
}

}
} // namespace ghostbasil
