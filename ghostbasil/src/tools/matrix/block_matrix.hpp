#pragma once
#include <Eigen/Core>

namespace ghostbasil {
namespace tools {

struct Assignment
{
    template <class XType, class YType>
    auto&& operator()(XType&& x, const YType& y)
    {
        return x = y;
    }
};

struct BlockMatrixUtil
{
    using value_t = double;
    using mat_t = Eigen::MatrixXd;

    template <class MatListType, class AssignType=Assignment>
    auto make_dense(
            const MatListType& ml,
            AssignType assign = AssignType())
    {
        size_t p = 0;
        for (size_t i = 0; i < ml.size(); ++i) {
            const auto& B = ml[i];
            assert(B.rows() == B.cols());
            p += B.rows();
        }

        mat_t dense(p, p);
        dense.setZero();

        size_t pi_cum = 0;
        for (size_t i = 0; i < ml.size(); ++i) {
            const auto& B = ml[i];
            size_t pi = B.rows();
            auto dense_i = dense.block(pi_cum, pi_cum, pi, pi);
            assign(dense_i, B);
            pi_cum += pi;
        }

        return dense;
    }
};

} // namespace tools
} // namespace ghostbasil
