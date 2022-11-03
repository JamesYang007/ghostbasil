#pragma once
#include <numeric>
#include <ghostbasil/util/exceptions.hpp>
#include <ghostbasil/util/functional.hpp>
#include <ghostbasil/util/macros.hpp>
#include <ghostbasil/util/types.hpp>
#include <ghostbasil/util/functor_iterator.hpp>
#include <ghostbasil/util/eigen/map_sparsevector.hpp>
#include <ghostbasil/matrix/forward_decl.hpp>

namespace ghostbasil {

/*
 * Solves the solution for the equation (w.r.t. x):
 *      x = (C + lmda / ||x|| * I)^{-1} y
 */
template <class CType, class YType, class XType>
GHOSTBASIL_STRONG_INLINE 
void solve_sub_coeffs(
    const CType& C,
    const YType& y,
    double lmda,
    double step_size,
    XType& x,
    size_t& iters,
    size_t max_iters=100,
    double tol=1e-8
)
{
    Eigen::VectorXd w(x.size());
    Eigen::VectorXd curr(x.size());
    Eigen::VectorXd prev = x;

    iters = 1;
    for (; iters <= max_iters; ++iters) {
        w = prev + step_size * (y - C * prev);
        const auto factor = 1.0 - step_size * lmda / w.norm();
        curr = std::max(factor, 0) * w;
        if ((curr - prev).norm() < tol) break;
        curr.swap(prev);
    }
    x = curr;
    iters = std::min(iters, max_iters);
}

} // namespace ghostbasil