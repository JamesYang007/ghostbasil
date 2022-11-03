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
    double tol=1e-10
)
{
    Eigen::VectorXd buffer(x.size());
    Eigen::VectorXd w(x.size());
    Eigen::VectorXd curr = x; 
    Eigen::VectorXd prev(curr.size());

    const auto nu_lmda = step_size * lmda;

    iters = 0;
    for (; iters < max_iters; ++iters) {
        if (iters) {
            // convergence measurement is based on
            // standardizing x.
            w = curr - prev; // use w as buffer
            buffer = C * w;
            const auto convg_measure = w.dot(buffer);
            if (convg_measure < tol) break;
        }
        buffer = C * curr;
        w = curr + step_size * (y - buffer);
        const auto w_norm = w.norm();
        const auto factor = (w_norm <= nu_lmda) ? 0.0 : (1.0 - nu_lmda / w_norm);
        curr.swap(prev);
        curr = factor * w;
    }
    x = curr;
}

} // namespace ghostbasil