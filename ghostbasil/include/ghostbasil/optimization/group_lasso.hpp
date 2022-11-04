#pragma once
#include <array>
#include <numeric>
#include <ghostbasil/util/exceptions.hpp>
#include <ghostbasil/util/functional.hpp>
#include <ghostbasil/util/macros.hpp>
#include <ghostbasil/util/types.hpp>
#include <ghostbasil/util/functor_iterator.hpp>
#include <ghostbasil/util/eigen/map_sparsevector.hpp>
#include <ghostbasil/matrix/forward_decl.hpp>

namespace ghostbasil {
    
template <class T>
GHOSTBASIL_STRONG_INLINE
T safe_sqrt(T x, T tol=-1e-10)
{
    x = (x >= tol) ? std::max(x, 0.0) : x;
    return std::sqrt(x);
}
    
/*
 * Solution to quartic equation:
 *  ax^4 + bx^3 + cx^2 + dx^1 + e = 0
 * where a != 0.
 * Returns the most positive solution.
 * If a solution does not exist, it is undefined behavior.
 */
template <class T>
GHOSTBASIL_STRONG_INLINE
void solve_quartic(
    T a,
    T b,
    T c,
    T d,
    T e,
    T* sol
)
{
    // get depressed quartic equation
    const T b_a = b / a;
    const T b_a_2 = b_a * b_a;
    const T b_a_3 = b_a_2 * b_a;
    const T b_a_4 = b_a_3 * b_a;
    const T c_a = c / a;
    const T d_a = d / a;
    const T e_a = e / a;
    T alpha = (-3.0 / 8.0) * b_a_2 + c_a;
    T beta = (1.0 / 8.0) * b_a_3 - 0.5 * b_a * c_a + d_a;
    T gamma = (-3.0 / 256.0) * b_a_4 + (1.0 / 16.0) * c_a * b_a_2 - 0.25 * b_a * d_a + e_a;
    
    // biquadratic form
    if (std::abs(beta) <= 1e-10) {
        const T discr = alpha * alpha - 4 * gamma;
        const T discr_sqrt = safe_sqrt(discr);
        const T x_p = 0.5 * (-alpha + discr_sqrt);
        const T x_p_sqrt = safe_sqrt(x_p);
        const T x_m = x_p - discr_sqrt;
        const T x_m_sqrt = safe_sqrt(x_m);
        const T correction = - 0.25 * b_a;
        sol[0] = -x_m_sqrt + correction;
        sol[1] = x_m_sqrt + correction;
        sol[2] = -x_p_sqrt + correction;
        sol[3] = x_p_sqrt + correction;
        return;
    }
    
    // solve: x^4 + alpha x^2 + beta x^1 + gamma = 0
    const T p_1 = (-1.0 / 12.0) * alpha * alpha;
    const T p = p_1 - gamma;
    const T q = (1.0 / 9.0) * p_1 * alpha + (1.0 / 3.0) * alpha * gamma - (1.0 / 8.0) * beta * beta;
    const T w_3 = -0.5 * q + safe_sqrt(q * q * 0.25 + p * p * p * (1.0 / 27.0));
    const T w = std::cbrt(w_3); 
    const T y = alpha * (1.0 / 6.0) + w - (1.0 / 3.0) * p / w;
    
    const T two_y_ma = safe_sqrt(2 * y - alpha);
    const T two_y_pa = 2 * y + alpha;
    const T two_beta = 2 * beta;
    const T correction = - 0.25 * b_a;
    sol[0] = 0.5 * (
        -two_y_ma - safe_sqrt(-two_y_pa + two_beta / two_y_ma)
    ) + correction;
    sol[1] = 0.5 * (
        -two_y_ma + safe_sqrt(-two_y_pa + two_beta / two_y_ma)
    ) + correction;
    sol[2] = 0.5 * (
        two_y_ma - safe_sqrt(-two_y_pa - two_beta / two_y_ma)
    ) + correction;
    sol[3] = 0.5 * (
        two_y_ma + safe_sqrt(-two_y_pa - two_beta / two_y_ma)
    ) + correction;
}

/*
 * Solves the solution for the equation (w.r.t. x >= 0):
 *      x * (a + b / sqrt(x^2 + c)) = d
 *  - a must be positive
 *  - b must be positive
 *  - c must be non-negative
 *  - d must be positive
 * Returns the (most positive) solution to the above equation.
 * If solution does not exist, it is undefined behavior.
 */
template <class T>
GHOSTBASIL_STRONG_INLINE
T solve_sub_coord_desc(
    T a, 
    T b, 
    T c, 
    T d    
)
{
    T d_sq = d * d;
    T a1 = a * a;
    T b1 = -2 * a * d;
    T c1 = a1 * c + d_sq - b * b;
    T d1 = b1 * c;
    T e1 = c * d_sq;
    std::array<T, 4> x;
    solve_quartic(a1, b1, c1, d1, e1, x.data());
    for (auto v : x) {
        if (0 <= v && v <= d / a) return v;
    }
    return std::numeric_limits<T>::quiet_NaN();
}
    

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
    const auto p = x.size();
    Eigen::VectorXd buffer(p);
    Eigen::VectorXd w(p);
    Eigen::VectorXd curr = x; 
    Eigen::VectorXd prev = x;
    Eigen::VectorXd prev2(p);

    const auto nu_lmda = step_size * lmda;

    iters = 0;
    for (; iters < max_iters; ++iters) {
        if (iters) {
            // convergence measurement is based on
            // standardizing x.
            w = curr - prev; // use w as buffer
            buffer = C * w;
            const auto convg_measure = w.dot(buffer) / p;
            if (convg_measure < tol) break;
        }
        // nesterov acceleration 
        const auto m = (static_cast<double>(iters) - 1) / (iters + 2);
        prev2 = curr + m * (curr - prev); // use as buffer
        
        // proximal gradient descent
        buffer = C * prev2;
        w = prev2 + step_size * (y - buffer);
        const auto w_norm = w.norm();
        const auto factor = (w_norm <= nu_lmda) ? 0.0 : (1.0 - nu_lmda / w_norm);
        prev.swap(prev2);
        curr.swap(prev);
        curr = factor * w;
    }
    x = curr;
}

} // namespace ghostbasil