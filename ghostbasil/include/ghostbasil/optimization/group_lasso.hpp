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
    if (c == 0) {
        const auto sol = (d-b) / a;
        return (sol >= 0) ? sol : std::numeric_limits<T>::quiet_NaN();
    }
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

template <class CType, class GradType, class BetaType>
GHOSTBASIL_STRONG_INLINE
void sub_coord_desc(
    const CType& C,
    GradType& grad,
    BetaType& beta,
    double& beta_normsq,
    double& convg_measure,
    double lmda
)
{
    convg_measure = 0;
    for (size_t i = 0; i < beta.size(); ++i) {
        const auto bi = beta[i];
        const auto gi = grad[i];
        const auto Cii = C(i,i);
        
        // get new coefficient
        const auto xi_resid = gi + Cii * bi;
        const auto xi_resid_abs = std::abs(xi_resid);
        auto& bi_ref = beta[i];
        beta_normsq -= bi * bi;
        const auto safe_beta_normsq = std::max(beta_normsq, 0.0);
        bi_ref = (xi_resid_abs <= lmda) ? 0.0 :
            std::copysign(1.0, xi_resid) *
            solve_sub_coord_desc(
                Cii, lmda, safe_beta_normsq, xi_resid_abs
            );
        beta_normsq += bi_ref * bi_ref;
        
        // TODO: debug
        auto h = std::abs(bi_ref);
        auto debug = h * (Cii + lmda / std::sqrt(h*h + safe_beta_normsq)) - xi_resid_abs;
        if (std::abs(debug) >= 1e-10) {
            Rcpp::Rcout << debug << std::endl;
            Rcpp::Rcout << Cii << std::endl;
            Rcpp::Rcout << lmda << std::endl;
            Rcpp::Rcout << safe_beta_normsq << std::endl;
            Rcpp::Rcout << xi_resid_abs << std::endl;
        }
        
        if (bi_ref == bi) continue;
        
        const auto del = bi_ref - bi;
        
        // update measure of convergence
        convg_measure = std::max(convg_measure, Cii * del * del);
        
        // update gradient
        grad -= del * C.col(i);
    }
}

/*
 * Solves the solution for the equation (w.r.t. x):
 *      minimize_x 1/2 [x^T C x - 2 x^T b] + lmda ||x||_2
 */
template <class CType, class BType, class YType>
GHOSTBASIL_STRONG_INLINE 
void solve_sub_coeffs(
    const CType& C,
    const BType& b,
    double lmda,
    double step_size,
    YType& y_curr,
    size_t& iters,
    size_t max_iters=100,
    double tol=1e-10
)
{
    const auto p = y_curr.size();
    Eigen::VectorXd y_prev(p);
    Eigen::VectorXd Cy_curr = C * y_curr;
    Eigen::VectorXd Cy_prev(p);
    Eigen::VectorXd w(p);
    Eigen::VectorXd x_curr = y_curr; 
    Eigen::VectorXd x_prev(p);
    Eigen::VectorXd x_diff(p);

    const auto nu_lmda = step_size * lmda;

    iters = 0;
    double accel_size = 1.0;
    for (; iters < max_iters; ++iters) {
        if (iters) {
            // adaptive restart
            if ((y_prev - x_curr).dot(x_diff) > 0) { 
                y_curr = x_curr;

                // keep invariance
                Cy_curr.swap(Cy_prev);
                Cy_curr = C * y_curr;    

                accel_size = 1.0;
            } 
            else {
                // keep invariance
                Cy_curr.swap(Cy_prev);
                Cy_curr = C * y_curr;    

                // convergence measurement is based on
                // standardizing solution vector.
                w = Cy_curr - Cy_prev; // use w as extra buffer
                const auto convg_measure = w.dot(y_curr - y_prev) / p;
                if (convg_measure < tol) break;
            }
        }

        // swap curr and prev to populate new curr
        x_curr.swap(x_prev);
        
        // proximal gradient descent
        w = y_curr + step_size * (b - Cy_curr);
        const auto w_norm = w.norm();
        const auto factor = (w_norm <= nu_lmda) ? 0.0 : (1.0 - nu_lmda / w_norm);
        x_curr = factor * w;

        // nesterov acceleration 
        const auto numer = accel_size - 1.0;
        accel_size = 0.5 * (1.0 + std::sqrt(1.0 + 4 * accel_size * accel_size));
        const auto m = numer / accel_size;
        y_curr.swap(y_prev);
        x_diff = x_curr - x_prev;
        y_curr = x_curr + m * x_diff; 
    }
}

template <class CType, class BType, class BetaType>
GHOSTBASIL_STRONG_INLINE
void solve_sub_coeffs_mix(
    const CType& C,
    const BType& b,
    BetaType& beta,
    double lmda,
    size_t max_cd_iters,
    double cd_tol,
    double step_size,
    size_t& iters,
    size_t max_iters,
    double tol
)
{
    // first try coord-desc to get a good guess
    Eigen::VectorXd grad = b - C * beta;
    auto beta_normsq = beta.squaredNorm();
    double convg_measure = 0.0; 
    for (size_t i = 0; i < max_cd_iters; ++i) {
        sub_coord_desc(
            C, grad, beta, beta_normsq, convg_measure, lmda
        );
        if (convg_measure < cd_tol) break;
    }
    
    // next finish off with guaranteed method
    solve_sub_coeffs(C, b, lmda, step_size, beta, iters, max_iters, tol);
}
    
} // namespace ghostbasil