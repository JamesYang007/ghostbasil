#pragma once
#include <ghostbasil/util/macros.hpp>

namespace ghostbasil {
    
class LassoBase
{
public:
    /*
     * Checks early stopping based on R^2 values.
     * Returns true (early stopping should occur) if both are true:
     *
     *      delta_u := (R^2_u - R^2_m)/R^2_u
     *      delta_m := (R^2_m - R^2_l)/R^2_m 
     *      delta_u < cond_0_thresh 
     *      AND
     *      (delta_u - delta_m) < cond_1_thresh
     *
     * @param   rsq_l   third to last R^2 value.
     * @param   rsq_m   second to last R^2 value.
     * @param   rsq_u   last R^2 value.
     */
    template <class ValueType>
    GHOSTBASIL_STRONG_INLINE
    static bool check_early_stop_rsq(
            ValueType rsq_l,
            ValueType rsq_m,
            ValueType rsq_u,
            ValueType cond_0_thresh = 1e-5,
            ValueType cond_1_thresh = 1e-5)
    {
        const auto delta_u = (rsq_u-rsq_m);
        const auto delta_m = (rsq_m-rsq_l);
        return ((delta_u < cond_0_thresh*rsq_u) &&
                ((delta_m*rsq_u-delta_u*rsq_m) < cond_1_thresh*rsq_m*rsq_u));
    }
};

} // namespace ghostbasil