#pragma once 
#include <stdexcept>
#include <string>

namespace ghostbasil {

class max_cds_error : std::exception
{
    int lmda_idx_;

public:
    max_cds_error(int lmda_idx)
        : lmda_idx_(lmda_idx)
    {}

    int error_code() const { return -lmda_idx_-1; }
};

class max_basil_iters_error : std::exception
{
    std::string msg_;
public:
    max_basil_iters_error(int n_iters): 
        msg_{"Basil max iterations " + std::to_string(n_iters) + " reached."}
    {}

    const char* what() const noexcept override {
        return msg_.data();
    }
};

} // namespace ghostbasil
