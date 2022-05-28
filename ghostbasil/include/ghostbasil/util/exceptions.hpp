#pragma once 
#include <stdexcept>
#include <string>

namespace ghostbasil {
namespace util {

class ghostbasil_error: public std::exception {};

class max_cds_error : public ghostbasil_error
{
    std::string msg_;

public:
    max_cds_error(int lmda_idx)
        : msg_{"Basil max coordinate descents reached at lambda index: " + std::to_string(lmda_idx) + "."}
    {}

    const char* what() const noexcept override {
        return msg_.data();
    }
};

class max_basil_strong_set : public ghostbasil_error
{
    std::string msg_;
public:
    max_basil_strong_set(): 
        msg_{"Basil maximum strong set size reached."}
    {}

    const char* what() const noexcept override {
        return msg_.data();
    }
};

class lasso_finished_early_error : public ghostbasil_error
{
    std::string msg_;
public:
    lasso_finished_early_error(): 
        msg_{"Lasso fitting on strong set finished early in the lambda sequence due to minimal change in R^2."}
    {}

    const char* what() const noexcept override {
        return msg_.data();
    }
};

} // namespace util
} // namespace ghostbasil
