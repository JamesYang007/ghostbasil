#pragma once
#include <chrono>

namespace ghostbasil {
namespace tools {

class Stopwatch
{
    std::chrono::time_point<std::chrono::steady_clock> start_;
    double& store_;
public:
    Stopwatch(double& store)
        : start_(std::chrono::steady_clock::now()),
          store_(store)
    {}

    ~Stopwatch()
    {
        auto end = std::chrono::steady_clock::now();
        store_ = std::chrono::duration_cast<
            std::chrono::nanoseconds>(end - start_).count() * 1e-9;
    }
};

} // namespace tools
} // namespace ghostbasil
