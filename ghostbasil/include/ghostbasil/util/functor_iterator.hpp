#pragma once
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <ghostbasil/util/macros.hpp>

namespace ghostbasil {
namespace util {

// forward declaration
template <class IntType, class F>
class functor_iterator;

template <class IntType, class F>
inline constexpr bool 
operator==(const functor_iterator<IntType, F>& it1,
           const functor_iterator<IntType, F>& it2)
{ 
    assert(&it1.f_ == &it2.f_);
    return (it1.curr_ == it2.curr_); 
}

template <class IntType, class F>
inline constexpr bool 
operator!=(const functor_iterator<IntType, F>& it1,
           const functor_iterator<IntType, F>& it2)
{ 
    assert(&it1.f_ == &it2.f_);
    return (it1.curr_ != it2.curr_); 
}

template <class IntType, class F>
class functor_iterator
{
    using int_t = IntType;
    using f_t = F;

public:
    using difference_type = int32_t;
#if __cplusplus >= 201703L
    using value_type = std::invoke_result_t<F, int_t>;
#elif __cplusplus >= 201103L 
    using value_type = typename std::result_of<F(int_t)>::type;
#endif
    using pointer = value_type*;
    using reference = value_type&;
    using iterator_category = std::forward_iterator_tag;

private:
    f_t& f_;
    int_t curr_;

public:

    functor_iterator(int_t begin, f_t& f)
        : f_(f), curr_(begin)
    {}

    GHOSTBASIL_STRONG_INLINE functor_iterator& operator++() { ++curr_; return *this; }

    // Weirdly, returning reference destroys speed in lasso coordinate descent.
    // Make sure to return by value!
    GHOSTBASIL_STRONG_INLINE auto operator*() { return f_(curr_); }

    friend constexpr bool operator==<>(const functor_iterator&,
                                       const functor_iterator&);
    friend constexpr bool operator!=<>(const functor_iterator&,
                                       const functor_iterator&);
};

template <class IntType, class F>
auto make_functor_iterator(IntType i, F& f)
{
    return functor_iterator<IntType, F>(i, f);
}

} // namespace util
} // namespace ghosbasil
