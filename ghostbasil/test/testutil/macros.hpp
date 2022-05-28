#ifndef STRINGIFY
#ifndef STRINGIFY_IMPL
#define STRINGIFY_IMPL(x) #x
#endif
#define STRINGIFY(x) STRINGIFY_IMPL(x)
#endif

