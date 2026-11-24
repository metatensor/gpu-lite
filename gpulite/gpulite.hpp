// gpulite: a lightweight C++ library for dynamic CUDA runtime compilation
#ifndef GPULITE_HPP
#define GPULITE_HPP

#if defined(_MSC_VER)
    // MSVC historically reports __cplusplus wrong unless /Zc:__cplusplus is enabled,
    // so prefer _MSVC_LANG there.
    #if !defined(_MSVC_LANG) || _MSVC_LANG < 201703L
        #error "This project requires C++17 or newer (/std:c++17)."
    #endif
#elif __cplusplus < 201703L
    #error "This project requires C++17 or newer (-std=c++17)."
#endif


#include "./dlopen.hpp"         // IWYU pragma: export
#include "./cuda_api.hpp"       // IWYU pragma: export
#include "./cuda_lib.hpp"       // IWYU pragma: export
#include "./kernel.hpp"         // IWYU pragma: export

#endif // GPULITE_HPP
