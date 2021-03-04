// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include <cstdlib>

namespace pose {

/** Use some template magic to determine an array's size. */
template <typename T, std::size_t N>
constexpr std::size_t array_size(const T (&)[N]) noexcept
{
    return N;
}

} // namespace pose
