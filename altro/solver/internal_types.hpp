#pragma once

#include <ArduinoEigenDense.h>
#include <altro\solver\typedefs.hpp>

namespace altro {

using Vector = Eigen::Vector<a_float, Eigen::Dynamic>;
using Matrix = Eigen::Vector<a_float, Eigen::Dynamic>;

}  // namespace altro
