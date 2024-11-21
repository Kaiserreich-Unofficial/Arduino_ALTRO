#pragma once

#include <chrono>

#include <altro/solver/typedefs.hpp>

namespace altro {

struct AltroStats {
  using millisd = std::chrono::duration<double, std::milli>;  // milliseconds in double

  SolveStatus status;
  millisd solve_time;
  int iterations;
  int outer_iterations;
  double objective_value;
  double stationarity;
  double primal_feasibility;
  double complimentarity;
};

}  // namespace altro
