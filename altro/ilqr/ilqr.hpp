#pragma once

#include <array>
#include <functional>
#include <iostream>
#include <map>
#include <vector>

#include <altro/common/solver_stats.hpp>
#include <altro/common/state_control_sized.hpp>
#include <altro/common/timer.hpp>
#include <altro/common/trajectory.hpp>
#include <altro/eigentypes.hpp>
#include <altro/ilqr/knot_point_function_type.hpp>
#include <altro/problem/problem.hpp>
#include <altro/utils/assert.hpp>

namespace altro
{
  namespace ilqr
  {

    /**
     * @brief 使用迭代LQR求解无约束的轨迹优化问题的类
     */
    template <int n = Eigen::Dynamic, int m = Eigen::Dynamic>
    class iLQR
    {
    public:
      /**
       * @brief 构造函数，通过指定的节点数N初始化iLQR
       * @param N 节点数
       */
      explicit iLQR(int N) : N_(N), knotpoints_() { ResetInternalVariables(); }

      /**
       * @brief 通过问题实例构造iLQR
       * @param prob 轨迹优化问题实例
       */
      explicit iLQR(const problem::Problem &prob)
          : N_(prob.NumSegments()), initial_state_(std::move(prob.GetInitialStatePointer()))
      {
        InitializeFromProblem(prob);
      }

      SolverOptions &GetOptions()
      {
        static SolverOptions options;
        return options;
      }

      const SolverOptions &GetOptions() const
      {
        static SolverOptions options;
        return options;
      }

      // 禁止拷贝构造和赋值
      iLQR(const iLQR &other) = delete;
      iLQR &operator=(const iLQR &other) = delete;

      /**
       * @brief Copy the data from a Problem class into the iLQR solver
       *
       * Capture shared pointers to the cost and dynamics objects for each
       * knot point, storing them in the correspoding KnotPointFunctions object.
       *
       * Assumes both the problem and the solver have the number of knot points.
       *
       * Allows for a subset of the knot points to be copied, since in the future
       * this method might be used to specify compile-time sizes for hybrid /
       * switched dynamics.
       *
       * Appends the knotpoints to those currently in the solver.
       *
       * Captures the initial state from the problem as a shared pointer, so the
       * initial state of the solver is changed by modifying the initial state of
       * the original problem.
       *
       * @tparam n2 Compile-time state dimension. Can be Eigen::Dynamic (-1)
       * @tparam m2 Compile-time control dimension. Can be Eigen::Dynamic (-1)
       * @param prob Trajectory optimization problem
       * @param k_start Starting index (inclusive) for data to copy. 0 <= k_start < N+1
       * @param k_stop Terminal index (exclusive) for data to copy. 0 < k_stop <= N+1
       */

      template <int n2 = n, int m2 = m>
      void CopyFromProblem(const problem::Problem &prob, int k_start, int k_stop)
      {
        ALTRO_ASSERT(0 <= k_start && k_start <= N_,
                     fmt::format("Start index must be in the interval [0,{}]", N_));
        ALTRO_ASSERT(0 <= k_stop && k_stop <= N_ + 1,
                     fmt::format("Start index must be in the interval [0,{}]", N_ + 1));
        ALTRO_ASSERT(prob.IsFullyDefined(), "Expected problem to be fully defined.");
        for (int k = k_start; k < k_stop; ++k)
        {
          if (n != Eigen::Dynamic)
          {
            ALTRO_ASSERT(
                prob.GetDynamics(k)->StateDimension() == n,
                fmt::format("Inconsistent state dimension at knot point {}. Expected {}, got {}", k, n,
                            prob.GetDynamics(k)->StateDimension()));
          }
          if (m != Eigen::Dynamic)
          {
            ALTRO_ASSERT(
                prob.GetDynamics(k)->ControlDimension() == m,
                fmt::format("Inconsistent control dimension at knot point {}. Expected {}, got {}", k,
                            m, prob.GetDynamics(k)->ControlDimension()));
          }
          std::shared_ptr<problem::DiscreteDynamics> model = prob.GetDynamics(k);
          std::shared_ptr<problem::CostFunction> costfun = prob.GetCostFunction(k);
          knotpoints_.emplace_back(std::make_unique<ilqr::KnotPointFunctions<n2, m2>>(model, costfun));
        }
        initial_state_ = prob.GetInitialStatePointer();
        is_initial_state_set = true;
      }

      /**
       * @brief 从问题中初始化iLQR设置
       * @tparam n2 状态维度
       * @tparam m2 控制维度
       * @param prob 轨迹优化问题实例
       */
      template <int n2 = n, int m2 = m>
      void InitializeFromProblem(const problem::Problem &prob)
      {
        ALTRO_ASSERT(prob.NumSegments() == N_,
                     fmt::format("Number of segments in problem {}, should be equal to the number of "
                                 "segments in the solver, {}",
                                 prob.NumSegments(), N_));
        CopyFromProblem<n2, m2>(prob, 0, N_ + 1);
        ResetInternalVariables();
      }

      /***************************** Getters **************************************/

      /**
       * @brief 获取轨迹的指针
       * @return 轨迹指针
       */
      std::shared_ptr<Trajectory<n, m>> GetTrajectory() { return Z_; }

      /**
       * @brief 返回轨迹中的段数
       * @return 段数
       */
      int NumSegments() const { return N_; }

      /**
       * @brief 获取Knot Point Function对象，包含所有节点的数据
       * @param k 节点索引
       * @return KnotPointFunctions类的引用
       */
      KnotPointFunctions<n, m> &GetKnotPointFunction(int k)
      {
        ALTRO_ASSERT((k >= 0) && (k <= N_), "Invalid knot point index.");
        return *(knotpoints_[k]);
      }

      SolverStats &GetStats() { return stats_; }
      SolverStatus GetStatus() const { return status_; }

      /***************************** Setters **************************************/

      /**
       * @brief 存储轨迹的指针
       * @param traj 轨迹指针
       */
      void SetTrajectory(std::shared_ptr<Trajectory<n, m>> traj)
      {
        Z_ = std::move(traj);
        Zbar_ = std::make_unique<Trajectory<n, m>>(*Z_);
        Zbar_->SetZero();
      }

      /**
       * @brief 设置节点索引的划分为可并行执行的任务
       * @param inds 严格递增的节点索引向量
       */
      void SetTaskAssignment(std::vector<int> inds)
      {
        work_inds_ = std::move(inds);
      }

      /***************************** Algorithm **************************************/

      /**
       * @brief 使用iLQR求解轨迹优化问题
       */
      void Solve()
      {
        ALTRO_ASSERT(is_initial_state_set, "Initial state must be set before solving.");
        ALTRO_ASSERT(Z_ != nullptr, "Invalid trajectory pointer. May be uninitialized.");

        ALTRO_ASSERT(Z_->NumSegments() == N_,
                     fmt::format("Initial trajectory must have length {}", N_));

        // Start profiler
        GetOptions().profiler_enable ? stats_.GetTimer()->Activate() : stats_.GetTimer()->Deactivate();

        Stopwatch sw = stats_.GetTimer()->Start("ilqr");

        SolveSetup(); // 重置任何内部变量
        Rollout();    // 使用初始控制前向模拟系统
        stats_.initial_cost = Cost();

        for (int iter = 0; iter < GetOptions().max_iterations_inner; ++iter)
        {
          UpdateExpansions();
          BackwardPass();
          ForwardPass();
          UpdateConvergenceStatistics();

          if (stats_.GetVerbosity() >= LogLevel::kInner)
          {
            stats_.PrintLast();
          }

          if (IsDone())
          {
            break;
          }
        }

        WrapUp();
      }

      /**
       * @brief 计算当前轨迹的成本
       * @return 当前成本
       */
      double Cost()
      {
        ALTRO_ASSERT(Z_ != nullptr, "Invalid trajectory pointer. May be uninitialized.");
        return Cost(*Z_);
      }

      double Cost(const Trajectory<n, m> &Z)
      {
        Stopwatch sw = stats_.GetTimer()->Start("cost");
        CalcIndividualCosts(Z);
        return costs_.sum();
      }

      /**
       * @brief 更新成本和动力学扩展
       */
      void UpdateExpansions()
      {
        Stopwatch sw = stats_.GetTimer()->Start("expansions");
        ALTRO_ASSERT(Z_ != nullptr, "Trajectory pointer must be set before updating the expansions.");

        UpdateExpansionsBlock(0, NumSegments() + 1);
      }

      /**
       * @brief 计算局部最优的线性反馈策略
       */
      void BackwardPass()
      {
        Stopwatch sw = stats_.GetTimer()->Start("backward_pass");

        // Regularization
        Eigen::ComputationInfo info;

        // Terminal Cost-to-go
        knotpoints_[N_]->CalcTerminalCostToGo();
        Eigen::Matrix<double, n, n> *Sxx_prev = &(knotpoints_[N_]->GetCostToGoHessian());
        Eigen::Matrix<double, n, 1> *Sx_prev = &(knotpoints_[N_]->GetCostToGoGradient());

        int max_reg_count = 0;
        deltaV_[0] = 0.0;
        deltaV_[1] = 0.0;

        bool repeat_backwardpass = true;
        while (repeat_backwardpass)
        {
          for (int k = N_ - 1; k >= 0; --k)
          {
            knotpoints_[k]->CalcActionValueExpansion(*Sxx_prev, *Sx_prev);
            knotpoints_[k]->RegularizeActionValue(rho_);
            info = knotpoints_[k]->CalcGains();

            // Handle solve failure
            if (info != Eigen::Success)
            {
              IncreaseRegularization();
              Sxx_prev = &(knotpoints_[N_]->GetCostToGoHessian());
              Sx_prev = &(knotpoints_[N_]->GetCostToGoGradient());

              if (rho_ >= GetOptions().bp_reg_max)
              {
                max_reg_count++;
              }
              if (max_reg_count >= GetOptions().bp_reg_fail_threshold)
              {
                status_ = SolverStatus::kBackwardPassRegularizationFailed;
                repeat_backwardpass = false;
              }
              break;
            }

            knotpoints_[k]->CalcCostToGo();
            knotpoints_[k]->AddCostToGo(&deltaV_);

            Sxx_prev = &(knotpoints_[k]->GetCostToGoHessian());
            Sx_prev = &(knotpoints_[k]->GetCostToGoGradient());

            if (k == 0)
            {
              repeat_backwardpass = false;
            }
          }
        }
        stats_.Log("reg", rho_);
        DecreaseRegularization();
      }

      /**
       * @brief 从初始状态前向模拟动力学
       */
      void Rollout()
      {
        Z_->State(0) = *initial_state_;
        for (int k = 0; k < N_; ++k)
        {
          knotpoints_[k]->Dynamics(Z_->State(k), Z_->Control(k), Z_->GetTime(k), Z_->GetStep(k),
                                   Z_->State(k + 1));
        }
      }

      /**
       * @brief 使用反馈和前馈增益从优化的策略进行闭环前向模拟
       * @param alpha 线搜索参数
       * @return 如果状态和控制边界未被违反则返回true
       */
      bool RolloutClosedLoop(const double alpha)
      {
        Stopwatch sw = stats_.GetTimer()->Start("rollout");

        Zbar_->State(0) = *initial_state_;
        for (int k = 0; k < N_; ++k)
        {
          MatrixNxMd<m, n> &K = GetKnotPointFunction(k).GetFeedbackGain();
          VectorNd<m> &d = GetKnotPointFunction(k).GetFeedforwardGain();

          VectorNd<n> dx = Zbar_->State(k) - Z_->State(k);
          Zbar_->Control(k) = Z_->Control(k) + K * dx + d * alpha;

          GetKnotPointFunction(k).Dynamics(Zbar_->State(k), Zbar_->Control(k), Zbar_->GetTime(k),
                                           Zbar_->GetStep(k), Zbar_->State(k + 1));

          if (GetOptions().check_forwardpass_bounds)
          {
            if (Zbar_->State(k + 1).norm() > GetOptions().state_max)
            {
              status_ = SolverStatus::kStateLimit;
              return false;
            }
            if (Zbar_->Control(k).norm() > GetOptions().control_max)
            {
              status_ = SolverStatus::kControlLimit;
              return false;
            }
          }
        }
        status_ = SolverStatus::kUnsolved;
        return true;
      }

      /**
       * @brief 尝试找到更好的状态-控制轨迹
       */
      void ForwardPass()
      {
        Stopwatch sw = stats_.GetTimer()->Start("forward_pass");
        SolverOptions &opts = GetOptions();

        double J0 = costs_.sum(); // 在UpdateExpansions中计算的

        double alpha = 1.0;
        double z = -1.0;
        int iter_fp = 0;
        bool success = false;

        double J = J0;

        for (; iter_fp < opts.line_search_max_iterations; ++iter_fp)
        {
          if (RolloutClosedLoop(alpha))
          {
            J = Cost(*Zbar_);
            double expected = -alpha * (deltaV_[0] + alpha * deltaV_[1]);
            if (expected > 0.0)
            {
              z = (J0 - J) / expected;
            }
            else
            {
              z = -1.0;
            }

            if (opts.line_search_lower_bound <= z && z <= opts.line_search_upper_bound && J < J0)
            {
              success = true;
              stats_.Log("cost", J);
              stats_.Log("alpha", alpha);
              stats_.Log("z", z);
              break;
            }
          }
          alpha /= opts.line_search_decrease_factor;
        }

        if (success)
        {
          (*Z_) = (*Zbar_);
        }
        else
        {
          IncreaseRegularization();
          J = J0;
        }

        if (J > J0)
        {
          status_ = SolverStatus::kCostIncrease;
        }
      }

      /**
       * @brief 评估检查收敛所需的信息
       */
      void UpdateConvergenceStatistics()
      {
        Stopwatch sw = stats_.GetTimer()->Start("stats");

        double dgrad = NormalizedFeedforwardGain();
        double dJ = 0.0;
        if (stats_.iterations_inner == 0)
        {
          dJ = stats_.initial_cost - stats_.cost.back();
        }
        else
        {
          dJ = stats_.cost.rbegin()[1] - stats_.cost.rbegin()[0];
        }

        stats_.iterations_inner++;
        stats_.iterations_total++;
        stats_.Log("dJ", dJ);
        stats_.Log("viol", max_violation_callback_());
        stats_.Log("iters", stats_.iterations_total);
        stats_.Log("grad", dgrad);
        stats_.NewIteration();
      }

      /**
       * @brief 检查求解器是否完成求解
       * @return 如果求解器应该停止迭代则返回true
       */
      bool IsDone()
      {
        Stopwatch sw = stats_.GetTimer()->Start("convergence_check");
        SolverOptions &opts = GetOptions();

        bool cost_decrease = stats_.cost_decrease.back() < opts.cost_tolerance;
        bool gradient = stats_.gradient.back() < opts.gradient_tolerance;
        bool is_done = false;

        if (cost_decrease && gradient)
        {
          status_ = SolverStatus::kSolved;
          is_done = true;
        }
        else if (stats_.iterations_inner >= opts.max_iterations_inner)
        {
          status_ = SolverStatus::kMaxInnerIterations;
          is_done = true;
        }
        else if (stats_.iterations_total >= opts.max_iterations_total)
        {
          status_ = SolverStatus::kMaxIterations;
          is_done = true;
        }
        else if (status_ != SolverStatus::kUnsolved)
        {
          is_done = true;
        }

        return is_done;
      }

      /**
       * @brief 初始化求解器以预计算任何所需信息
       */
      void SolveSetup()
      {
        Stopwatch sw = stats_.GetTimer()->Start("init");
        stats_.iterations_inner = 0;
        stats_.SetVerbosity(GetOptions().verbose);

        if (Z_ != nullptr)
        {
          int k;
          for (k = 0; k < N_; ++k)
          {
            Zbar_->SetStep(k, Z_->GetStep(k));
            Zbar_->SetTime(k, Z_->GetTime(k));
          }
          Zbar_->SetTime(N_, Z_->GetTime(N_));
        }

        ResetInternalVariables();
      }

      /**
       * @brief 在迭代停止后执行任何所需的操作
       */
      void WrapUp() {}

      /**
       * @brief 计算前馈增益的无穷范数
       * @return 归一化前馈增益的值
       */
      double NormalizedFeedforwardGain()
      {
        for (int k = 0; k < N_; ++k)
        {
          VectorNd<m> &d = GetKnotPointFunction(k).GetFeedforwardGain();
          grad_(k) = (d.array().abs() / (Z_->Control(k).array().abs() + 1)).maxCoeff();
        }
        return grad_.sum() / grad_.size();
      }

      /**
       * @brief 更新扩展的块
       * @param start 起始指标
       * @param stop 停止指标
       */
      void UpdateExpansionsBlock(int start, int stop)
      {
        for (int k = start; k < stop; ++k)
        {
          KnotPoint<n, m> &z = Z_->GetKnotPoint(k);
          knotpoints_[k]->CalcCostExpansion(z.State(), z.Control());
          knotpoints_[k]->CalcDynamicsExpansion(z.State(), z.Control(), z.GetTime(), z.GetStep());
          costs_(k) = GetKnotPointFunction(k).Cost(z.State(), z.Control());
        }
      }

    private:
      /**
       * @brief 重置内部变量
       */
      void ResetInternalVariables()
      {
        status_ = SolverStatus::kUnsolved;
        costs_ = VectorXd::Zero(N_ + 1);
        grad_ = VectorXd::Zero(N_);
        deltaV_[0] = 0.0;
        deltaV_[1] = 0.0;
        rho_ = GetOptions().bp_reg_initial;
        drho_ = 0.0;
      }

      /**
       * @brief 计算每个个体节点的成本
       * @param Z 轨迹
       */
      void CalcIndividualCosts(const Trajectory<n, m> &Z)
      {
        for (int k = 0; k <= N_; ++k)
        {
          costs_(k) = GetKnotPointFunction(k).Cost(Z.State(k), Z.Control(k));
        }
      }

      /**
       * @brief 增加正则化
       */
      void IncreaseRegularization()
      {
        const SolverOptions &opts = GetOptions();
        drho_ = std::max(drho_ * opts.bp_reg_increase_factor, opts.bp_reg_increase_factor);
        rho_ = std::max(rho_ * drho_, opts.bp_reg_min);
        rho_ = std::min(rho_, opts.bp_reg_max);
      }

      /**
       * @brief 减小正则化项
       */
      void DecreaseRegularization()
      {
        const SolverOptions &opts = GetOptions();
        drho_ = std::min(drho_ / opts.bp_reg_increase_factor, 1 / opts.bp_reg_increase_factor);
        rho_ = std::max(rho_ * drho_, opts.bp_reg_min);
        rho_ = std::min(rho_, opts.bp_reg_max);
      }

      int N_;                                   // 段数
      std::shared_ptr<VectorXd> initial_state_; // 初始状态指针
      SolverStats stats_;                       // 求解器统计信息（迭代次数、每次迭代成本等）

      std::vector<std::unique_ptr<KnotPointFunctions<n, m>>> knotpoints_; // 问题描述和数据
      std::shared_ptr<Trajectory<n, m>> Z_;                               // 当前轨迹猜测
      std::unique_ptr<Trajectory<n, m>> Zbar_;                            // 临时轨迹用于前向传递

      SolverStatus status_ = SolverStatus::kUnsolved; // 求解状态

      VectorXd costs_;               // 每个节点的成本
      VectorXd grad_;                // 每个节点的梯度
      double rho_ = 0.0;             // 正则化
      double drho_ = 0.0;            // 正则化导数（阻尼）
      std::array<double, 2> deltaV_; // 用于存储增量值

      bool is_initial_state_set = false; // 初始状态是否设置
      std::function<double()> max_violation_callback_ = []()
      { return 0.0; };             // 最大违反回调
      std::vector<int> work_inds_; // 工作索引
    };

  } // namespace ilqr
} // namespace altro
