#include "handeye_nonlinear_optimizer.h"

#ifdef _MSC_VER
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
#include <ceres/ceres.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <algorithm>
#include <cmath>

namespace
{
struct FixedCornerResidual
{
    FixedCornerResidual(
        const Eigen::Isometry3d& baseFromFlange,
        const Eigen::Vector3d& cameraPoint,
        const Eigen::Vector3d& targetBase,
        double weight)
        : baseRotation(baseFromFlange.linear()),
          baseTranslation(baseFromFlange.translation()),
          cameraPoint(cameraPoint),
          targetBase(targetBase),
          weight(weight)
    {
    }

    template <typename T>
    bool operator()(const T* const qFcData, const T* const tFcData, T* residual) const
    {
        const Eigen::Map<const Eigen::Quaternion<T>> qFc(qFcData);
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> tFc(tFcData);

        const Eigen::Matrix<T, 3, 1> cameraPointT = cameraPoint.cast<T>();
        const Eigen::Matrix<T, 3, 1> flangePoint = qFc * cameraPointT + tFc;
        const Eigen::Matrix<T, 3, 1> basePoint =
            baseRotation.cast<T>() * flangePoint + baseTranslation.cast<T>();
        const Eigen::Matrix<T, 3, 1> error = basePoint - targetBase.cast<T>();
        const T sqrtWeight = T(std::sqrt(weight));

        residual[0] = sqrtWeight * error.x();
        residual[1] = sqrtWeight * error.y();
        residual[2] = sqrtWeight * error.z();
        return true;
    }

    Eigen::Matrix3d baseRotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d baseTranslation = Eigen::Vector3d::Zero();
    Eigen::Vector3d cameraPoint = Eigen::Vector3d::Zero();
    Eigen::Vector3d targetBase = Eigen::Vector3d::Zero();
    double weight = 1.0;
};

void ComputeMetrics(
    const std::vector<HandEyeNonlinearOptimizer::Observation>& observations,
    const Eigen::Isometry3d& flangeFromCamera,
    HandEyeNonlinearOptimizer::Result& result)
{
    result.basePredictions.clear();
    result.errors.clear();
    result.basePredictions.reserve(observations.size());
    result.errors.reserve(observations.size());
    result.meanError = 0.0;
    result.rmsError = 0.0;
    result.maxError = 0.0;

    double sum = 0.0;
    double sumSq = 0.0;
    for (const auto& observation : observations)
    {
        const Eigen::Vector3d flangePoint = flangeFromCamera * observation.cameraPoint;
        const Eigen::Vector3d basePrediction = observation.baseFromFlange * flangePoint;
        const Eigen::Vector3d error = basePrediction - observation.targetBase;
        const double norm = error.norm();

        result.basePredictions.push_back(basePrediction);
        result.errors.push_back(error);
        sum += norm;
        sumSq += norm * norm;
        result.maxError = std::max(result.maxError, norm);
    }

    if (!observations.empty())
    {
        result.meanError = sum / static_cast<double>(observations.size());
        result.rmsError = std::sqrt(sumSq / static_cast<double>(observations.size()));
    }
}
} // namespace

HandEyeNonlinearOptimizer::HandEyeNonlinearOptimizer(Options options)
    : options_(options)
{
}

HandEyeNonlinearOptimizer::Result HandEyeNonlinearOptimizer::Optimize(
    const std::vector<Observation>& observations,
    const Eigen::Isometry3d& initialFlangeFromCamera) const
{
    Result result;
    result.flangeFromCamera = initialFlangeFromCamera;
    if (observations.empty())
    {
        result.briefReport = "No observations.";
        result.fullReport = result.briefReport;
        return result;
    }

    Eigen::Quaterniond qFc(initialFlangeFromCamera.linear());
    qFc.normalize();
    Eigen::Vector3d tFc = initialFlangeFromCamera.translation();

    ceres::Problem problem;
    problem.AddParameterBlock(qFc.coeffs().data(), 4, new ceres::EigenQuaternionManifold());
    problem.AddParameterBlock(tFc.data(), 3);

    for (const auto& observation : observations)
    {
        ceres::CostFunction* cost = new ceres::AutoDiffCostFunction<FixedCornerResidual, 3, 4, 3>(
            new FixedCornerResidual(
                observation.baseFromFlange,
                observation.cameraPoint,
                observation.targetBase,
                observation.weight));
        ceres::LossFunction* loss = nullptr;
        if (options_.huberLossMm > 0.0)
        {
            loss = new ceres::HuberLoss(options_.huberLossMm);
        }
        problem.AddResidualBlock(cost, loss, qFc.coeffs().data(), tFc.data());
    }

    ceres::Solver::Options solverOptions;
    solverOptions.linear_solver_type = ceres::DENSE_QR;
    solverOptions.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    solverOptions.max_num_iterations = options_.maxNumIterations;
    solverOptions.function_tolerance = 1e-10;
    solverOptions.gradient_tolerance = 1e-12;
    solverOptions.parameter_tolerance = 1e-12;
    solverOptions.minimizer_progress_to_stdout = options_.minimizerProgressToStdout;

    ceres::Solver::Summary summary;
    ceres::Solve(solverOptions, &problem, &summary);

    qFc.normalize();
    result.success = summary.IsSolutionUsable();
    result.initialCost = summary.initial_cost;
    result.finalCost = summary.final_cost;
    result.briefReport = summary.BriefReport();
    result.fullReport = summary.FullReport();
    result.flangeFromCamera = Eigen::Isometry3d::Identity();
    result.flangeFromCamera.linear() = qFc.toRotationMatrix();
    result.flangeFromCamera.translation() = tFc;
    ComputeMetrics(observations, result.flangeFromCamera, result);
    return result;
}
