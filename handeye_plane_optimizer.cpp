#include "handeye_plane_optimizer.h"

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
#include <limits>
#include <numeric>

namespace
{
struct MatchedPlane
{
    Eigen::Vector3d normalBase = Eigen::Vector3d::UnitX();
    double dBase = 0.0;
};

struct PlaneResidual
{
    PlaneResidual(
        const Eigen::Isometry3d& baseFromFlange,
        const Eigen::Vector3d& cameraPoint,
        const Eigen::Vector3d& planeNormalBase,
        double planeDBase)
        : baseRotation(baseFromFlange.linear()),
          baseTranslation(baseFromFlange.translation()),
          cameraPoint(cameraPoint),
          planeNormalBase(planeNormalBase),
          planeDBase(planeDBase)
    {
    }

    template <typename T>
    bool operator()(const T* const qFcData, const T* const tFcData, T* residual) const
    {
        const Eigen::Map<const Eigen::Quaternion<T>> qFc(qFcData);
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> tFc(tFcData);

        const Eigen::Matrix<T, 3, 1> flangePoint = qFc * cameraPoint.cast<T>() + tFc;
        const Eigen::Matrix<T, 3, 1> basePoint =
            baseRotation.cast<T>() * flangePoint + baseTranslation.cast<T>();
        residual[0] = planeNormalBase.cast<T>().dot(basePoint) + T(planeDBase);
        return true;
    }

    Eigen::Matrix3d baseRotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d baseTranslation = Eigen::Vector3d::Zero();
    Eigen::Vector3d cameraPoint = Eigen::Vector3d::Zero();
    Eigen::Vector3d planeNormalBase = Eigen::Vector3d::UnitX();
    double planeDBase = 0.0;
};

Eigen::Vector3d TransformedPlaneNormal(
    const HandEyePlaneOptimizer::StationObservation& station,
    const HandEyePlaneOptimizer::PlaneObservation& plane,
    const Eigen::Isometry3d& flangeFromCamera)
{
    return (station.baseFromFlange.linear() * flangeFromCamera.linear() * plane.normal).normalized();
}

double TransformedPlaneD(
    const HandEyePlaneOptimizer::StationObservation& station,
    const HandEyePlaneOptimizer::PlaneObservation& plane,
    const Eigen::Isometry3d& flangeFromCamera,
    const Eigen::Vector3d& transformedNormal)
{
    const Eigen::Isometry3d baseFromCamera = station.baseFromFlange * flangeFromCamera;
    const Eigen::Vector3d pointOnPlaneCamera = -plane.d * plane.normal;
    const Eigen::Vector3d pointOnPlaneBase = baseFromCamera * pointOnPlaneCamera;
    return -transformedNormal.dot(pointOnPlaneBase);
}

std::array<int, 3> BestPlanePermutation(
    const std::array<Eigen::Vector3d, 3>& referenceNormals,
    const std::array<Eigen::Vector3d, 3>& normals)
{
    std::array<int, 3> best = {0, 1, 2};
    std::array<int, 3> perm = {0, 1, 2};
    double bestScore = -std::numeric_limits<double>::infinity();
    do
    {
        double score = 0.0;
        for (int i = 0; i < 3; ++i)
        {
            score += std::abs(referenceNormals[i].dot(normals[perm[i]]));
        }
        if (score > bestScore)
        {
            bestScore = score;
            best = perm;
        }
    } while (std::next_permutation(perm.begin(), perm.end()));
    return best;
}

std::array<MatchedPlane, 3> MatchAndFitTargetPlanes(
    const std::vector<HandEyePlaneOptimizer::StationObservation>& stations,
    const Eigen::Isometry3d& flangeFromCamera)
{
    std::array<std::vector<Eigen::Vector3d>, 3> normalGroups;
    std::array<std::vector<double>, 3> dGroups;

    std::array<Eigen::Vector3d, 3> referenceNormals;
    for (int i = 0; i < 3; ++i)
    {
        referenceNormals[i] = TransformedPlaneNormal(stations.front(), stations.front().planes[i], flangeFromCamera);
    }

    for (const HandEyePlaneOptimizer::StationObservation& station : stations)
    {
        std::array<Eigen::Vector3d, 3> normals;
        std::array<double, 3> distances;
        for (int i = 0; i < 3; ++i)
        {
            normals[i] = TransformedPlaneNormal(station, station.planes[i], flangeFromCamera);
            distances[i] = TransformedPlaneD(station, station.planes[i], flangeFromCamera, normals[i]);
        }

        const std::array<int, 3> perm = BestPlanePermutation(referenceNormals, normals);
        for (int targetIndex = 0; targetIndex < 3; ++targetIndex)
        {
            const int stationPlaneIndex = perm[targetIndex];
            Eigen::Vector3d normal = normals[stationPlaneIndex];
            double d = distances[stationPlaneIndex];
            if (referenceNormals[targetIndex].dot(normal) < 0.0)
            {
                normal = -normal;
                d = -d;
            }
            normalGroups[targetIndex].push_back(normal);
            dGroups[targetIndex].push_back(d);
        }
    }

    std::array<MatchedPlane, 3> targetPlanes;
    for (int targetIndex = 0; targetIndex < 3; ++targetIndex)
    {
        Eigen::Vector3d normalSum = Eigen::Vector3d::Zero();
        for (const Eigen::Vector3d& normal : normalGroups[targetIndex])
        {
            normalSum += normal;
        }
        Eigen::Vector3d normal = normalSum.norm() > 0.0 ? normalSum.normalized() : referenceNormals[targetIndex];

        double d = 0.0;
        for (size_t i = 0; i < normalGroups[targetIndex].size(); ++i)
        {
            double alignedD = dGroups[targetIndex][i];
            if (normal.dot(normalGroups[targetIndex][i]) < 0.0)
            {
                alignedD = -alignedD;
            }
            d += alignedD;
        }
        if (!dGroups[targetIndex].empty())
        {
            d /= static_cast<double>(dGroups[targetIndex].size());
        }

        targetPlanes[targetIndex].normalBase = normal;
        targetPlanes[targetIndex].dBase = d;
    }
    return targetPlanes;
}

void ComputePlaneMetrics(
    const std::vector<HandEyePlaneOptimizer::StationObservation>& stations,
    const Eigen::Isometry3d& flangeFromCamera,
    const std::array<MatchedPlane, 3>& targetPlanes,
    HandEyePlaneOptimizer::Result& result)
{
    double sumAbs = 0.0;
    double sumSq = 0.0;
    double maxAbs = 0.0;
    size_t count = 0;

    for (const HandEyePlaneOptimizer::StationObservation& station : stations)
    {
        const Eigen::Isometry3d baseFromCamera = station.baseFromFlange * flangeFromCamera;
        std::array<Eigen::Vector3d, 3> normals;
        for (int i = 0; i < 3; ++i)
        {
            normals[i] = TransformedPlaneNormal(station, station.planes[i], flangeFromCamera);
        }
        std::array<Eigen::Vector3d, 3> referenceNormals = {
            targetPlanes[0].normalBase,
            targetPlanes[1].normalBase,
            targetPlanes[2].normalBase,
        };
        const std::array<int, 3> perm = BestPlanePermutation(referenceNormals, normals);
        for (int targetIndex = 0; targetIndex < 3; ++targetIndex)
        {
            const auto& plane = station.planes[perm[targetIndex]];
            const MatchedPlane& target = targetPlanes[targetIndex];
            for (const Eigen::Vector3d& cameraPoint : plane.points)
            {
                const Eigen::Vector3d basePoint = baseFromCamera * cameraPoint;
                const double distance = target.normalBase.dot(basePoint) + target.dBase;
                const double absDistance = std::abs(distance);
                sumAbs += absDistance;
                sumSq += distance * distance;
                maxAbs = std::max(maxAbs, absDistance);
                ++count;
            }
        }
    }

    result.residualCount = count;
    if (count == 0)
    {
        return;
    }
    result.meanAbsDistanceMm = sumAbs / static_cast<double>(count);
    result.rmsDistanceMm = std::sqrt(sumSq / static_cast<double>(count));
    result.maxAbsDistanceMm = maxAbs;
}
} // namespace

HandEyePlaneOptimizer::HandEyePlaneOptimizer(Options options)
    : options_(options)
{
}

HandEyePlaneOptimizer::Result HandEyePlaneOptimizer::Optimize(
    const std::vector<StationObservation>& stations,
    const Eigen::Isometry3d& initialFlangeFromCamera,
    double cornerMaxErrorMm) const
{
    Result result;
    result.flangeFromCamera = initialFlangeFromCamera;
    if (cornerMaxErrorMm >= options_.maxCornerErrorForPlaneStageMm)
    {
        result.briefReport = "Skipped: corner max error is not below plane optimization threshold.";
        return result;
    }
    result.attempted = true;

    if (stations.size() < 3)
    {
        result.briefReport = "Not enough plane stations.";
        return result;
    }

    size_t sourcePointCount = 0;
    for (const StationObservation& station : stations)
    {
        for (const PlaneObservation& plane : station.planes)
        {
            sourcePointCount += plane.points.size();
        }
    }
    if (sourcePointCount == 0)
    {
        result.briefReport = "No plane points.";
        return result;
    }

    const std::array<MatchedPlane, 3> targetPlanes = MatchAndFitTargetPlanes(stations, initialFlangeFromCamera);

    Eigen::Quaterniond qFc(initialFlangeFromCamera.linear());
    qFc.normalize();
    Eigen::Vector3d tFc = initialFlangeFromCamera.translation();

    ceres::Problem problem;
    problem.AddParameterBlock(qFc.coeffs().data(), 4, new ceres::EigenQuaternionManifold());
    problem.AddParameterBlock(tFc.data(), 3);

    for (const StationObservation& station : stations)
    {
        std::array<Eigen::Vector3d, 3> normals;
        std::array<Eigen::Vector3d, 3> referenceNormals = {
            targetPlanes[0].normalBase,
            targetPlanes[1].normalBase,
            targetPlanes[2].normalBase,
        };
        for (int i = 0; i < 3; ++i)
        {
            normals[i] = TransformedPlaneNormal(station, station.planes[i], initialFlangeFromCamera);
        }
        const std::array<int, 3> perm = BestPlanePermutation(referenceNormals, normals);

        for (int targetIndex = 0; targetIndex < 3; ++targetIndex)
        {
            const PlaneObservation& plane = station.planes[perm[targetIndex]];
            const MatchedPlane& targetPlane = targetPlanes[targetIndex];
            for (const Eigen::Vector3d& cameraPoint : plane.points)
            {
                ceres::CostFunction* cost = new ceres::AutoDiffCostFunction<PlaneResidual, 1, 4, 3>(
                    new PlaneResidual(station.baseFromFlange, cameraPoint, targetPlane.normalBase, targetPlane.dBase));
                ceres::LossFunction* loss = nullptr;
                if (options_.huberLossMm > 0.0)
                {
                    loss = new ceres::HuberLoss(options_.huberLossMm);
                }
                problem.AddResidualBlock(cost, loss, qFc.coeffs().data(), tFc.data());
            }
        }
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

    const std::array<MatchedPlane, 3> finalTargetPlanes = MatchAndFitTargetPlanes(stations, result.flangeFromCamera);
    ComputePlaneMetrics(stations, result.flangeFromCamera, finalTargetPlanes, result);
    return result;
}
