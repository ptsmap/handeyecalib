#include "handeye_trihedral_bundle_optimizer.h"

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
#include <map>
#include <numeric>

namespace
{
using StationObservation = HandEyeTrihedralBundleOptimizer::StationObservation;
using PlaneObservation = HandEyeTrihedralBundleOptimizer::PlaneObservation;

struct MatchedStation
{
    const StationObservation* station = nullptr;
    std::array<int, 3> planeForAxis = {0, 1, 2};
};

struct TrihedralPlaneResidual
{
    TrihedralPlaneResidual(
        const Eigen::Isometry3d& baseFromFlange,
        const Eigen::Vector3d& cameraPoint,
        int axisIndex)
        : baseRotation(baseFromFlange.linear()),
          baseTranslation(baseFromFlange.translation()),
          cameraPoint(cameraPoint)
    {
        axis.setZero();
        axis(axisIndex) = 1.0;
    }

    template <typename T>
    bool operator()(
        const T* const qFcData,
        const T* const tFcData,
        const T* const qBkData,
        const T* const cornerBaseData,
        T* residual) const
    {
        const Eigen::Map<const Eigen::Quaternion<T>> qFc(qFcData);
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> tFc(tFcData);
        const Eigen::Map<const Eigen::Quaternion<T>> qBk(qBkData);
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> cornerBase(cornerBaseData);

        const Eigen::Matrix<T, 3, 1> flangePoint = qFc * cameraPoint.cast<T>() + tFc;
        const Eigen::Matrix<T, 3, 1> basePoint =
            baseRotation.cast<T>() * flangePoint + baseTranslation.cast<T>();
        const Eigen::Matrix<T, 3, 1> normalBase = qBk * axis.cast<T>();
        residual[0] = normalBase.dot(basePoint - cornerBase);
        return true;
    }

    Eigen::Matrix3d baseRotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d baseTranslation = Eigen::Vector3d::Zero();
    Eigen::Vector3d cameraPoint = Eigen::Vector3d::Zero();
    Eigen::Vector3d axis = Eigen::Vector3d::UnitX();
};

struct TargetTranslationResidual
{
    TargetTranslationResidual(
        const Eigen::Isometry3d& baseFromFlange,
        const Eigen::Matrix3d& flangeFromCameraRotation,
        const Eigen::Vector3d& cornerCamera,
        const Eigen::Vector3d& targetBase,
        double sigmaMm)
        : baseRotation(baseFromFlange.linear()),
          baseTranslation(baseFromFlange.translation()),
          flangeFromCameraRotation(flangeFromCameraRotation),
          cornerCamera(cornerCamera),
          targetBase(targetBase),
          invSigma(sigmaMm > 0.0 ? 1.0 / sigmaMm : 1.0)
    {
    }

    template <typename T>
    bool operator()(const T* const tFcData, T* residual) const
    {
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> tFc(tFcData);
        const Eigen::Matrix<T, 3, 1> flangePoint =
            flangeFromCameraRotation.cast<T>() * cornerCamera.cast<T>() + tFc;
        const Eigen::Matrix<T, 3, 1> basePoint =
            baseRotation.cast<T>() * flangePoint + baseTranslation.cast<T>();
        const Eigen::Matrix<T, 3, 1> error = (basePoint - targetBase.cast<T>()) * T(invSigma);
        residual[0] = error.x();
        residual[1] = error.y();
        residual[2] = error.z();
        return true;
    }

    Eigen::Matrix3d baseRotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d baseTranslation = Eigen::Vector3d::Zero();
    Eigen::Matrix3d flangeFromCameraRotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d cornerCamera = Eigen::Vector3d::Zero();
    Eigen::Vector3d targetBase = Eigen::Vector3d::Zero();
    double invSigma = 1.0;
};

struct TargetPoseResidual
{
    TargetPoseResidual(
        const Eigen::Isometry3d& baseFromFlange,
        const Eigen::Vector3d& cornerCamera,
        const Eigen::Vector3d& targetBase,
        double sigmaMm)
        : baseRotation(baseFromFlange.linear()),
          baseTranslation(baseFromFlange.translation()),
          cornerCamera(cornerCamera),
          targetBase(targetBase),
          invSigma(sigmaMm > 0.0 ? 1.0 / sigmaMm : 1.0)
    {
    }

    template <typename T>
    bool operator()(const T* const qFcData, const T* const tFcData, T* residual) const
    {
        const Eigen::Map<const Eigen::Quaternion<T>> qFc(qFcData);
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> tFc(tFcData);
        const Eigen::Matrix<T, 3, 1> flangePoint = qFc * cornerCamera.cast<T>() + tFc;
        const Eigen::Matrix<T, 3, 1> basePoint =
            baseRotation.cast<T>() * flangePoint + baseTranslation.cast<T>();
        const Eigen::Matrix<T, 3, 1> error = (basePoint - targetBase.cast<T>()) * T(invSigma);
        residual[0] = error.x();
        residual[1] = error.y();
        residual[2] = error.z();
        return true;
    }

    Eigen::Matrix3d baseRotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d baseTranslation = Eigen::Vector3d::Zero();
    Eigen::Vector3d cornerCamera = Eigen::Vector3d::Zero();
    Eigen::Vector3d targetBase = Eigen::Vector3d::Zero();
    double invSigma = 1.0;
};
Eigen::Vector3d TransformPlaneNormal(
    const StationObservation& station,
    const PlaneObservation& plane,
    const Eigen::Isometry3d& flangeFromCamera)
{
    return (station.baseFromFlange.linear() * flangeFromCamera.linear() * plane.normal).normalized();
}

std::array<int, 3> BestPlanePermutation(
    const std::array<Eigen::Vector3d, 3>& referenceNormals,
    const std::array<Eigen::Vector3d, 3>& normals,
    double* minAbsDot)
{
    std::array<int, 3> best = {0, 1, 2};
    std::array<int, 3> perm = {0, 1, 2};
    double bestScore = -std::numeric_limits<double>::infinity();
    double bestMinDot = 0.0;
    do
    {
        double score = 0.0;
        double localMinDot = std::numeric_limits<double>::max();
        for (int i = 0; i < 3; ++i)
        {
            const double absDot = std::abs(referenceNormals[i].dot(normals[perm[i]]));
            score += absDot;
            localMinDot = std::min(localMinDot, absDot);
        }
        if (score > bestScore)
        {
            bestScore = score;
            bestMinDot = localMinDot;
            best = perm;
        }
    } while (std::next_permutation(perm.begin(), perm.end()));

    if (minAbsDot)
    {
        *minAbsDot = bestMinDot;
    }
    return best;
}

Eigen::Matrix3d OrthonormalizeAxes(const std::array<Eigen::Vector3d, 3>& axes)
{
    Eigen::Matrix3d m;
    m.col(0) = axes[0];
    m.col(1) = axes[1];
    m.col(2) = axes[2];

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(m, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d r = svd.matrixU() * svd.matrixV().transpose();
    if (r.determinant() < 0.0)
    {
        Eigen::Matrix3d u = svd.matrixU();
        u.col(2) *= -1.0;
        r = u * svd.matrixV().transpose();
    }
    return r;
}

std::vector<MatchedStation> MatchStations(
    const std::vector<StationObservation>& stations,
    const Eigen::Isometry3d& flangeFromCamera,
    double minMatchedNormalDot,
    std::array<Eigen::Vector3d, 3>& averagedAxes)
{
    std::vector<MatchedStation> matched;
    averagedAxes = {
        Eigen::Vector3d::UnitX(),
        Eigen::Vector3d::UnitY(),
        Eigen::Vector3d::UnitZ(),
    };
    if (stations.empty())
    {
        return matched;
    }

    std::array<Eigen::Vector3d, 3> referenceNormals;
    for (int i = 0; i < 3; ++i)
    {
        referenceNormals[i] = TransformPlaneNormal(stations.front(), stations.front().planes[i], flangeFromCamera);
    }

    std::array<Eigen::Vector3d, 3> normalSums = {
        Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero(),
    };
    for (const StationObservation& station : stations)
    {
        std::array<Eigen::Vector3d, 3> normals;
        for (int i = 0; i < 3; ++i)
        {
            normals[i] = TransformPlaneNormal(station, station.planes[i], flangeFromCamera);
        }

        double minAbsDot = 0.0;
        const std::array<int, 3> perm = BestPlanePermutation(referenceNormals, normals, &minAbsDot);
        if (minAbsDot < minMatchedNormalDot)
        {
            continue;
        }

        MatchedStation matchedStation;
        matchedStation.station = &station;
        matchedStation.planeForAxis = perm;
        matched.push_back(matchedStation);

        for (int axisIndex = 0; axisIndex < 3; ++axisIndex)
        {
            Eigen::Vector3d normal = normals[perm[axisIndex]];
            if (referenceNormals[axisIndex].dot(normal) < 0.0)
            {
                normal = -normal;
            }
            normalSums[axisIndex] += normal;
        }
    }

    if (!matched.empty())
    {
        for (int i = 0; i < 3; ++i)
        {
            averagedAxes[i] = normalSums[i].norm() > 0.0
                ? normalSums[i].normalized()
                : referenceNormals[i];
        }
    }
    return matched;
}

void ComputeMetrics(
    const std::vector<MatchedStation>& matchedStations,
    const Eigen::Vector3d& cornerBase,
    const Eigen::Isometry3d& flangeFromCamera,
    const Eigen::Quaterniond& baseFromBlockRotation,
    HandEyeTrihedralBundleOptimizer::Result& result)
{
    double planeSumAbs = 0.0;
    double planeSumSq = 0.0;
    double planeMaxAbs = 0.0;
    size_t planeCount = 0;

    double cornerSum = 0.0;
    double cornerSumSq = 0.0;
    double cornerMax = 0.0;
    size_t cornerCount = 0;

    for (const MatchedStation& matchedStation : matchedStations)
    {
        const StationObservation& station = *matchedStation.station;
        const Eigen::Isometry3d baseFromCamera = station.baseFromFlange * flangeFromCamera;
        for (int axisIndex = 0; axisIndex < 3; ++axisIndex)
        {
            const Eigen::Vector3d normalBase = baseFromBlockRotation * Eigen::Vector3d::Unit(axisIndex);
            const PlaneObservation& plane = station.planes[matchedStation.planeForAxis[axisIndex]];
            for (const Eigen::Vector3d& cameraPoint : plane.points)
            {
                const Eigen::Vector3d basePoint = baseFromCamera * cameraPoint;
                const double distance = normalBase.dot(basePoint - cornerBase);
                const double absDistance = std::abs(distance);
                planeSumAbs += absDistance;
                planeSumSq += distance * distance;
                planeMaxAbs = std::max(planeMaxAbs, absDistance);
                ++planeCount;
            }
        }

        const Eigen::Vector3d observedCornerBase = baseFromCamera * station.cornerCamera;
        const double cornerError = (observedCornerBase - cornerBase).norm();
        cornerSum += cornerError;
        cornerSumSq += cornerError * cornerError;
        cornerMax = std::max(cornerMax, cornerError);
        ++cornerCount;
    }

    result.stationCount = matchedStations.size();
    result.planeResidualCount = planeCount;
    result.cornerResidualCount = cornerCount * 3;
    if (planeCount > 0)
    {
        result.planeMeanAbsDistanceMm = planeSumAbs / static_cast<double>(planeCount);
        result.planeRmsDistanceMm = std::sqrt(planeSumSq / static_cast<double>(planeCount));
        result.planeMaxAbsDistanceMm = planeMaxAbs;
    }
    if (cornerCount > 0)
    {
        result.cornerMeanErrorMm = cornerSum / static_cast<double>(cornerCount);
        result.cornerRmsErrorMm = std::sqrt(cornerSumSq / static_cast<double>(cornerCount));
        result.cornerMaxErrorMm = cornerMax;
    }
}
struct MetricsAccumulator
{
    double planeSumAbs = 0.0;
    double planeSumSq = 0.0;
    double planeMaxAbs = 0.0;
    size_t planeCount = 0;
    double cornerSum = 0.0;
    double cornerSumSq = 0.0;
    double cornerMax = 0.0;
    size_t cornerCount = 0;
};

void AccumulateMetrics(
    const std::vector<MatchedStation>& matchedStations,
    const Eigen::Vector3d& cornerBase,
    const Eigen::Isometry3d& flangeFromCamera,
    const Eigen::Quaterniond& baseFromBlockRotation,
    MetricsAccumulator& metrics)
{
    for (const MatchedStation& matchedStation : matchedStations)
    {
        const StationObservation& station = *matchedStation.station;
        const Eigen::Isometry3d baseFromCamera = station.baseFromFlange * flangeFromCamera;
        for (int axisIndex = 0; axisIndex < 3; ++axisIndex)
        {
            const Eigen::Vector3d normalBase = baseFromBlockRotation * Eigen::Vector3d::Unit(axisIndex);
            const PlaneObservation& plane = station.planes[matchedStation.planeForAxis[axisIndex]];
            for (const Eigen::Vector3d& cameraPoint : plane.points)
            {
                const Eigen::Vector3d basePoint = baseFromCamera * cameraPoint;
                const double distance = normalBase.dot(basePoint - cornerBase);
                const double absDistance = std::abs(distance);
                metrics.planeSumAbs += absDistance;
                metrics.planeSumSq += distance * distance;
                metrics.planeMaxAbs = std::max(metrics.planeMaxAbs, absDistance);
                ++metrics.planeCount;
            }
        }

        const Eigen::Vector3d observedCornerBase = baseFromCamera * station.cornerCamera;
        const double cornerError = (observedCornerBase - cornerBase).norm();
        metrics.cornerSum += cornerError;
        metrics.cornerSumSq += cornerError * cornerError;
        metrics.cornerMax = std::max(metrics.cornerMax, cornerError);
        ++metrics.cornerCount;
    }
}

void FinishMetrics(const MetricsAccumulator& metrics, HandEyeTrihedralBundleOptimizer::Result& result)
{
    result.stationCount = metrics.cornerCount;
    result.planeResidualCount = metrics.planeCount;
    result.cornerResidualCount = metrics.cornerCount * 3;
    if (metrics.planeCount > 0)
    {
        result.planeMeanAbsDistanceMm = metrics.planeSumAbs / static_cast<double>(metrics.planeCount);
        result.planeRmsDistanceMm = std::sqrt(metrics.planeSumSq / static_cast<double>(metrics.planeCount));
        result.planeMaxAbsDistanceMm = metrics.planeMaxAbs;
    }
    if (metrics.cornerCount > 0)
    {
        result.cornerMeanErrorMm = metrics.cornerSum / static_cast<double>(metrics.cornerCount);
        result.cornerRmsErrorMm = std::sqrt(metrics.cornerSumSq / static_cast<double>(metrics.cornerCount));
        result.cornerMaxErrorMm = metrics.cornerMax;
    }
}
} // namespace

HandEyeTrihedralBundleOptimizer::HandEyeTrihedralBundleOptimizer(Options options)
    : options_(options)
{
}

HandEyeTrihedralBundleOptimizer::Result HandEyeTrihedralBundleOptimizer::Optimize(
    const std::vector<StationObservation>& stations,
    const Eigen::Vector3d& targetBase,
    const Eigen::Isometry3d& initialFlangeFromCamera,
    double cornerMaxErrorMm) const
{
    return Optimize(stations, std::vector<Eigen::Vector3d>{targetBase}, initialFlangeFromCamera, cornerMaxErrorMm, false);
}

HandEyeTrihedralBundleOptimizer::Result HandEyeTrihedralBundleOptimizer::Optimize(
    const std::vector<StationObservation>& stations,
    const std::vector<Eigen::Vector3d>& targetBases,
    const Eigen::Isometry3d& initialFlangeFromCamera,
    double cornerMaxErrorMm,
    bool optimizeTargetPoseInSecondStage) const
{
    struct BlockState
    {
        int targetIndex = 0;
        std::vector<MatchedStation> matchedStations;
        Eigen::Quaterniond qBk = Eigen::Quaterniond::Identity();
        Eigen::Vector3d cornerBase = Eigen::Vector3d::Zero();
    };

    Result result;
    result.flangeFromCamera = initialFlangeFromCamera;
    result.baseFromBlock = Eigen::Isometry3d::Identity();
    if (!targetBases.empty())
    {
        result.baseFromBlock.translation() = targetBases.front();
    }
    result.baseFromBlocks.assign(targetBases.size(), Eigen::Isometry3d::Identity());
    for (size_t i = 0; i < targetBases.size(); ++i)
    {
        result.baseFromBlocks[i].translation() = targetBases[i];
    }
    result.targetPoseOptimized = optimizeTargetPoseInSecondStage && targetBases.size() >= 2;

    if (cornerMaxErrorMm >= options_.maxCornerErrorForTrihedralStageMm)
    {
        result.briefReport = "Skipped: corner max error is not below trihedral optimization threshold.";
        return result;
    }
    result.attempted = true;

    if (stations.size() < 3)
    {
        result.briefReport = "Not enough trihedral stations.";
        return result;
    }
    if (targetBases.empty())
    {
        result.briefReport = "No target base points.";
        return result;
    }

    std::map<int, std::vector<StationObservation>> stationsByTarget;
    size_t sourcePointCount = 0;
    for (const StationObservation& station : stations)
    {
        if (station.targetIndex < 0 || static_cast<size_t>(station.targetIndex) >= targetBases.size())
        {
            continue;
        }
        stationsByTarget[station.targetIndex].push_back(station);
        for (const PlaneObservation& plane : station.planes)
        {
            sourcePointCount += plane.points.size();
        }
    }
    if (sourcePointCount == 0)
    {
        result.briefReport = "No trihedral plane points.";
        return result;
    }

    std::vector<BlockState> blocks;
    for (const auto& entry : stationsByTarget)
    {
        if (entry.second.size() < 3)
        {
            continue;
        }

        std::array<Eigen::Vector3d, 3> averagedAxes;
        std::vector<MatchedStation> matchedStations = MatchStations(
            entry.second,
            initialFlangeFromCamera,
            options_.minMatchedNormalDot,
            averagedAxes);
        if (matchedStations.size() < 3)
        {
            continue;
        }

        BlockState block;
        block.targetIndex = entry.first;
        block.matchedStations = std::move(matchedStations);
        const Eigen::Matrix3d initialBaseFromBlockRotation = OrthonormalizeAxes(averagedAxes);
        block.qBk = Eigen::Quaterniond(initialBaseFromBlockRotation);
        block.qBk.normalize();
        block.cornerBase = Eigen::Vector3d::Zero();
        for (const MatchedStation& matchedStation : block.matchedStations)
        {
            const StationObservation& station = *matchedStation.station;
            block.cornerBase += station.baseFromFlange * (initialFlangeFromCamera * station.cornerCamera);
        }
        block.cornerBase /= static_cast<double>(block.matchedStations.size());
        blocks.push_back(std::move(block));
    }

    if (blocks.empty())
    {
        result.briefReport = "Not enough stations passed trihedral plane matching.";
        return result;
    }
    if (result.targetPoseOptimized && blocks.size() < 2)
    {
        result.targetPoseOptimized = false;
    }

    Eigen::Quaterniond qFc(initialFlangeFromCamera.linear());
    qFc.normalize();
    Eigen::Vector3d tFc = initialFlangeFromCamera.translation();

    ceres::Problem planeProblem;
    planeProblem.AddParameterBlock(qFc.coeffs().data(), 4, new ceres::EigenQuaternionManifold());
    planeProblem.AddParameterBlock(tFc.data(), 3);
    for (BlockState& block : blocks)
    {
        planeProblem.AddParameterBlock(block.qBk.coeffs().data(), 4, new ceres::EigenQuaternionManifold());
        planeProblem.AddParameterBlock(block.cornerBase.data(), 3);
    }

    for (BlockState& block : blocks)
    {
        for (const MatchedStation& matchedStation : block.matchedStations)
        {
            const StationObservation& station = *matchedStation.station;
            for (int axisIndex = 0; axisIndex < 3; ++axisIndex)
            {
                const PlaneObservation& plane = station.planes[matchedStation.planeForAxis[axisIndex]];
                for (const Eigen::Vector3d& cameraPoint : plane.points)
                {
                    ceres::CostFunction* cost = new ceres::AutoDiffCostFunction<TrihedralPlaneResidual, 1, 4, 3, 4, 3>(
                        new TrihedralPlaneResidual(station.baseFromFlange, cameraPoint, axisIndex));
                    ceres::LossFunction* loss = nullptr;
                    if (options_.planeHuberLossMm > 0.0)
                    {
                        loss = new ceres::HuberLoss(options_.planeHuberLossMm);
                    }
                    planeProblem.AddResidualBlock(
                        cost,
                        loss,
                        qFc.coeffs().data(),
                        tFc.data(),
                        block.qBk.coeffs().data(),
                        block.cornerBase.data());
                }
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

    ceres::Solver::Summary planeSummary;
    ceres::Solve(solverOptions, &planeProblem, &planeSummary);

    qFc.normalize();
    for (BlockState& block : blocks)
    {
        block.qBk.normalize();
    }
    const Eigen::Vector3d planeAlignedTranslation = tFc;
    const Eigen::Quaterniond planeAlignedRotation = qFc;

    ceres::Problem targetProblem;
    if (result.targetPoseOptimized)
    {
        targetProblem.AddParameterBlock(qFc.coeffs().data(), 4, new ceres::EigenQuaternionManifold());
        targetProblem.AddParameterBlock(tFc.data(), 3);
        for (const BlockState& block : blocks)
        {
            const Eigen::Vector3d& targetBase = targetBases[static_cast<size_t>(block.targetIndex)];
            for (const MatchedStation& matchedStation : block.matchedStations)
            {
                const StationObservation& station = *matchedStation.station;
                ceres::CostFunction* cost = new ceres::AutoDiffCostFunction<TargetPoseResidual, 3, 4, 3>(
                    new TargetPoseResidual(
                        station.baseFromFlange,
                        station.cornerCamera,
                        targetBase,
                        options_.targetAlignmentSigmaMm));
                ceres::LossFunction* loss = nullptr;
                if (options_.targetAlignmentHuberLossMm > 0.0)
                {
                    loss = new ceres::HuberLoss(options_.targetAlignmentHuberLossMm);
                }
                targetProblem.AddResidualBlock(cost, loss, qFc.coeffs().data(), tFc.data());
            }
        }
    }
    else
    {
        targetProblem.AddParameterBlock(tFc.data(), 3);
        const Eigen::Matrix3d flangeFromCameraRotation = qFc.toRotationMatrix();
        for (const BlockState& block : blocks)
        {
            const Eigen::Vector3d& targetBase = targetBases[static_cast<size_t>(block.targetIndex)];
            for (const MatchedStation& matchedStation : block.matchedStations)
            {
                const StationObservation& station = *matchedStation.station;
                ceres::CostFunction* cost = new ceres::AutoDiffCostFunction<TargetTranslationResidual, 3, 3>(
                    new TargetTranslationResidual(
                        station.baseFromFlange,
                        flangeFromCameraRotation,
                        station.cornerCamera,
                        targetBase,
                        options_.targetAlignmentSigmaMm));
                ceres::LossFunction* loss = nullptr;
                if (options_.targetAlignmentHuberLossMm > 0.0)
                {
                    loss = new ceres::HuberLoss(options_.targetAlignmentHuberLossMm);
                }
                targetProblem.AddResidualBlock(cost, loss, tFc.data());
            }
        }
    }

    ceres::Solver::Summary targetSummary;
    ceres::Solve(solverOptions, &targetProblem, &targetSummary);
    qFc.normalize();

    result.success = planeSummary.IsSolutionUsable() && targetSummary.IsSolutionUsable();
    result.initialCost = planeSummary.initial_cost + targetSummary.initial_cost;
    result.finalCost = planeSummary.final_cost + targetSummary.final_cost;
    result.briefReport = std::string("plane: ") + planeSummary.BriefReport()
        + (result.targetPoseOptimized ? "; target_pose: " : "; target_translation: ")
        + targetSummary.BriefReport();
    result.fullReport = std::string("Trihedral paired-plane alignment:\n") + planeSummary.FullReport()
        + (result.targetPoseOptimized ? "\nTarget pose alignment:\n" : "\nTarget translation alignment:\n")
        + targetSummary.FullReport();
    result.flangeFromCamera = Eigen::Isometry3d::Identity();
    result.flangeFromCamera.linear() = qFc.toRotationMatrix();
    result.flangeFromCamera.translation() = tFc;
    result.targetAlignmentTranslationDeltaMm = tFc - planeAlignedTranslation;
    const double rotationDeltaCos = std::clamp(std::abs(planeAlignedRotation.dot(qFc)), 0.0, 1.0);
    result.targetAlignmentRotationDeltaDeg = 2.0 * std::acos(rotationDeltaCos) * 180.0 / 3.14159265358979323846;

    result.baseFromBlocks.assign(targetBases.size(), Eigen::Isometry3d::Identity());
    for (size_t i = 0; i < targetBases.size(); ++i)
    {
        result.baseFromBlocks[i].translation() = targetBases[i];
    }
    for (const BlockState& block : blocks)
    {
        Eigen::Isometry3d baseFromBlock = Eigen::Isometry3d::Identity();
        baseFromBlock.linear() = block.qBk.toRotationMatrix();
        baseFromBlock.translation() = targetBases[static_cast<size_t>(block.targetIndex)];
        result.baseFromBlocks[static_cast<size_t>(block.targetIndex)] = baseFromBlock;
        if (block.targetIndex == 0)
        {
            result.baseFromBlock = baseFromBlock;
        }
    }
    if (!blocks.empty() && blocks.front().targetIndex != 0)
    {
        result.baseFromBlock = result.baseFromBlocks[static_cast<size_t>(blocks.front().targetIndex)];
    }

    MetricsAccumulator metrics;
    for (const BlockState& block : blocks)
    {
        AccumulateMetrics(
            block.matchedStations,
            targetBases[static_cast<size_t>(block.targetIndex)],
            result.flangeFromCamera,
            block.qBk,
            metrics);
    }
    FinishMetrics(metrics, result);
    return result;
}
