#include "handeye_auto_calibrator.h"

#include "point_cloud_io.h"
#include "pose_io.h"

#ifdef _MSC_VER
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
#include <ceres/ceres.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace
{
constexpr double kPi = 3.14159265358979323846;

struct CandidateObservation
{
    std::string timestamp;
    fs::path cloudPath;
    fs::path featurePath;
    std::string groupName;
    Eigen::Isometry3d baseFromFlange = Eigen::Isometry3d::Identity();
    Eigen::Vector3d cameraPoint = Eigen::Vector3d::Zero();
    std::array<int, 3> planeIndices{};
    double candidateScore = 0.0;
    bool centroidDistanceOk = false;
    int candidateIndex = 0;
    int groupIndex = 0;
};

struct AutoSolveResult
{
    bool success = false;
    Eigen::Isometry3d flangeFromCamera = Eigen::Isometry3d::Identity();
    std::vector<Eigen::Vector3d> targetBases;
    std::vector<Eigen::Vector3d> basePredictions;
    std::vector<Eigen::Vector3d> errors;
    double initialCost = 0.0;
    double finalCost = 0.0;
    double meanErrorMm = 0.0;
    double rmsErrorMm = 0.0;
    double maxErrorMm = 0.0;
    std::string briefReport;
};

struct StationGroup
{
    std::string name;
    fs::path dataDir;
    fs::path featureDir;
    std::vector<handeye::StationFeatureInfo> stations;
};

struct FloatingTargetResidual
{
    FloatingTargetResidual(
        const Eigen::Isometry3d& baseFromFlange,
        const Eigen::Vector3d& cameraPoint,
        double weight)
        : baseRotation(baseFromFlange.linear()),
          baseTranslation(baseFromFlange.translation()),
          cameraPoint(cameraPoint),
          weight(weight)
    {
    }

    template <typename T>
    bool operator()(
        const T* const qFcData,
        const T* const tFcData,
        const T* const targetBaseData,
        T* residual) const
    {
        const Eigen::Map<const Eigen::Quaternion<T>> qFc(qFcData);
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> tFc(tFcData);
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> targetBase(targetBaseData);

        const Eigen::Matrix<T, 3, 1> cameraPointT = cameraPoint.cast<T>();
        const Eigen::Matrix<T, 3, 1> flangePoint = qFc * cameraPointT + tFc;
        const Eigen::Matrix<T, 3, 1> basePoint =
            baseRotation.cast<T>() * flangePoint + baseTranslation.cast<T>();
        const Eigen::Matrix<T, 3, 1> error = basePoint - targetBase;
        const T sqrtWeight = T(std::sqrt(weight));

        residual[0] = sqrtWeight * error.x();
        residual[1] = sqrtWeight * error.y();
        residual[2] = sqrtWeight * error.z();
        return true;
    }

    Eigen::Matrix3d baseRotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d baseTranslation = Eigen::Vector3d::Zero();
    Eigen::Vector3d cameraPoint = Eigen::Vector3d::Zero();
    double weight = 1.0;
};

std::array<double, 3> RpyFromRotation(const Eigen::Matrix3d& r)
{
    double rx = 0.0;
    double ry = 0.0;
    double rz = 0.0;

    const double sy = -r(2, 0);
    if (std::abs(sy) < 1.0 - 1e-12)
    {
        ry = std::asin(sy);
        rx = std::atan2(r(2, 1), r(2, 2));
        rz = std::atan2(r(1, 0), r(0, 0));
    }
    else
    {
        ry = std::copysign(kPi / 2.0, sy);
        rx = std::atan2(-r(0, 1), r(1, 1));
        rz = 0.0;
    }

    return {
        rx * 180.0 / kPi,
        ry * 180.0 / kPi,
        rz * 180.0 / kPi,
    };
}

void PrintVector(std::ostream& out, const Eigen::Vector3d& value)
{
    out << value.x() << ' ' << value.y() << ' ' << value.z();
}

bool WriteHandEyeMatrix(const Eigen::Isometry3d& flangeFromCamera, const fs::path& matrixPath)
{
    std::ofstream out(matrixPath);
    if (!out)
    {
        std::cerr << "Cannot write hand-eye matrix: " << matrixPath.string() << "\n";
        return false;
    }

    const Eigen::Matrix4d matrix = flangeFromCamera.matrix();
    out << std::fixed << std::setprecision(6);
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            if (row != 0 || col != 0)
            {
                out << ' ';
            }
            out << matrix(row, col);
        }
    }
    out << '\n';
    return true;
}

std::vector<Eigen::Matrix3d> BuildRotationSeeds()
{
    std::vector<Eigen::Matrix3d> seeds;
    seeds.push_back(Eigen::Matrix3d::Identity());

    std::array<int, 3> perm = {0, 1, 2};
    do
    {
        for (int sx : {-1, 1})
        {
            for (int sy : {-1, 1})
            {
                for (int sz : {-1, 1})
                {
                    Eigen::Matrix3d rotation = Eigen::Matrix3d::Zero();
                    rotation(perm[0], 0) = static_cast<double>(sx);
                    rotation(perm[1], 1) = static_cast<double>(sy);
                    rotation(perm[2], 2) = static_cast<double>(sz);
                    if (rotation.determinant() < 0.5)
                    {
                        continue;
                    }

                    const bool exists = std::any_of(seeds.begin(), seeds.end(),
                        [&rotation](const Eigen::Matrix3d& seed) {
                            return (seed - rotation).norm() < 1e-12;
                        });
                    if (!exists)
                    {
                        seeds.push_back(rotation);
                    }
                }
            }
        }
    }
    while (std::next_permutation(perm.begin(), perm.end()));

    return seeds;
}

int MaxGroupIndex(const std::vector<CandidateObservation>& observations)
{
    int maxGroupIndex = -1;
    for (const CandidateObservation& observation : observations)
    {
        maxGroupIndex = std::max(maxGroupIndex, observation.groupIndex);
    }
    return maxGroupIndex;
}

size_t GroupCount(const std::vector<CandidateObservation>& observations)
{
    const int maxGroupIndex = MaxGroupIndex(observations);
    return maxGroupIndex >= 0 ? static_cast<size_t>(maxGroupIndex + 1) : 0;
}

std::vector<int> CountObservationsPerGroup(
    const std::vector<CandidateObservation>& observations,
    size_t groupCount)
{
    std::vector<int> counts(groupCount, 0);
    for (const CandidateObservation& observation : observations)
    {
        if (observation.groupIndex >= 0 && static_cast<size_t>(observation.groupIndex) < groupCount)
        {
            ++counts[static_cast<size_t>(observation.groupIndex)];
        }
    }
    return counts;
}

bool HasMinimumObservationsPerGroup(
    const std::vector<CandidateObservation>& observations,
    const handeye::AutoCalibrateParams& params)
{
    const size_t groupCount = GroupCount(observations);
    if (groupCount == 0)
    {
        return false;
    }

    const std::vector<int> counts = CountObservationsPerGroup(observations, groupCount);
    return std::all_of(counts.begin(), counts.end(), [&params](int count) {
        return count >= params.minStationCount;
    });
}

bool SolveTranslationAndTargetsForRotation(
    const std::vector<CandidateObservation>& observations,
    size_t groupCount,
    const Eigen::Matrix3d& cameraRotation,
    Eigen::Vector3d& flangeTranslation,
    std::vector<Eigen::Vector3d>& targetBases,
    double& rmsError)
{
    if (observations.empty() || groupCount == 0)
    {
        return false;
    }

    const Eigen::Index unknownCount = static_cast<Eigen::Index>(3 + 3 * groupCount);
    Eigen::MatrixXd a(static_cast<Eigen::Index>(observations.size() * 3), unknownCount);
    Eigen::VectorXd b(static_cast<Eigen::Index>(observations.size() * 3));
    a.setZero();
    for (size_t i = 0; i < observations.size(); ++i)
    {
        const CandidateObservation& observation = observations[i];
        if (observation.groupIndex < 0 || static_cast<size_t>(observation.groupIndex) >= groupCount)
        {
            return false;
        }

        const Eigen::Matrix3d baseRotation = observation.baseFromFlange.linear();
        const Eigen::Vector3d baseTranslation = observation.baseFromFlange.translation();
        const Eigen::Index row = static_cast<Eigen::Index>(i * 3);
        const Eigen::Index targetCol = static_cast<Eigen::Index>(3 + 3 * observation.groupIndex);

        a.block<3, 3>(row, 0) = baseRotation;
        a.block<3, 3>(row, targetCol) = -Eigen::Matrix3d::Identity();
        b.segment<3>(row) = -(baseRotation * cameraRotation * observation.cameraPoint + baseTranslation);
    }

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(a);
    if (qr.rank() < unknownCount)
    {
        return false;
    }

    const Eigen::VectorXd x = qr.solve(b);
    flangeTranslation = x.segment<3>(0);
    targetBases.assign(groupCount, Eigen::Vector3d::Zero());
    for (size_t groupIndex = 0; groupIndex < groupCount; ++groupIndex)
    {
        targetBases[groupIndex] = x.segment<3>(static_cast<Eigen::Index>(3 + 3 * groupIndex));
    }

    double sumSq = 0.0;
    for (const CandidateObservation& observation : observations)
    {
        const Eigen::Vector3d flangePoint = cameraRotation * observation.cameraPoint + flangeTranslation;
        const Eigen::Vector3d basePoint = observation.baseFromFlange * flangePoint;
        const Eigen::Vector3d error = basePoint - targetBases[static_cast<size_t>(observation.groupIndex)];
        sumSq += error.squaredNorm();
    }
    rmsError = std::sqrt(sumSq / static_cast<double>(observations.size()));
    return std::isfinite(rmsError);
}

bool EstimateInitialTransformAndTargets(
    const std::vector<CandidateObservation>& observations,
    Eigen::Isometry3d& flangeFromCamera,
    std::vector<Eigen::Vector3d>& targetBases,
    double& rmsError)
{
    const size_t groupCount = GroupCount(observations);
    if (groupCount == 0)
    {
        return false;
    }

    const std::vector<Eigen::Matrix3d> seeds = BuildRotationSeeds();
    bool found = false;
    double bestRms = std::numeric_limits<double>::max();
    Eigen::Matrix3d bestRotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d bestTranslation = Eigen::Vector3d::Zero();
    std::vector<Eigen::Vector3d> bestTargets;

    for (const Eigen::Matrix3d& seed : seeds)
    {
        Eigen::Vector3d t = Eigen::Vector3d::Zero();
        std::vector<Eigen::Vector3d> targets;
        double seedRms = std::numeric_limits<double>::max();
        if (!SolveTranslationAndTargetsForRotation(observations, groupCount, seed, t, targets, seedRms))
        {
            continue;
        }
        if (seedRms < bestRms)
        {
            found = true;
            bestRms = seedRms;
            bestRotation = seed;
            bestTranslation = t;
            bestTargets = std::move(targets);
        }
    }

    if (!found)
    {
        return false;
    }

    flangeFromCamera = Eigen::Isometry3d::Identity();
    flangeFromCamera.linear() = bestRotation;
    flangeFromCamera.translation() = bestTranslation;
    targetBases = std::move(bestTargets);
    rmsError = bestRms;
    return true;
}
void ComputeMetrics(
    const std::vector<CandidateObservation>& observations,
    const Eigen::Isometry3d& flangeFromCamera,
    const std::vector<Eigen::Vector3d>& targetBases,
    AutoSolveResult& result)
{
    result.basePredictions.clear();
    result.errors.clear();
    result.basePredictions.reserve(observations.size());
    result.errors.reserve(observations.size());
    result.meanErrorMm = 0.0;
    result.rmsErrorMm = 0.0;
    result.maxErrorMm = 0.0;

    double sum = 0.0;
    double sumSq = 0.0;
    for (const CandidateObservation& observation : observations)
    {
        if (observation.groupIndex < 0 || static_cast<size_t>(observation.groupIndex) >= targetBases.size())
        {
            continue;
        }

        const Eigen::Vector3d flangePoint = flangeFromCamera * observation.cameraPoint;
        const Eigen::Vector3d basePoint = observation.baseFromFlange * flangePoint;
        const Eigen::Vector3d error = basePoint - targetBases[static_cast<size_t>(observation.groupIndex)];
        const double norm = error.norm();
        result.basePredictions.push_back(basePoint);
        result.errors.push_back(error);
        sum += norm;
        sumSq += norm * norm;
        result.maxErrorMm = std::max(result.maxErrorMm, norm);
    }

    if (!result.errors.empty())
    {
        result.meanErrorMm = sum / static_cast<double>(result.errors.size());
        result.rmsErrorMm = std::sqrt(sumSq / static_cast<double>(result.errors.size()));
    }
}

std::optional<AutoSolveResult> SolveAutoCalibration(
    const std::vector<CandidateObservation>& observations,
    const handeye::AutoCalibrateParams& params)
{
    if (!HasMinimumObservationsPerGroup(observations, params))
    {
        return std::nullopt;
    }

    Eigen::Isometry3d initialFlangeFromCamera = Eigen::Isometry3d::Identity();
    std::vector<Eigen::Vector3d> targetBases;
    double initialRms = std::numeric_limits<double>::max();
    if (!EstimateInitialTransformAndTargets(observations, initialFlangeFromCamera, targetBases, initialRms))
    {
        return std::nullopt;
    }

    Eigen::Quaterniond qFc(initialFlangeFromCamera.linear());
    qFc.normalize();
    Eigen::Vector3d tFc = initialFlangeFromCamera.translation();

    ceres::Problem problem;
    problem.AddParameterBlock(qFc.coeffs().data(), 4, new ceres::EigenQuaternionManifold());
    problem.AddParameterBlock(tFc.data(), 3);
    for (Eigen::Vector3d& targetBase : targetBases)
    {
        problem.AddParameterBlock(targetBase.data(), 3);
    }

    for (const CandidateObservation& observation : observations)
    {
        if (observation.groupIndex < 0 || static_cast<size_t>(observation.groupIndex) >= targetBases.size())
        {
            return std::nullopt;
        }

        ceres::CostFunction* cost = new ceres::AutoDiffCostFunction<FloatingTargetResidual, 3, 4, 3, 3>(
            new FloatingTargetResidual(observation.baseFromFlange, observation.cameraPoint, 1.0));
        ceres::LossFunction* loss = nullptr;
        if (params.cornerHuberLossMm > 0.0)
        {
            loss = new ceres::HuberLoss(params.cornerHuberLossMm);
        }
        problem.AddResidualBlock(
            cost,
            loss,
            qFc.coeffs().data(),
            tFc.data(),
            targetBases[static_cast<size_t>(observation.groupIndex)].data());
    }

    ceres::Solver::Options solverOptions;
    solverOptions.linear_solver_type = ceres::DENSE_QR;
    solverOptions.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    solverOptions.max_num_iterations = params.cornerMaxIterations;
    solverOptions.function_tolerance = 1e-10;
    solverOptions.gradient_tolerance = 1e-12;
    solverOptions.parameter_tolerance = 1e-12;
    solverOptions.minimizer_progress_to_stdout = params.cornerMinimizerProgressToStdout;

    ceres::Solver::Summary summary;
    ceres::Solve(solverOptions, &problem, &summary);

    qFc.normalize();
    AutoSolveResult result;
    result.success = summary.IsSolutionUsable();
    result.initialCost = summary.initial_cost;
    result.finalCost = summary.final_cost;
    result.briefReport = summary.BriefReport();
    result.flangeFromCamera = Eigen::Isometry3d::Identity();
    result.flangeFromCamera.linear() = qFc.toRotationMatrix();
    result.flangeFromCamera.translation() = tFc;
    result.targetBases = std::move(targetBases);
    ComputeMetrics(observations, result.flangeFromCamera, result.targetBases, result);
    return result;
}

CandidateObservation MakeCandidateObservation(
    const handeye::StationFeatureInfo& station,
    const handeye::TrihedralCornerInfo& corner,
    int candidateIndex,
    int groupIndex,
    const std::string& groupName,
    const fs::path& featureDir)
{
    CandidateObservation observation;
    observation.timestamp = station.timestamp;
    observation.cloudPath = station.cloudPath;
    observation.featurePath = featureDir.empty() ? fs::path() : (featureDir / station.cloudPath.filename());
    observation.groupName = groupName;
    observation.baseFromFlange = station.baseFromFlange;
    observation.cameraPoint = corner.point;
    observation.planeIndices = corner.planeIndices;
    observation.candidateScore = corner.score;
    observation.centroidDistanceOk = corner.centroidDistanceOk;
    observation.candidateIndex = candidateIndex;
    observation.groupIndex = groupIndex;
    return observation;
}

std::vector<CandidateObservation> BuildObservationsFromState(
    const std::vector<StationGroup>& groups,
    const std::vector<std::vector<int>>& candidateIndices)
{
    std::vector<CandidateObservation> observations;
    for (size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex)
    {
        const StationGroup& group = groups[groupIndex];
        if (groupIndex >= candidateIndices.size())
        {
            break;
        }
        if (candidateIndices[groupIndex].size() > group.stations.size())
        {
            return {};
        }

        for (size_t stationIndex = 0; stationIndex < candidateIndices[groupIndex].size(); ++stationIndex)
        {
            const int selectedIndex = candidateIndices[groupIndex][stationIndex];
            if (selectedIndex < 0 || static_cast<size_t>(selectedIndex) >= group.stations[stationIndex].cornerCandidates.size())
            {
                return {};
            }

            observations.push_back(MakeCandidateObservation(
                group.stations[stationIndex],
                group.stations[stationIndex].cornerCandidates[static_cast<size_t>(selectedIndex)],
                selectedIndex,
                static_cast<int>(groupIndex),
                group.name,
                group.featureDir));
        }
    }
    return observations;
}

std::vector<CandidateObservation> SelectGloballyConsistentObservations(
    const std::vector<StationGroup>& groups,
    const handeye::AutoCalibrateParams& params)
{
    if (groups.empty())
    {
        return {};
    }
    for (const StationGroup& group : groups)
    {
        if (static_cast<int>(group.stations.size()) < params.minStationCount)
        {
            return {};
        }
    }

    struct BeamState
    {
        std::vector<std::vector<int>> candidateIndices;
        double score = std::numeric_limits<double>::max();
        double rmsErrorMm = std::numeric_limits<double>::max();
    };

    const size_t beamWidth = std::max<size_t>(1, params.candidateBeamWidth);
    const size_t candidatesPerStation = std::max<size_t>(1, params.maxCandidatesPerStation);
    std::vector<BeamState> beam(1);
    beam.front().candidateIndices.resize(groups.size());

    for (size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex)
    {
        const StationGroup& group = groups[groupIndex];
        for (size_t stationIndex = 0; stationIndex < group.stations.size(); ++stationIndex)
        {
            const handeye::StationFeatureInfo& station = group.stations[stationIndex];
            const size_t candidateCount = std::min(candidatesPerStation, station.cornerCandidates.size());
            if (candidateCount == 0)
            {
                return {};
            }

            std::vector<BeamState> next;
            next.reserve(beam.size() * candidateCount);
            for (const BeamState& state : beam)
            {
                for (size_t candidateIndex = 0; candidateIndex < candidateCount; ++candidateIndex)
                {
                    BeamState candidateState = state;
                    candidateState.candidateIndices[groupIndex].push_back(static_cast<int>(candidateIndex));

                    std::vector<CandidateObservation> observations =
                        BuildObservationsFromState(groups, candidateState.candidateIndices);
                    if (observations.empty())
                    {
                        continue;
                    }

                    double geometryScore = 0.0;
                    for (const CandidateObservation& observation : observations)
                    {
                        geometryScore += observation.candidateScore;
                    }
                    geometryScore /= static_cast<double>(observations.size());

                    if (HasMinimumObservationsPerGroup(observations, params))
                    {
                        const std::optional<AutoSolveResult> solveResult = SolveAutoCalibration(observations, params);
                        if (!solveResult)
                        {
                            continue;
                        }
                        candidateState.rmsErrorMm = solveResult->rmsErrorMm;
                        candidateState.score =
                            solveResult->rmsErrorMm + params.candidateGeometryScoreWeight * geometryScore;
                    }
                    else
                    {
                        candidateState.rmsErrorMm = geometryScore;
                        candidateState.score = geometryScore;
                    }
                    next.push_back(std::move(candidateState));
                }
            }

            if (next.empty())
            {
                return {};
            }

            std::sort(next.begin(), next.end(), [](const BeamState& lhs, const BeamState& rhs) {
                return lhs.score < rhs.score;
            });
            if (next.size() > beamWidth)
            {
                next.resize(beamWidth);
            }
            beam = std::move(next);
        }
    }

    if (beam.empty())
    {
        return {};
    }
    return BuildObservationsFromState(groups, beam.front().candidateIndices);
}
std::string GroupNameFromIndex(size_t groupIndex)
{
    return "group_" + std::to_string(groupIndex + 1);
}

bool IsGroupDataDir(const fs::path& path)
{
    if (!fs::is_directory(path))
    {
        return false;
    }

    const std::string name = path.filename().string();
    const std::string lower = [&name]() {
        std::string out = name;
        std::transform(out.begin(), out.end(), out.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        return out;
    }();

    if (lower.rfind("group", 0) != 0 && lower.rfind("pos", 0) != 0 && lower.rfind("position", 0) != 0)
    {
        return false;
    }

    return fs::exists(path / "robot_records.pose")
        || std::any_of(fs::directory_iterator(path), fs::directory_iterator(), [](const fs::directory_entry& entry) {
            return entry.is_regular_file() && entry.path().extension() == ".pose";
        });
}

std::vector<fs::path> FindGroupDataDirs(const fs::path& dataDir)
{
    std::vector<fs::path> groupDirs;
    for (const fs::directory_entry& entry : fs::directory_iterator(dataDir))
    {
        if (entry.is_directory() && IsGroupDataDir(entry.path()))
        {
            groupDirs.push_back(entry.path());
        }
    }
    std::sort(groupDirs.begin(), groupDirs.end(), [](const fs::path& lhs, const fs::path& rhs) {
        return lhs.filename().string() < rhs.filename().string();
    });
    return groupDirs;
}

void AppendFeatureExtractionResult(
    handeye::FeatureExtractionResult& aggregate,
    const handeye::FeatureExtractionResult& current)
{
    aggregate.stationCount += current.stationCount;
    aggregate.skippedStationCount += current.skippedStationCount;
    aggregate.stations.insert(aggregate.stations.end(), current.stations.begin(), current.stations.end());
    if (aggregate.tcpFilePath.empty())
    {
        aggregate.tcpFilePath = current.tcpFilePath;
    }
    if (aggregate.targetBases.empty())
    {
        aggregate.targetBases = current.targetBases;
    }
}

std::vector<StationGroup> BuildStationGroups(
    const handeye::AutoCalibrateParams& params,
    const fs::path& absoluteDataDir,
    handeye::FeatureExtractionResult& featureExtraction,
    std::string& errorMessage)
{
    std::vector<StationGroup> groups;
    featureExtraction = handeye::FeatureExtractionResult{};
    featureExtraction.code = 0;

    if (!params.stationGroups.empty())
    {
        groups.reserve(params.stationGroups.size());
        for (size_t groupIndex = 0; groupIndex < params.stationGroups.size(); ++groupIndex)
        {
            StationGroup group;
            group.name = GroupNameFromIndex(groupIndex);
            group.dataDir = absoluteDataDir;
            group.featureDir = absoluteDataDir.parent_path() / "feature" / group.name;
            group.stations = params.stationGroups[groupIndex];
            featureExtraction.stations.insert(
                featureExtraction.stations.end(), group.stations.begin(), group.stations.end());
            groups.push_back(std::move(group));
        }
        featureExtraction.stationCount = static_cast<int>(featureExtraction.stations.size());
        featureExtraction.message = "Using pre-extracted grouped station features";
        return groups;
    }

    if (!params.stations.empty())
    {
        StationGroup group;
        group.name = GroupNameFromIndex(0);
        group.dataDir = absoluteDataDir;
        group.featureDir = absoluteDataDir.parent_path() / "feature";
        group.stations = params.stations;
        featureExtraction.stations = params.stations;
        featureExtraction.stationCount = static_cast<int>(params.stations.size());
        featureExtraction.message = "Using pre-extracted station features";
        groups.push_back(std::move(group));
        return groups;
    }

    std::vector<fs::path> groupDirs = FindGroupDataDirs(absoluteDataDir);
    if (groupDirs.empty())
    {
        groupDirs.push_back(absoluteDataDir);
    }

    groups.reserve(groupDirs.size());
    for (size_t groupIndex = 0; groupIndex < groupDirs.size(); ++groupIndex)
    {
        const fs::path& groupDir = groupDirs[groupIndex];
        StationGroup group;
        group.name = groupDirs.size() == 1 && groupDir == absoluteDataDir
            ? GroupNameFromIndex(0)
            : groupDir.filename().string();
        group.dataDir = groupDir;
        group.featureDir = groupDirs.size() == 1 && groupDir == absoluteDataDir
            ? absoluteDataDir.parent_path() / "feature"
            : absoluteDataDir / "feature" / group.name;

        handeye::FeatureExtractionParams featureParams = params.featureParams;
        featureParams.dataDir = groupDir;
        if (featureParams.featureOutputDir.empty())
        {
            featureParams.featureOutputDir = group.featureDir;
        }

        const handeye::FeatureExtractionResult current = handeye::RunFeatureExtraction(featureParams);
        if (current.code != 0)
        {
            errorMessage = current.message;
            featureExtraction = current;
            return {};
        }

        group.stations = current.stations;
        AppendFeatureExtractionResult(featureExtraction, current);
        groups.push_back(std::move(group));
    }

    featureExtraction.message = "Feature extraction completed for " + std::to_string(groups.size()) + " group(s)";
    return groups;
}
void PopulatePublicResult(
    const std::vector<CandidateObservation>& observations,
    const AutoSolveResult& solveResult,
    int removedStationCount,
    handeye::AutoCalibrateResult& result)
{
    const Eigen::Isometry3d& flangeFromCamera = solveResult.flangeFromCamera;
    result.flangeFromCamera = flangeFromCamera;
    result.cameraCenterInFlange = flangeFromCamera.translation();
    result.quaternion = Eigen::Quaterniond(flangeFromCamera.linear());
    const std::array<double, 3> rpy = RpyFromRotation(flangeFromCamera.linear());
    result.rpyDeg = Eigen::Vector3d(rpy[0], rpy[1], rpy[2]);

    const Eigen::Matrix4d matrix = flangeFromCamera.matrix();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            result.matrix4x4[static_cast<size_t>(row * 4 + col)] = matrix(row, col);
        }
    }

    result.stationCount = static_cast<int>(observations.size());
    result.targetCount = static_cast<int>(solveResult.targetBases.size());
    result.targetBases = solveResult.targetBases;
    result.estimatedTargetBases = solveResult.targetBases;
    result.estimatedTargetBase = solveResult.targetBases.empty()
        ? Eigen::Vector3d::Zero()
        : solveResult.targetBases.front();

    result.nonlinearOptimizer.success = solveResult.success;
    result.nonlinearOptimizer.initialCost = solveResult.initialCost;
    result.nonlinearOptimizer.finalCost = solveResult.finalCost;
    result.nonlinearOptimizer.meanErrorMm = solveResult.meanErrorMm;
    result.nonlinearOptimizer.rmsErrorMm = solveResult.rmsErrorMm;
    result.nonlinearOptimizer.maxErrorMm = solveResult.maxErrorMm;
    result.nonlinearOptimizer.briefReport = solveResult.briefReport;

    result.meanErrorMm = solveResult.meanErrorMm;
    result.rmsErrorMm = solveResult.rmsErrorMm;
    result.maxErrorMm = solveResult.maxErrorMm;
    result.removedStationCount = removedStationCount;

    result.stationErrors.clear();
    result.observations.clear();
    for (size_t i = 0; i < observations.size() && i < solveResult.errors.size(); ++i)
    {
        handeye::StationCornerError stationError;
        stationError.timestamp = observations[i].timestamp;
        stationError.cameraPoint = observations[i].cameraPoint;
        stationError.predictedBase = solveResult.basePredictions[i];
        stationError.error = solveResult.errors[i];
        stationError.errorNorm = solveResult.errors[i].norm();
        stationError.targetIndex = observations[i].groupIndex;
        stationError.planeIndices = observations[i].planeIndices;
        stationError.candidateScore = observations[i].candidateScore;
        stationError.centroidDistanceOk = observations[i].centroidDistanceOk;
        result.stationErrors.push_back(stationError);

        handeye::CalibrationObservation observation;
        observation.timestamp = observations[i].timestamp;
        observation.cloudPath = observations[i].cloudPath;
        observation.baseFromFlange = observations[i].baseFromFlange;
        observation.targetIndex = observations[i].groupIndex;
        observation.cameraPoint = observations[i].cameraPoint;
        observation.planeIndices = observations[i].planeIndices;
        observation.candidateScore = observations[i].candidateScore;
        observation.centroidDistanceOk = observations[i].centroidDistanceOk;
        result.observations.push_back(observation);
    }
}

void WriteAutoReport(
    std::ostream& out,
    const fs::path& dataDir,
    const std::vector<CandidateObservation>& observations,
    const AutoSolveResult& solveResult,
    int removedStationCount)
{
    const Eigen::Matrix3d rotation = solveResult.flangeFromCamera.linear();
    const Eigen::Vector3d translation = solveResult.flangeFromCamera.translation();
    const Eigen::Quaterniond quaternion(rotation);
    const std::array<double, 3> rpy = RpyFromRotation(rotation);

    out << std::fixed << std::setprecision(6);
    out << "Auto hand-eye calibration report\n";
    out << "Data dir: " << dataDir.string() << "\n";
    out << "Manual target teaching: no\n";
    out << "Target group count: " << solveResult.targetBases.size() << "\n";
    for (size_t groupIndex = 0; groupIndex < solveResult.targetBases.size(); ++groupIndex)
    {
        out << "target_group_" << groupIndex << "_base_mm: ";
        PrintVector(out, solveResult.targetBases[groupIndex]);
        out << "\n";
    }
    out << "Valid stations: " << observations.size() << "\n";
    out << "Removed stations: " << removedStationCount << "\n\n";

    out << "T_flange_camera maps camera coordinates to flange coordinates.\n";
    out << "camera_center_in_flange_mm: ";
    PrintVector(out, translation);
    out << "\n";
    out << "quaternion_wxyz: " << quaternion.w() << ' ' << quaternion.x() << ' '
        << quaternion.y() << ' ' << quaternion.z() << "\n";
    out << "rpy_xyz_deg: " << rpy[0] << ' ' << rpy[1] << ' ' << rpy[2] << "\n";
    out << "rotation_matrix:\n";
    for (int row = 0; row < 3; ++row)
    {
        out << "  " << rotation(row, 0) << ' ' << rotation(row, 1) << ' ' << rotation(row, 2) << "\n";
    }
    out << "ceres_brief_report: " << solveResult.briefReport << "\n";
    out << "ceres_initial_cost: " << solveResult.initialCost << "\n";
    out << "ceres_final_cost: " << solveResult.finalCost << "\n";

    out << "\nPer-station corner error in robot base (mm):\n";
    out << "timestamp group_index group_name camera_x camera_y camera_z predicted_x predicted_y predicted_z "
        << "error_x error_y error_z error_norm candidate_index planes candidate_score centroid_distance_ok\n";
    for (size_t i = 0; i < observations.size() && i < solveResult.errors.size(); ++i)
    {
        const CandidateObservation& observation = observations[i];
        out << observation.timestamp << ' '
            << observation.groupIndex << ' '
            << observation.groupName << ' ';
        PrintVector(out, observation.cameraPoint);
        out << ' ';
        PrintVector(out, solveResult.basePredictions[i]);
        out << ' ';
        PrintVector(out, solveResult.errors[i]);
        out << ' ' << solveResult.errors[i].norm() << ' '
            << observation.candidateIndex << ' '
            << observation.planeIndices[0] << ','
            << observation.planeIndices[1] << ','
            << observation.planeIndices[2] << ' '
            << observation.candidateScore << ' '
            << (observation.centroidDistanceOk ? "yes" : "no") << "\n";
    }

    out << "\nmean_error_mm: " << solveResult.meanErrorMm << "\n";
    out << "rms_error_mm: " << solveResult.rmsErrorMm << "\n";
    out << "max_error_mm: " << solveResult.maxErrorMm << "\n";
}
bool ExportBaseClouds(
    const std::vector<CandidateObservation>& observations,
    const fs::path& baseDir,
    const Eigen::Isometry3d& flangeFromCamera)
{
    std::error_code ec;
    fs::create_directories(baseDir, ec);
    if (ec)
    {
        std::cerr << "Cannot create base cloud directory " << baseDir.string()
            << ": " << ec.message() << "\n";
        return false;
    }

    bool allOk = true;
    for (const CandidateObservation& observation : observations)
    {
        const fs::path outputPath = baseDir / observation.groupName / observation.cloudPath.filename();
        fs::create_directories(outputPath.parent_path(), ec);
        const Eigen::Isometry3d baseFromCamera = observation.baseFromFlange * flangeFromCamera;
        if (!handeye::PointCloudIo::TransformPcdFileToBase(
            observation.cloudPath, outputPath, baseFromCamera, &std::cerr))
        {
            allOk = false;
        }
    }
    return allOk;
}

bool ExportFeatureBaseClouds(
    const std::vector<CandidateObservation>& observations,
    const fs::path& featureBaseDir,
    const Eigen::Isometry3d& flangeFromCamera)
{
    std::error_code ec;
    fs::create_directories(featureBaseDir, ec);
    if (ec)
    {
        std::cerr << "Cannot create feature base cloud directory " << featureBaseDir.string()
            << ": " << ec.message() << "\n";
        return false;
    }

    bool allOk = true;
    for (const CandidateObservation& observation : observations)
    {
        if (observation.featurePath.empty() || !fs::exists(observation.featurePath))
        {
            allOk = false;
            continue;
        }

        const fs::path outputPath = featureBaseDir / observation.groupName / observation.cloudPath.filename();
        fs::create_directories(outputPath.parent_path(), ec);
        const Eigen::Isometry3d baseFromCamera = observation.baseFromFlange * flangeFromCamera;
        if (!handeye::PointCloudIo::TransformPcdFileToBase(
            observation.featurePath, outputPath, baseFromCamera, &std::cerr))
        {
            allOk = false;
        }
    }
    return allOk;
}
} // namespace

namespace handeye
{

HandEyeAutoCalibrator::HandEyeAutoCalibrator(AutoCalibrateParams params)
    : params_(std::move(params))
{
}

AutoCalibrateResult HandEyeAutoCalibrator::Run() const
{
    return Run(params_);
}

AutoCalibrateResult HandEyeAutoCalibrator::Run(const AutoCalibrateParams& params) const
{
    AutoCalibrateResult result;
    try
    {
        const fs::path absoluteDataDir = fs::absolute(params.dataDir);
        if (params.dataDir.empty())
        {
            result.code = -1;
            result.message = "Data directory is empty";
            return result;
        }
        if (!fs::exists(absoluteDataDir) || !fs::is_directory(absoluteDataDir))
        {
            result.code = -1;
            result.message = "Data directory does not exist: " + absoluteDataDir.string();
            return result;
        }

        std::string groupLoadError;
        std::vector<StationGroup> groups = BuildStationGroups(
            params,
            absoluteDataDir,
            result.featureExtraction,
            groupLoadError);
        if (!groupLoadError.empty())
        {
            result.code = result.featureExtraction.code != 0 ? result.featureExtraction.code : -2;
            result.message = groupLoadError;
            return result;
        }
        if (groups.empty())
        {
            result.code = -2;
            result.message = "No station groups found in: " + absoluteDataDir.string();
            return result;
        }

        for (const StationGroup& group : groups)
        {
            if (static_cast<int>(group.stations.size()) < params.minStationCount)
            {
                result.code = -2;
                result.message = "At least " + std::to_string(params.minStationCount)
                    + " valid stations are required per group; " + group.name
                    + " has " + std::to_string(group.stations.size());
                return result;
            }
        }

        std::vector<CandidateObservation> observations =
            SelectGloballyConsistentObservations(groups, params);
        if (!HasMinimumObservationsPerGroup(observations, params))
        {
            result.code = -3;
            result.message = "No globally consistent grouped corner candidate set found";
            return result;
        }

        std::optional<AutoSolveResult> solveResult = SolveAutoCalibration(observations, params);
        if (!solveResult)
        {
            result.code = -4;
            result.message = "Auto calibration optimization failed";
            return result;
        }

        int removedCount = 0;
        if (params.maxStationErrorMm > 0.0001)
        {
            while (true)
            {
                const size_t groupCount = solveResult->targetBases.size();
                std::vector<int> groupCounts = CountObservationsPerGroup(observations, groupCount);
                size_t worstIndex = observations.size();
                double worstError = 0.0;
                for (size_t i = 0; i < solveResult->errors.size() && i < observations.size(); ++i)
                {
                    const int groupIndex = observations[i].groupIndex;
                    if (groupIndex < 0 || static_cast<size_t>(groupIndex) >= groupCounts.size())
                    {
                        continue;
                    }
                    if (groupCounts[static_cast<size_t>(groupIndex)] <= params.minStationCount)
                    {
                        continue;
                    }

                    const double errorNorm = solveResult->errors[i].norm();
                    if (errorNorm > worstError)
                    {
                        worstError = errorNorm;
                        worstIndex = i;
                    }
                }

                if (worstIndex >= observations.size() || worstError <= params.maxStationErrorMm)
                {
                    break;
                }

                observations.erase(observations.begin() + static_cast<long>(worstIndex));
                ++removedCount;

                std::optional<AutoSolveResult> recalibResult = SolveAutoCalibration(observations, params);
                if (!recalibResult)
                {
                    result.code = -5;
                    result.message = "Re-calibration failed after removing outlier stations";
                    return result;
                }
                solveResult = std::move(recalibResult);
            }
        }

        PopulatePublicResult(observations, *solveResult, removedCount, result);

        const fs::path outputDir = params.outputDir.empty() ? absoluteDataDir : params.outputDir;
        if (params.writeOutputs)
        {
            fs::create_directories(outputDir);

            const fs::path reportPath = outputDir / "auto_handeye_calibration_result.txt";
            {
                std::ofstream report(reportPath);
                if (report)
                {
                    WriteAutoReport(report, absoluteDataDir, observations, *solveResult, removedCount);
                }

                std::ostringstream reportText;
                WriteAutoReport(reportText, absoluteDataDir, observations, *solveResult, removedCount);
                result.reportText = reportText.str();
            }
            result.reportPath = reportPath;

            const fs::path matrixPath = outputDir / "handeye_matrix.txt";
            WriteHandEyeMatrix(solveResult->flangeFromCamera, matrixPath);
            result.matrixPath = matrixPath;
        }

        if (params.exportBaseClouds)
        {
            const fs::path baseDir = params.baseCloudOutputDir.empty()
                ? absoluteDataDir / "base"
                : params.baseCloudOutputDir;
            ExportBaseClouds(observations, baseDir, solveResult->flangeFromCamera);
        }

        if (params.exportFeatureBaseClouds)
        {
            const fs::path featureBaseDir = params.featureBaseOutputDir.empty()
                ? absoluteDataDir / "featureBase"
                : params.featureBaseOutputDir;
            ExportFeatureBaseClouds(observations, featureBaseDir, solveResult->flangeFromCamera);
        }

        std::ostringstream message;
        message << "Auto calibration completed: rms=" << solveResult->rmsErrorMm
            << " mm, groups=" << solveResult->targetBases.size();
        for (size_t groupIndex = 0; groupIndex < solveResult->targetBases.size(); ++groupIndex)
        {
            const Eigen::Vector3d& targetBase = solveResult->targetBases[groupIndex];
            message << " targetBase[" << groupIndex << "]=("
                << targetBase.x() << ", " << targetBase.y() << ", " << targetBase.z() << ")";
        }
        if (removedCount > 0)
        {
            message << " (removed " << removedCount << " outliers)";
        }

        result.code = 0;
        result.message = message.str();
    }
    catch (const std::exception& ex)
    {
        result.code = -99;
        result.message = std::string("HandEyeAutoCalibrator exception: ") + ex.what();
    }
    return result;
}
} // namespace handeye
