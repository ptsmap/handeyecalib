#pragma once

#include "handeye_auto_calibrator.h"

#include <filesystem>
#include <vector>

namespace handeye
{

/// Parameters for calibrating from workpiece trihedral inner corners.
/// Each scan group observes one different workpiece inner corner whose
/// base-coordinate position is estimated as an independent unknown.
struct WorkpieceCornerCalibrateParams
{
    WorkpieceCornerCalibrateParams()
    {
        featureParams.planeMinExtentMm = 50.0;
        featureParams.planeMaxExtentMm = 2000.0;
        featureParams.orthogonalToleranceDeg = 15.0;
        featureParams.orthogonalMateMaxDistanceMm = 500.0;
        featureParams.expectedCentroidDistanceMm = 100.0;
        featureParams.centroidDistanceToleranceMm = 100000.0;
        featureParams.maxCandidatesPerStation = 1;
        featureParams.exportFeatureCloud = true;
    }

    /// Directory containing .pcd/.pose files, or grouped subdirectories.
    std::filesystem::path dataDir;

    /// Optional pre-extracted features. If non-empty, feature extraction is
    /// skipped and all stations are treated as observations of one workpiece
    /// corner group.
    std::vector<StationFeatureInfo> stations;

    /// Optional grouped pre-extracted features. Each inner vector is one
    /// different workpiece inner-corner point and gets its own unknown base
    /// coordinate during optimization.
    std::vector<std::vector<StationFeatureInfo>> stationGroups;

    /// Feature extraction parameters tuned for workpiece inner corners.
    FeatureExtractionParams featureParams;

    /// If true, group_* / pos* / position* subdirectories are treated as
    /// different unknown workpiece inner-corner points.
    bool useGroupedDataDirs = true;

    /// Maximum number of ranked trihedral inner-corner candidates kept for
    /// each scan station in every group.
    size_t maxCandidatesPerStation = 1;
    size_t candidateBeamWidth = 512;
    int minStationCount = 3;

    double cornerHuberLossMm = 1.0;
    int cornerMaxIterations = 1000;
    bool cornerMinimizerProgressToStdout = false;
    double candidateGeometryScoreWeight = 0.0;
    double maxStationErrorMm = 15.0;

    std::filesystem::path outputDir;
    bool writeOutputs = true;
    bool exportBaseClouds = true;
    std::filesystem::path baseCloudOutputDir;
};

struct WorkpieceCornerCalibrateResult : AutoCalibrateResult
{
    /// Estimated workpiece inner-corner point for group 0 in robot base
    /// coordinates. Kept for single-group compatibility.
    Eigen::Vector3d estimatedWorkpieceCornerBase = Eigen::Vector3d::Zero();
    /// Estimated workpiece inner-corner points, one per group.
    std::vector<Eigen::Vector3d> estimatedWorkpieceCornerBases;
};

class __declspec(dllexport) HandEyeWorkpieceCornerCalibrator
{
public:
    HandEyeWorkpieceCornerCalibrator() = default;
    explicit HandEyeWorkpieceCornerCalibrator(WorkpieceCornerCalibrateParams params);

    WorkpieceCornerCalibrateResult Run() const;
    WorkpieceCornerCalibrateResult Run(const WorkpieceCornerCalibrateParams& params) const;

private:
    WorkpieceCornerCalibrateParams params_;
};

} // namespace handeye
