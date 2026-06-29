#pragma once

#include "handeye_calibrator.h"

#include <cstddef>
#include <filesystem>
#include <vector>

namespace handeye
{

/// Parameters for hand-eye calibration without a manually taught trihedral
/// corner position in robot base coordinates.
struct AutoCalibrateParams
{
    /// Directory containing .pcd and .pose files.
    std::filesystem::path dataDir;

    /// Optional pre-extracted features. If empty, RunFeatureExtraction() is
    /// called with featureParams. This is treated as one target group.
    std::vector<StationFeatureInfo> stations;

    /// Optional grouped pre-extracted features. Each inner vector is one
    /// physical calibration-block position and gets its own unknown targetBase.
    /// If non-empty, this takes precedence over stations.
    std::vector<std::vector<StationFeatureInfo>> stationGroups;

    /// Feature extraction parameters. dataDir is filled from AutoCalibrateParams
    /// when left empty.
    FeatureExtractionParams featureParams;

    /// Maximum number of ranked corner candidates considered per station.
    size_t maxCandidatesPerStation = 2;
    /// Beam width used while searching one corner candidate per station.
    size_t candidateBeamWidth = 256;
    /// Minimum number of stations required in each target group.
    int minStationCount = 3;

    /// Nonlinear optimizer settings.
    double cornerHuberLossMm = 1.0;
    int cornerMaxIterations = 1000;
    bool cornerMinimizerProgressToStdout = false;

    /// Weight for the geometric feature score during candidate selection.
    double candidateGeometryScoreWeight = 0.02;

    /// Maximum per-station corner error (mm). Stations exceeding this are
    /// removed and the calibration is re-run. Set to 0 to disable.
    double maxStationErrorMm = 10.0;

    /// Directory where handeye_matrix.txt and report are written.
    std::filesystem::path outputDir;
    /// If true, write handeye_matrix.txt and the auto-calibration report.
    bool writeOutputs = true;
    /// If true, export all PCD files transformed to base coordinates.
    bool exportBaseClouds = true;
    /// Base coordinate cloud output directory (default: <dataDir>/base).
    std::filesystem::path baseCloudOutputDir;
    /// If true, export feature PCD files transformed to base coordinates.
    bool exportFeatureBaseClouds = true;
    /// Feature base cloud output directory (default: <dataDir>/featureBase).
    std::filesystem::path featureBaseOutputDir;
};

/// Result of auto hand-eye calibration. The inherited targetBases contains one
/// algorithm-estimated fixed trihedral corner point per target group in robot
/// base coordinates.
struct AutoCalibrateResult : CalibrateResult
{
    /// Feature extraction result used by this run.
    FeatureExtractionResult featureExtraction;
    /// Selected station/corner observations used in the final solve.
    std::vector<CalibrationObservation> observations;
    /// Algorithm-estimated trihedral corner point for group 0 in robot base
    /// coordinates. Kept for single-group compatibility.
    Eigen::Vector3d estimatedTargetBase = Eigen::Vector3d::Zero();
    /// Algorithm-estimated trihedral corner points, one per group.
    std::vector<Eigen::Vector3d> estimatedTargetBases;
};

/// Hand-eye calibration algorithm that does not require manual teaching of the
/// trihedral corner in robot base coordinates.
class __declspec(dllexport) HandEyeAutoCalibrator
{
public:
    HandEyeAutoCalibrator() = default;
    explicit HandEyeAutoCalibrator(AutoCalibrateParams params);

    AutoCalibrateResult Run() const;
    AutoCalibrateResult Run(const AutoCalibrateParams& params) const;

private:
    AutoCalibrateParams params_;
};

} // namespace handeye
