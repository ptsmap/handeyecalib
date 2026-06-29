#pragma once

#include "app_config.h"

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <array>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

// ============================================================================
// Common status code and status message, returned by every public API function.
// ============================================================================

namespace handeye
{

/// Unified operation status used across all pipeline stages.
struct OperationStatus
{
    /// 0 = success, non-zero = failure (negative for internal errors,
    /// positive for user-correctable issues such as missing data).
    int code = 0;
    /// Human-readable status message, always populated.
    std::string message;
};

// ============================================================================
// Module 1 – Scan collection & save
// ============================================================================


/// Parameters for saving a single scan station (point cloud + robot pose).
struct ScanSaveParams
{
    /// Timestamp string, e.g. "20260612_143022".
    std::string timestamp;
    /// Path where the .pcd and .pose files will be written.
    std::filesystem::path outputDir;
    /// Filename stem for the output files (without extension).
    /// If empty, "scan_<timestamp>" is used.
    std::string fileStem;

    // ---- Point cloud data (raw camera output) ----
    /// Opaque pointer to the point-cloud buffer (must match resultType).
    const void* cloudData = nullptr;
    /// Number of points in cloudData.
    unsigned int cloudPointCount = 0;
    /// Format of cloudData (see VZNL_Types.h / EVzResultDataType).
    int cloudDataType = 0;

    // ---- Robot pose ----
    Eigen::Vector3d flangeTranslation = Eigen::Vector3d::Zero();  // mm
    Eigen::Quaterniond flangeQuaternion = Eigen::Quaterniond::Identity();
    Eigen::Vector3d flangeRotation = Eigen::Vector3d::Zero();//rxryrz
    Eigen::Vector3d tcpTranslation = Eigen::Vector3d::Zero();     // mm

    /// Optional: if non-empty, also writes a base-coordinate PCD using
    /// the provided hand-eye matrix (reads handeye_matrix.txt from this path).
    std::filesystem::path baseCloudOutputDir;

    /// Optional: hand-eye matrix for base-cloud export.
    /// If identity and baseCloudOutputDir is set, the matrix is read from
    /// handeye_matrix.txt next to the executable.
    Eigen::Isometry3d flangeFromCamera = Eigen::Isometry3d::Identity();
};

/// Result of saving a single scan station.
struct ScanSaveResult : OperationStatus
{
    /// Full path to the saved .pcd file.
    std::filesystem::path cloudPath;
    /// Full path to the saved .pose file.
    std::filesystem::path posePath;
    /// Full path to the base-coordinate .pcd file (empty if not written).
    std::filesystem::path baseCloudPath;
    /// Number of points saved.
    unsigned int savedPointCount = 0;
};

/// Save one captured scan station (PCD + robot pose + append index).
/// On success the result contains the paths of the written files.
ScanSaveResult __declspec(dllexport) SaveScanStation(const ScanSaveParams& params);

/// Parameters for a batch-scan collection session.
struct ScanCollectionParams
{
    /// Directory where all scan data will be written.
    std::filesystem::path outputDir;
    /// Optional base-cloud output directory.
    std::filesystem::path baseCloudOutputDir;
    /// Optional hand-eye matrix for base-cloud export.
    Eigen::Isometry3d flangeFromCamera = Eigen::Isometry3d::Identity();
};

/// Result of a scan collection session (aggregate over multiple scans).
struct ScanCollectionResult : OperationStatus
{
    /// Number of scan stations saved in this session.
    int stationCount = 0;
    /// Full paths of all saved .pcd files.
    std::vector<std::filesystem::path> cloudPaths;
    /// Full paths of all saved .pose files.
    std::vector<std::filesystem::path> posePaths;
    /// Full path of the robot_records.pose index file.
    std::filesystem::path indexFilePath;
};

/// Start or resume a scan-collection session, returning the index path and
/// output directory so the caller can subsequently call SaveScanStation.
ScanCollectionResult __declspec(dllexport) BeginScanCollection(const ScanCollectionParams& params);

// ============================================================================
// Module 2 – Feature extraction (trihedral plane / corner) & save
// ============================================================================

/// A single plane observation extracted from a point cloud.
struct PlaneInfo
{
    /// Unit normal of the plane (pointing away from sensor origin).
    Eigen::Vector3d normal = Eigen::Vector3d::UnitX();
    /// Signed distance from origin along the normal (mm).
    double d = 0.0;
    /// Centroid of the inlier points (mm).
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    /// Number of points supporting this plane.
    size_t pointCount = 0;
    /// Width of the bounding rectangle on the plane (mm).
    double extentX = 0.0;
    /// Height of the bounding rectangle on the plane (mm).
    double extentY = 0.0;
};

/// A trihedral corner candidate formed by three mutually orthogonal planes.
struct TrihedralCornerInfo
{
    /// Indices of the three planes in the station's plane list.
    std::array<int, 3> planeIndices{};
    /// Intersection point of the three planes in camera coordinates (mm).
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    /// Quality score (lower is better): sum of angle residual (deg) +
    /// 0.02 * centroid distance residual (mm).
    double score = std::numeric_limits<double>::max();
    /// Sum of orthogonal angle residuals (deg).
    double angleResidualDeg = 0.0;
    /// Sum of centroid-pair distance residuals (mm).
    double centroidPairResidualMm = 0.0;
    /// Sum of centroid-to-corner distance residuals (mm).
    double centroidCornerResidualMm = 0.0;
    /// Whether the expected centroid-distance constraint is satisfied.
    bool centroidDistanceOk = false;
};

/// Extracted features for a single scan station.
struct StationFeatureInfo
{
    /// Timestamp / station identifier.
    std::string timestamp;
    /// Path to the source point cloud file.
    std::filesystem::path cloudPath;
    /// Robot base-from-flange transform at scan time.
    Eigen::Isometry3d baseFromFlange = Eigen::Isometry3d::Identity();
    /// Number of planes detected by RANSAC.
    int detectedPlaneCount = 0;
    /// Number of planes after merge + orthogonal-mate filtering.
    int mergedPlaneCount = 0;
    /// All detected planes (before merge).
    std::vector<PlaneInfo> rawPlanes;
    /// Planes after merging similar planes.
    std::vector<PlaneInfo> mergedPlanes;
    /// Ranked trihedral corner candidates (best first).
    std::vector<TrihedralCornerInfo> cornerCandidates;
};

/// Parameters for the feature-extraction stage.
struct FeatureExtractionParams
{
    /// Directory containing .pcd and .pose files.
    std::filesystem::path dataDir;

    /// ---- RANSAC plane detection ----
    /// Minimum points for a plane inlier set.
    int ransacMinPoints = 100;
    /// RANSAC epsilon (mm).
    double ransacEpsilon = 1.0;
    /// RANSAC bitmap epsilon (mm).
    double ransacBitmapEpsilon = 2.0;
    /// Maximum allowed angle between a plane normal and ground truth (deg).
    double planeAngleTolerance = 25.0;

    /// ---- Plane size filter ----
    double planeMinExtentMm = 65.0;
    double planeMaxExtentMm = 150.0;

    /// ---- Merge & orthogonal filter ----
    /// Maximum angle between normals of planes to be merged (deg).
    double mergeAngleDeg = 5.0;
    /// Maximum distance between parallel planes to be merged (mm).
    double mergeDistanceMm = 12.0;
    /// Tolerance for orthogonal check (deg).
    double orthogonalToleranceDeg = 8.0;
    /// Maximum centroid distance for orthogonal-mate check (mm).
    double orthogonalMateMaxDistanceMm = 200.0;

    /// ---- Corner candidate ----
    /// Expected distance between plane centroids of the trihedral (mm).
    double expectedCentroidDistanceMm = 100.0;
    /// Tolerance for expected-centroid-distance check (mm).
    double centroidDistanceToleranceMm = 15.0;
    /// Maximum number of corner candidates kept per station.
    size_t maxCandidatesPerStation = 2;

    /// ---- Output ----
    /// If true, export per-station feature PCD files.
    bool exportFeatureCloud = true;
    /// Directory for feature PCD output (default: <dataDir>/../feature).
    std::filesystem::path featureOutputDir;
};

/// Result of the feature-extraction stage.
struct FeatureExtractionResult : OperationStatus
{
    /// Number of stations that were successfully processed.
    int stationCount = 0;
    /// Number of stations skipped (fewer than 3 planes / no valid corner).
    int skippedStationCount = 0;
    /// Per-station extracted features (only stations with valid corners).
    std::vector<StationFeatureInfo> stations;
    /// Path to the TCP file that was used (empty if none).
    std::filesystem::path tcpFilePath;
    /// Target TCP positions read from the TCP file.
    std::vector<Eigen::Vector3d> targetBases;
};

/// Extract trihedral-plane features and corner candidates from scan data.
/// Reads .pcd + .pose from dataDir, runs RANSAC plane detection, merge,
/// orthogonal filtering, and corner intersection. Optionally exports
/// feature PCD files.
FeatureExtractionResult __declspec(dllexport) RunFeatureExtraction(const FeatureExtractionParams& params);

/// Extract workpiece trihedral inner-corner features. Candidate corners are
/// formed by three mutually orthogonal planes and ranked by the total support
/// point count of those three planes, from large to small.
FeatureExtractionResult __declspec(dllexport) RunWorkpieceCornerFeatureExtraction(const FeatureExtractionParams& params);

// ============================================================================
// Module 2b – Single PCD corner extraction (lightweight, no pose files needed)
// ============================================================================

/// A single extracted trihedral corner from one PCD file.
struct ExtractedCornerInfo
{
    /// Intersection point in camera coordinates (mm).
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    /// Candidate quality score (lower is better).
    double score = std::numeric_limits<double>::max();
    /// Whether the centroid-distance constraint is satisfied.
    bool centroidDistanceOk = false;
    /// Indices of the three planes forming this corner.
    std::array<int, 3> planeIndices{};
    /// Normals of the three planes forming this corner (camera coords).
    std::array<Eigen::Vector3d, 3> planeNormals{};
    /// Signed distance of the three planes from origin (mm).
    std::array<double, 3> planeDs{};
    /// True if the corner was generated from merged planes; false for raw planes.
    bool usesMergedPlanes = false;
};

/// Result of extracting trihedral corner points from a single PCD file.
struct ExtractCornerResult : OperationStatus
{
    /// Best corner intersection point in camera coordinates (mm).
    Eigen::Vector3d cornerPoint = Eigen::Vector3d::Zero();
    /// Candidate quality score (lower is better).
    double score = std::numeric_limits<double>::max();
    /// Whether the centroid-distance constraint is satisfied.
    bool centroidDistanceOk = false;
    /// Number of planes detected by RANSAC.
    int detectedPlaneCount = 0;
    /// Number of planes after merge + orthogonal filtering.
    int mergedPlaneCount = 0;
    /// Normals of the three planes forming the best corner (camera coords).
    std::array<Eigen::Vector3d, 3> planeNormals;
    /// Signed distance of the three planes from origin (mm).
    std::array<double, 3> planeDs;
    /// Ranked, spatially de-duplicated corner points (best first).
    std::vector<ExtractedCornerInfo> corners;
};

/// Extract trihedral corner points from a single PCD file.
/// For backward compatibility, ExtractCornerResult::cornerPoint / score keep
/// the best extracted corner, while ExtractCornerResult::corners contains all
/// ranked, spatially de-duplicated corner points.
/// Runs RANSAC plane detection, plane merging, orthogonal filtering,
/// and corner intersection. No pose file is required.
///
/// @param pcdPath  Path to the input .pcd file (XYZ format).
/// @param params   Optional feature extraction parameters (uses defaults if empty).
ExtractCornerResult __declspec(dllexport) ExtractCornerFromPcd(
    const std::filesystem::path& pcdPath,
    const FeatureExtractionParams& params = FeatureExtractionParams{});

// ============================================================================
// Module 3 – Calibrate mode (single / multi-target trihedral calibration)
// ============================================================================

/// A matched observation used for calibration (one station → one corner).
struct CalibrationObservation
{
    std::string timestamp;
    std::filesystem::path cloudPath;
    /// Robot base-from-flange transform.
    Eigen::Isometry3d baseFromFlange = Eigen::Isometry3d::Identity();
    /// Which target TCP index this station is observing (0-based).
    int targetIndex = 0;
    /// Intersection corner point in camera coordinates (mm).
    Eigen::Vector3d cameraPoint = Eigen::Vector3d::Zero();
    /// Indices of the three planes forming this corner.
    std::array<int, 3> planeIndices{};
    /// Candidate score from feature extraction.
    double candidateScore = 0.0;
    /// Whether the expected centroid-distance constraint is satisfied.
    bool centroidDistanceOk = false;
};

/// Per-station error details.
struct StationCornerError
{
    std::string timestamp;
    Eigen::Vector3d cameraPoint = Eigen::Vector3d::Zero();
    Eigen::Vector3d predictedBase = Eigen::Vector3d::Zero();
    Eigen::Vector3d error = Eigen::Vector3d::Zero();
    double errorNorm = 0.0;
    int targetIndex = 0;
    std::array<int, 3> planeIndices{};
    double candidateScore = 0.0;
    bool centroidDistanceOk = false;
};

/// Non-linear optimizer stage result.
struct NonlinearOptimizerInfo
{
    bool success = false;
    double initialCost = 0.0;
    double finalCost = 0.0;
    double meanErrorMm = 0.0;
    double rmsErrorMm = 0.0;
    double maxErrorMm = 0.0;
    std::string briefReport;
};

/// Trihedral bundle-adjustment stage result.
struct TrihedralBundleInfo
{
    bool attempted = false;
    bool success = false;
    size_t stationCount = 0;
    size_t planeResidualCount = 0;
    size_t cornerResidualCount = 0;
    double initialCost = 0.0;
    double finalCost = 0.0;
    double planeMeanAbsDistanceMm = 0.0;
    double planeRmsDistanceMm = 0.0;
    double planeMaxAbsDistanceMm = 0.0;
    double cornerMeanErrorMm = 0.0;
    double cornerRmsErrorMm = 0.0;
    double cornerMaxErrorMm = 0.0;
    Eigen::Vector3d cornerBase = Eigen::Vector3d::Zero();
    bool cornerBaseAvailable = false;
    Eigen::Vector3d targetAlignmentDeltaMm = Eigen::Vector3d::Zero();
    double targetAlignmentRotationDeltaDeg = 0.0;
    bool targetPoseOptimized = false;
    std::string briefReport;
};

/// Parameters for the calibrate stage.
struct CalibrateParams
{
    /// Directory containing .pcd and .pose files.
    std::filesystem::path dataDir;

    /// ---- Pre-extracted features (optional) ----
    /// If non-empty, skip feature extraction and use these stations directly.
    std::vector<StationFeatureInfo> stations;

    /// ---- Feature extraction parameters (used if stations is empty) ----
    FeatureExtractionParams featureParams;

    /// ---- Target TCP(s) in robot base coordinates (mm) ----
    /// If empty, read from <dataDir>/corner.xyz.
    std::vector<Eigen::Vector3d> targetBases;

    /// ---- Nonlinear optimizer ----
    double cornerHuberLossMm = 1.0;
    int cornerMaxIterations = 1000;
    bool cornerMinimizerProgressToStdout = false;

    /// ---- Trihedral bundle adjustment ----
    /// Set to false to skip trihedral bundle optimization.
    bool enableTrihedralOptimization = false;
    /// Skip trihedral if the initial corner RMS error exceeds this (mm).
    double maxCornerErrorForTrihedralMm = 5.0;
    double trihedralPlaneHuberLossMm = 0.5;
    double trihedralTargetAlignmentHuberLossMm = 1.0;
    double trihedralTargetAlignmentSigmaMm = 1.0;
    double trihedralMinMatchedNormalDot = 0.94;
    int trihedralMaxIterations = 500;
    bool trihedralMinimizerProgressToStdout = false;

    /// ---- Output ----
    /// Directory where handeye_matrix.txt and report are written.
    std::filesystem::path outputDir;
    /// If true, export all PCD files transformed to base coordinates.
    bool exportBaseClouds = true;
    /// Base coordinate cloud output directory (default: <dataDir>/../base).
    std::filesystem::path baseCloudOutputDir;
    /// If true, export feature PCD files in base coordinates.
    bool exportFeatureBaseClouds = true;
    /// Feature base cloud output directory (default: <dataDir>/../featureBase).
    std::filesystem::path featureBaseOutputDir;

    /// ---- Outlier rejection ----
    /// Maximum per-station corner error (mm). Stations exceeding this are
    /// removed and the calibration is re-run. Set to 0 to disable.
    double maxStationErrorMm = 10.0;
};

/// Result of the calibrate stage.
struct CalibrateResult : OperationStatus
{
    /// Calibrated flange-from-camera transform.
    Eigen::Isometry3d flangeFromCamera = Eigen::Isometry3d::Identity();
    /// Camera center in flange coordinates (mm).
    Eigen::Vector3d cameraCenterInFlange = Eigen::Vector3d::Zero();
    /// Quaternion (w, x, y, z).
    Eigen::Quaterniond quaternion = Eigen::Quaterniond::Identity();
    /// Roll-pitch-yaw in degrees.
    Eigen::Vector3d rpyDeg = Eigen::Vector3d::Zero();
    /// 4x4 homogeneous matrix (row-major in the vector: 16 elements).
    std::array<double, 16> matrix4x4{};

    /// Number of valid observation stations used.
    int stationCount = 0;
    /// Number of target TCPs used.
    int targetCount = 1;
    /// Target TCP positions in base (mm).
    std::vector<Eigen::Vector3d> targetBases;

    /// Non-linear optimizer details.
    NonlinearOptimizerInfo nonlinearOptimizer;
    /// Trihedral bundle-adjustment details.
    TrihedralBundleInfo trihedralBundle;

    /// Corner error summary (from the final transform).
    double meanErrorMm = 0.0;
    double rmsErrorMm = 0.0;
    double maxErrorMm = 0.0;

    /// Per-station corner error details.
    std::vector<StationCornerError> stationErrors;
    /// Number of stations removed due to excessive error during iterative refinement.
    int removedStationCount = 0;

    /// Full calibration report text (same as written to file).
    std::string reportText;
    /// Path to the written report file.
    std::filesystem::path reportPath;
    /// Path to the written hand-eye matrix file.
    std::filesystem::path matrixPath;
};

/// Run the full calibration pipeline (feature extraction + candidate selection
/// + nonlinear optimization + optional trihedral bundle adjustment) and save
/// results.
///
/// If CalibrateParams::stations is non-empty, feature extraction is skipped
/// and the pre-extracted features are used directly.
CalibrateResult __declspec(dllexport) RunCalibrate(const CalibrateParams& params);


// ============================================================================
// Module 4 – Multiblock mode (multi-block plane calibration)
// ============================================================================


/// A single plane observation in the multiblock pipeline.
struct MultiBlockPlaneInfo
{
    /// Unit normal in camera coordinates.
    Eigen::Vector3d normal = Eigen::Vector3d::UnitX();
    /// Signed distance from origin (mm).
    double d = 0.0;
    /// Centroid of inlier points (mm).
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    /// Number of supporting points.
    size_t pointCount = 0;
    /// Bounding-rectangle extent X (mm).
    double extentX = 0.0;
    /// Bounding-rectangle extent Y (mm).
    double extentY = 0.0;
};

/// Extracted multiblock station info.
struct MultiBlockStationInfo
{
    std::string timestamp;
    std::filesystem::path cloudPath;
    Eigen::Isometry3d baseFromFlange = Eigen::Isometry3d::Identity();
    int detectedPlaneCount = 0;
    int mergedPlaneCount = 0;
    std::vector<MultiBlockPlaneInfo> rawPlanes;
    std::vector<MultiBlockPlaneInfo> mergedPlanes;
};

/// Info for a matched two-face pair within one block.
struct MultiBlockPairMatchInfo
{
    std::array<int, 2> planeIndices{};
    double score = 0.0;
    double angleResidualDeg = 0.0;
    double centroidDistanceResidualMm = 0.0;
};

/// How a station's multiblock faces were matched across stations.
struct MultiBlockStationMatchInfo
{
    std::string timestamp;
    /// Plane indices assigned to each block/face: [block][face].
    std::array<std::array<int, 2>, 2> sourcePlaneIndices{};
    double score = 0.0;
    double minMatchedNormalDot = 0.0;
    /// "gicp", "reference", or "initial".
    std::string matchingMode;
    double gicpFitnessMm2 = 0.0;
};

/// Post-optimization per-face result.
struct MultiBlockFaceResultInfo
{
    int blockIndex = 0;
    int faceIndex = 0;
    Eigen::Vector3d blockNormal = Eigen::Vector3d::UnitX();
    Eigen::Vector3d normalBase = Eigen::Vector3d::UnitX();
    double dBase = 0.0;
    size_t observationCount = 0;
    size_t residualCount = 0;
    double rmsDistanceMm = 0.0;
    double maxAbsDistanceMm = 0.0;
};

/// Parameters for the multiblock calibration stage.
struct MultiBlockCalibrateParams
{
    /// Directory containing .pcd and .pose files.
    std::filesystem::path dataDir;

    /// ---- Feature extraction parameters ----
    int ransacMinPoints = 100;
    double ransacEpsilon = 1.0;
    double ransacBitmapEpsilon = 2.0;
    double planeAngleTolerance = 25.0;
    double planeMinExtentMm = 45.0;
    double planeMaxExtentMm = 180.0;
    double mergeAngleDeg = 5.0;
    double mergeDistanceMm = 12.0;
    double orthogonalToleranceDeg = 8.0;
    double orthogonalMateMaxDistanceMm = 200.0;
    double expectedCentroidDistanceMm = 100.0;
    double centroidDistanceToleranceMm = 15.0;
    size_t maxPairCandidatesPerStation = 80;
    size_t maxTwoBlockCandidatesPerStation = 160;

    /// ---- Cross-station matching ----
    /// Minimum absolute dot product between matched face normals in base frame.
    double minMatchedNormalDot = 0.70;

    /// ---- Path to an initial hand-eye matrix (optional) ----
    /// If provided and usable, used for cross-station plane matching.
    /// If empty or identity, adjacent-station GICP is used for matching.
    std::filesystem::path initialMatrixPath;

    /// ---- GICP matching parameters (used when no initial matrix) ----
    size_t gicpMinPoints = 80;
    double gicpVoxelLeafMm = 3.0;
    double gicpMaxCorrespondenceDistanceMm = 2.0;
    int gicpMaxIterations = 10000;
    double gicpMaxFitnessMm2 = 200000.0;
    size_t maxGicpCandidateTestsPerStation = 80;

    /// ---- Optimizer ----
    double planeHuberLossMm = 0.5;
    int maxIterations = 500;
    bool minimizerProgressToStdout = false;

    /// ---- Output ----
    std::filesystem::path outputDir;
};

/// Result of the multiblock calibration stage.
struct MultiBlockCalibrateResult : OperationStatus
{
    /// Calibrated flange-from-camera transform.
    Eigen::Isometry3d flangeFromCamera = Eigen::Isometry3d::Identity();
    Eigen::Vector3d cameraCenterInFlange = Eigen::Vector3d::Zero();
    Eigen::Quaterniond quaternion = Eigen::Quaterniond::Identity();
    Eigen::Vector3d rpyDeg = Eigen::Vector3d::Zero();
    std::array<double, 16> matrix4x4{};

    /// Initial hand-eye used for matching.
    Eigen::Isometry3d initialFlangeFromCamera = Eigen::Isometry3d::Identity();
    bool initialMatrixWasLoaded = false;
    /// Matching strategy used: "initial_handeye" or "adjacent_gicp".
    std::string matchingStrategy;

    /// Number of valid stations.
    int stationCount = 0;
    /// Per-station matching details.
    std::vector<MultiBlockStationMatchInfo> stationMatches;

    /// Optimizer details.
    bool optimizationSuccess = false;
    double initialCost = 0.0;
    double finalCost = 0.0;
    size_t residualCount = 0;
    int translationConstraintRank = 0;
    double planeMeanAbsDistanceMm = 0.0;
    double planeRmsDistanceMm = 0.0;
    double planeMaxAbsDistanceMm = 0.0;
    double normalMeanAngleDeg = 0.0;
    double normalMaxAngleDeg = 0.0;
    std::string optimizerBriefReport;

    /// Per-face results after optimization.
    std::vector<MultiBlockFaceResultInfo> faceResults;

    /// Full report text.
    std::string reportText;
    std::filesystem::path reportPath;
    std::filesystem::path matrixPath;
};

/// Run the multiblock calibration pipeline (plane extraction → cross-station
/// matching → plane-based optimization) and save results.
MultiBlockCalibrateResult __declspec(dllexport) RunMultiBlockCalibrate(const MultiBlockCalibrateParams& params);

// ============================================================================
// Legacy convenience wrappers (backward-compatible with existing main.cpp)
// ============================================================================

int __declspec(dllexport) RunHandEyeCalibration(
    const std::filesystem::path& dataDir,
    const std::filesystem::path& executableDir,
    const handeye::AppConfig& config = handeye::AppConfig{});

int __declspec(dllexport) RunHandEyeMultiBlockCalibration(
    const std::filesystem::path& dataDir,
    const std::filesystem::path& executableDir,
    const std::filesystem::path& initialMatrixPath = {});
}
