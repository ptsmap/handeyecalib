#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <array>
#include <string>
#include <vector>

class HandEyeTrihedralBundleOptimizer
{
public:
    struct PlaneObservation
    {
        Eigen::Vector3d normal = Eigen::Vector3d::UnitX();
        double d = 0.0;
        std::vector<Eigen::Vector3d> points;
    };

    struct StationObservation
    {
        std::string timestamp;
        Eigen::Isometry3d baseFromFlange = Eigen::Isometry3d::Identity();
        int targetIndex = 0;
        // Noisy station-level trihedral corner observation in camera coordinates.
        Eigen::Vector3d cornerCamera = Eigen::Vector3d::Zero();
        std::array<PlaneObservation, 3> planes;
    };

    struct Options
    {
        double maxCornerErrorForTrihedralStageMm = 5.0;
        double planeHuberLossMm = 0.5;
        double targetAlignmentHuberLossMm = 1.0;
        double targetAlignmentSigmaMm = 1.0;
        double minMatchedNormalDot = 0.94;
        int maxNumIterations = 500;
        bool minimizerProgressToStdout = true;
    };

    struct Result
    {
        bool attempted = false;
        bool success = false;
        Eigen::Isometry3d flangeFromCamera = Eigen::Isometry3d::Identity();
        Eigen::Isometry3d baseFromBlock = Eigen::Isometry3d::Identity();
        std::vector<Eigen::Isometry3d> baseFromBlocks;
        Eigen::Vector3d targetAlignmentTranslationDeltaMm = Eigen::Vector3d::Zero();
        double targetAlignmentRotationDeltaDeg = 0.0;
        bool targetPoseOptimized = false;
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
        std::string briefReport;
        std::string fullReport;
    };

    explicit HandEyeTrihedralBundleOptimizer(Options options = Options());

    Result Optimize(
        const std::vector<StationObservation>& stations,
        const Eigen::Vector3d& targetBase,
        const Eigen::Isometry3d& initialFlangeFromCamera,
        double cornerMaxErrorMm) const;

    Result Optimize(
        const std::vector<StationObservation>& stations,
        const std::vector<Eigen::Vector3d>& targetBases,
        const Eigen::Isometry3d& initialFlangeFromCamera,
        double cornerMaxErrorMm,
        bool optimizeTargetPoseInSecondStage) const;

private:
    Options options_;
};
