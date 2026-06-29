#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <array>
#include <string>
#include <vector>

class HandEyePlaneOptimizer
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
        std::array<PlaneObservation, 3> planes;
    };

    struct Options
    {
        double maxCornerErrorForPlaneStageMm = 10.0;
        double huberLossMm = 1.0;
        int maxNumIterations = 100;
        bool minimizerProgressToStdout = false;
    };

    struct Result
    {
        bool attempted = false;
        bool success = false;
        Eigen::Isometry3d flangeFromCamera = Eigen::Isometry3d::Identity();
        size_t residualCount = 0;
        double initialCost = 0.0;
        double finalCost = 0.0;
        double meanAbsDistanceMm = 0.0;
        double rmsDistanceMm = 0.0;
        double maxAbsDistanceMm = 0.0;
        std::string briefReport;
        std::string fullReport;
    };

    explicit HandEyePlaneOptimizer(Options options = Options());

    Result Optimize(
        const std::vector<StationObservation>& stations,
        const Eigen::Isometry3d& initialFlangeFromCamera,
        double cornerMaxErrorMm) const;

private:
    Options options_;
};
