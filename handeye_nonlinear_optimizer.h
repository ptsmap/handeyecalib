#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <string>
#include <vector>

class HandEyeNonlinearOptimizer
{
public:
    struct Observation
    {
        Eigen::Isometry3d baseFromFlange = Eigen::Isometry3d::Identity();
        Eigen::Vector3d cameraPoint = Eigen::Vector3d::Zero();
        Eigen::Vector3d targetBase = Eigen::Vector3d::Zero();
        double weight = 1.0;
    };

    struct Options
    {
        double huberLossMm = 3.0;
        int maxNumIterations = 100;
        bool minimizerProgressToStdout = false;
    };

    struct Result
    {
        bool success = false;
        Eigen::Isometry3d flangeFromCamera = Eigen::Isometry3d::Identity();
        std::vector<Eigen::Vector3d> basePredictions;
        std::vector<Eigen::Vector3d> errors;
        double initialCost = 0.0;
        double finalCost = 0.0;
        double meanError = 0.0;
        double rmsError = 0.0;
        double maxError = 0.0;
        std::string briefReport;
        std::string fullReport;
    };

    explicit HandEyeNonlinearOptimizer(Options options = Options());

    Result Optimize(
        const std::vector<Observation>& observations,
        const Eigen::Isometry3d& initialFlangeFromCamera) const;

private:
    Options options_;
};
