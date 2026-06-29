#pragma once

#include "app_config.h"
#include "transform_utils.h"

#include <array>
#include <memory>

namespace handeye
{
struct RobotSnapshot
{
    Pose tcpPose;
    Pose tcpOffset;
    Pose ucsPose;
    Pose flangePose;
    Quaternion flangeQuaternion;
    std::array<double, 6> joints{};
};

class __declspec(dllexport) Robot
{
public:
    explicit Robot(const AppConfig& config);
    ~Robot();

    Robot(const Robot&) = delete;
    Robot& operator=(const Robot&) = delete;

    RobotSnapshot ReadSnapshot(const std::string& toolName = "TCP") const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace handeye
