#pragma once

#include "app_config.h"
#include "pose_io.h"

#include <Eigen/Dense>

#include <memory>
#include <string>
#include <vector>

namespace ModuleDevice
{
class DeviceManager;
class ScannerDevice;
class MotionDevice;
}

namespace handeye
{
struct DeviceCapture
{
    std::vector<Eigen::Vector3f> points;
    PoseIo::Pose flangePose;
    PoseIo::Quaternion flangeQuaternion;
    PoseIo::Pose tcpPose;
};

class __declspec(dllexport) DeviceSession
{
public:
    explicit DeviceSession(const AppConfig& config);
    ~DeviceSession();

    DeviceSession(const DeviceSession&) = delete;
    DeviceSession& operator=(const DeviceSession&) = delete;

    DeviceCapture captureOnce(const std::string& scanName, const std::string& scanTag = "handeye_calib");
    PoseIo::Pose readWelderPose() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
