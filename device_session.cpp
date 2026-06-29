#include "device_session.h"

#include "Device.h"
#include "MotionDevice.h"
#include "ScannerDevice.h"
#include "transform_utils.h"

#include <QCoreApplication>
#include <QEventLoop>
#include <QObject>

#include <chrono>
#include <cmath>
#include <functional>
#include <stdexcept>

namespace handeye
{
namespace
{
constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;

std::string NormalizeRobotType(const std::string& type)
{
    const std::string lower = ToLower(type);
    if (lower == "huayan" || lower == "hans")
    {
        return "Hans";
    }
    if (lower == "duco")
    {
        return "Duco";
    }
    if (lower == "sim" || lower == "jmc")
    {
        return "SIM";
    }
    return type;
}

std::string NormalizeScannerType(const std::string& type)
{
    const std::string lower = ToLower(type);
    if (lower == "vznl" || lower == "vizum")
    {
        return "VzNL";
    }
    if (lower == "sim")
    {
        return "SIM";
    }
    return type;
}

PoseIo::Pose PoseFromStdArrayDeg(const std::array<double, 6>& poseRad)
{
    PoseIo::Pose pose;
    pose.x = poseRad[0];
    pose.y = poseRad[1];
    pose.z = poseRad[2];
    pose.rx = poseRad[3] * kRadToDeg;
    pose.ry = poseRad[4] * kRadToDeg;
    pose.rz = poseRad[5] * kRadToDeg;
    return pose;
}
}

struct DeviceSession::Impl
{
    explicit Impl(const AppConfig& cfg)
        : config(cfg)
    {
    }

    AppConfig config;
    std::unique_ptr<QCoreApplication> appGuard;
    std::unique_ptr<ModuleDevice::DeviceManager> deviceManager;

    ModuleDevice::Device* realDevice() const;
    ModuleDevice::ScannerDevice* scanner() const;
    ModuleDevice::MotionDevice* motion() const;
    void processEventsUntil(const std::function<bool()>& done) const;
};

ModuleDevice::Device* DeviceSession::Impl::realDevice() const
{
    return deviceManager ? deviceManager->realDevice() : nullptr;
}

ModuleDevice::ScannerDevice* DeviceSession::Impl::scanner() const
{
    return realDevice() ? realDevice()->scannerDevice() : nullptr;
}

ModuleDevice::MotionDevice* DeviceSession::Impl::motion() const
{
    return realDevice() ? realDevice()->motionDevice() : nullptr;
}

void DeviceSession::Impl::processEventsUntil(const std::function<bool()>& done) const
{
    while (!done())
    {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 50);
    }
}

DeviceSession::DeviceSession(const AppConfig& config)
    : impl_(std::make_unique<Impl>(config))
{
    if (QCoreApplication::instance() == nullptr)
    {
        int argc = 1;
        char appName[] = "handeye_collector";
        char* argv[] = {appName, nullptr};
        impl_->appGuard = std::make_unique<QCoreApplication>(argc, argv);
    }

    impl_->deviceManager = std::make_unique<ModuleDevice::DeviceManager>();

    const DeviceProfile& robot = impl_->config.robotProfile();
    const DeviceProfile& camera = impl_->config.cameraProfile();
    if (!impl_->deviceManager->setRealScannerType(NormalizeScannerType(camera.type)))
    {
        throw std::runtime_error("Unsupported scanner type: " + camera.type);
    }
    if (!impl_->deviceManager->setRealMotionType(NormalizeRobotType(robot.type)))
    {
        throw std::runtime_error("Unsupported robot type: " + robot.type);
    }
    
    impl_->deviceManager->disableSimulateDevice();
    impl_->deviceManager->realDevice()->motionDevice()->setEnable(true);

    if (!robot.ip.empty() && impl_->realDevice())
    {
        impl_->realDevice()->setMotionDeviceIp(robot.ip);
    }

    const int openCode = impl_->deviceManager->open();
    if (openCode != 0)
    {
        throw std::runtime_error("DeviceManager open failed: " + std::to_string(openCode));
    }

    if (ModuleDevice::ScannerDevice* scanner = impl_->scanner())
    {
        if (const int rc = scanner->setSwingAngleRangeAndSpeed(camera.swingStart, camera.swingEnd, camera.swingSpeed))
        {
            throw std::runtime_error("setSwingAngleRangeAndSpeed failed: " + std::to_string(rc));
        }
        if (const int rc = scanner->setExposureTime(static_cast<int>(camera.exposureTime)))
        {
            throw std::runtime_error("setExposureTime failed: " + std::to_string(rc));
        }
        if (const int rc = scanner->setGain(camera.gain))
        {
            throw std::runtime_error("setGain failed: " + std::to_string(rc));
        }
        if (const int rc = scanner->setFrameRate(static_cast<int>(camera.frameRate)))
        {
            throw std::runtime_error("setFrameRate failed: " + std::to_string(rc));
        }
    }
}

DeviceSession::~DeviceSession()
{
    if (impl_ && impl_->deviceManager)
    {
        impl_->deviceManager->close();
    }
}

DeviceCapture DeviceSession::captureOnce(const std::string& scanName, const std::string& scanTag)
{
    ModuleDevice::ScannerDevice* scanner = impl_->scanner();
    ModuleDevice::MotionDevice* motion = impl_->motion();
    if (!scanner || !motion)
    {
        throw std::runtime_error("Scanner or motion device unavailable.");
    }

    bool finished = false;
    bool success = false;
    ModuleDevice::ScannerDevice::ScanData scanData;
    QMetaObject::Connection connection = QObject::connect(
        scanner,
        &ModuleDevice::ScannerDevice::scanStopped,
        [&finished, &success, &scanData](const QString&, bool ok, const ModuleDevice::ScannerDevice::ScanData& data) {
            success = ok;
            scanData = data;
            finished = true;
        });

    const int startCode = scanner->startScan(scanName, Eigen::Matrix4d::Identity(), scanTag);
    if (startCode != 0)
    {
        QObject::disconnect(connection);
        throw std::runtime_error("startScan failed: " + std::to_string(startCode));
    }

    impl_->processEventsUntil([&finished]() { return finished; });
    QObject::disconnect(connection);

    if (!success)
    {
        throw std::runtime_error("Scanner reported failed capture.");
    }

    std::array<double, 6> flangePoseRad{};
    motion->setCurrentTcp("", "");
    motion->queryTcpPose("", flangePoseRad, "");
    motion->setCurrentTcp("", "");
    if (const int rc = motion->queryTcpPose("", flangePoseRad, ""))
    {
        throw std::runtime_error("queryTcpPose(flange) failed: " + std::to_string(rc));
    }

    DeviceCapture capture;
    capture.points.resize(static_cast<size_t>(scanData.pcd.rows()));
    Eigen::Vector3f point;
    for (int row = 0; row < scanData.pcd.rows(); ++row)
    {
        point.x() = static_cast<float>(scanData.pcd(row, 0));
        point.y() = static_cast<float>(scanData.pcd(row, 1));
        point.z() = static_cast<float>(scanData.pcd(row, 2));
        capture.points[static_cast<size_t>(row)] = point;
    }

    capture.flangePose = PoseFromStdArrayDeg(flangePoseRad);
    capture.flangeQuaternion = QuaternionFromRotation(RotationFromRpy(
        capture.flangePose.rx,
        capture.flangePose.ry,
        capture.flangePose.rz));
    return capture;
}

PoseIo::Pose DeviceSession::readWelderPose() const
{
    ModuleDevice::MotionDevice* motion = impl_->motion();
    if (!motion)
    {
        throw std::runtime_error("Motion device unavailable.");
    }

    const DeviceProfile& robot = impl_->config.robotProfile();
    std::array<double, 6> poseRad{};
    motion->setCurrentTcp("", "welder");
    motion->queryTcpPose("", poseRad, "welder");
    motion->setCurrentTcp("", "welder");
    if (const int rc = motion->queryTcpPose("", poseRad, "current"))
    {
        throw std::runtime_error("queryTcpPose(current) failed: " + std::to_string(rc));
    }
    return PoseFromStdArrayDeg(poseRad);
}

} // namespace handeye
