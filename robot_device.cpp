#include "robot_device.h"

#include <HR_Pro.h>
#include <DucoCobot.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace handeye
{
namespace
{
// =========================================================================
// Unit conversion
// =========================================================================
constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;
constexpr double kMToMm = 1000.0;
constexpr double kMmToM = 0.001;
inline double RadToDeg(double r) { return r * kRadToDeg; }
inline double MToMm(double m) { return m * kMToMm; }
inline double MmToM(double mm) { return mm * kMmToM; }

bool NearlyZeroTcp(const Pose& pose)
{
    return std::abs(pose.x) < 1e-9 && std::abs(pose.y) < 1e-9 && std::abs(pose.z) < 1e-9 &&
           std::abs(pose.rx) < 1e-9 && std::abs(pose.ry) < 1e-9 && std::abs(pose.rz) < 1e-9;
}

// =========================================================================
// Backend interface
// =========================================================================
class RobotBackend
{
public:
    virtual ~RobotBackend() = default;
    virtual RobotSnapshot ReadSnapshot(const std::string& toolName) const = 0;
};

// =========================================================================
// Huayan robot backend
// =========================================================================
class HuayanBackend : public RobotBackend
{
public:
    explicit HuayanBackend(const AppConfig& config) : config_(config)
    {
        const int ret = HRIF_Connect(config_.robotBoxId, config_.robotIp.c_str(), config_.robotPort);
        Check(config_.robotBoxId, ret, "Connect robot");
        if (!HRIF_IsConnected(config_.robotBoxId))
            throw std::runtime_error("Robot SDK reports disconnected after HRIF_Connect.");
        std::cout << "Robot connected (Huayan): " << config_.robotIp
                  << ":" << config_.robotPort << "\n";
    }
    ~HuayanBackend() override
    {
        if (HRIF_IsConnected(config_.robotBoxId)) HRIF_DisConnect(config_.robotBoxId);
    }

    RobotSnapshot ReadSnapshot(const std::string& toolName) const override
    {
        RobotSnapshot snap;
        double j1=0,j2=0,j3=0,j4=0,j5=0,j6=0;
        (void)HRIF_SetTCPByName(config_.robotBoxId, config_.robotId, toolName.c_str());
        const int ret = HRIF_ReadActPos(
            config_.robotBoxId, config_.robotId,
            snap.flangePose.x,snap.flangePose.y,snap.flangePose.z,
            snap.flangePose.rx,snap.flangePose.ry,snap.flangePose.rz,
            j1,j2,j3,j4,j5,j6,
            snap.tcpPose.x,snap.tcpPose.y,snap.tcpPose.z,
            snap.tcpPose.rx,snap.tcpPose.ry,snap.tcpPose.rz,
            snap.ucsPose.x,snap.ucsPose.y,snap.ucsPose.z,
            snap.ucsPose.rx,snap.ucsPose.ry,snap.ucsPose.rz);
        Check(config_.robotBoxId, ret, "Read robot actual pose");
        snap.joints = {{j1,j2,j3,j4,j5,j6}};

        double qw=0,qx=0,qy=0,qz=0;
        const int qRet = HRIF_RPY2Quaternion(
            config_.robotBoxId, config_.robotId,
            snap.flangePose.rx,snap.flangePose.ry,snap.flangePose.rz, qw,qx,qy,qz);
        if (qRet == 0)
            snap.flangeQuaternion = {qw,qx,qy,qz};
        else
        {
            snap.flangeQuaternion = QuaternionFromRotation(
                RotationFromRpy(snap.flangePose.rx, snap.flangePose.ry, snap.flangePose.rz));
            std::cerr << "Robot SDK RPY2Quaternion failed, code=" << qRet << "\n";
        }
        return snap;
    }

private:
    AppConfig config_;
    static std::string ErrStr(unsigned int b, int c) {
        if (c==0) return "OK";
        std::string e; HRIF_GetErrorCodeStr(b,c,e);
        return "["+std::to_string(c)+"]"+(e.empty()?"":" "+e);
    }
    static void Check(unsigned int b, int c, const std::string& a) {
        if (c!=0) throw std::runtime_error(a+" failed: "+ErrStr(b,c));
    }
};

// =========================================================================
// Duco robot backend
// =========================================================================
class DucoBackend : public RobotBackend
{
public:
    explicit DucoBackend(const AppConfig& config) : config_(config)
    {
        robot_ = std::make_unique<DucoRPC::DucoCobot>(config_.robotIp, config_.robotPort);
        const int ret = robot_->open();
        if (ret != 0)
            throw std::runtime_error("Duco robot open failed, code=" + std::to_string(ret));
        std::cout << "Robot connected (Duco): " << config_.robotIp
                  << ":" << config_.robotPort << "\n";
    }
    ~DucoBackend() override { if (robot_) robot_->close(); }

    RobotSnapshot ReadSnapshot(const std::string& toolName) const override
    {
        RobotSnapshot snap;
        DucoRPC::RobotStatusList status;
        robot_->getRobotStatus(status);

        // cartActualPosition: [x,y,z,rx,ry,rz] in m, rad → mm, deg
        if (status.cartActualPosition.size() >= 6)
        {
            snap.tcpPose.x = MToMm(status.cartActualPosition[0]);
            snap.tcpPose.y = MToMm(status.cartActualPosition[1]);
            snap.tcpPose.z = MToMm(status.cartActualPosition[2]);
            snap.tcpPose.rx = RadToDeg(status.cartActualPosition[3]);
            snap.tcpPose.ry = RadToDeg(status.cartActualPosition[4]);
            snap.tcpPose.rz = RadToDeg(status.cartActualPosition[5]);
        }

        // TCP offset: [x,y,z,rx,ry,rz] in m, rad → mm, deg
        std::vector<double> tcpOff;
        robot_->get_tcp_offset(tcpOff);
        if (tcpOff.size() >= 6)
        {
            snap.tcpOffset.x = MToMm(tcpOff[0]);
            snap.tcpOffset.y = MToMm(tcpOff[1]);
            snap.tcpOffset.z = MToMm(tcpOff[2]);
            snap.tcpOffset.rx = RadToDeg(tcpOff[3]);
            snap.tcpOffset.ry = RadToDeg(tcpOff[4]);
            snap.tcpOffset.rz = RadToDeg(tcpOff[5]);
        }

        // Compute flange pose: baseToFlange = baseToTcp * inv(tcpOffset)
        if (!NearlyZeroTcp(snap.tcpOffset))
        {
            const Transform baseToTcp = ToTransform(snap.tcpPose);
            const Transform flangeToTcp = ToTransform(snap.tcpOffset);
            const Transform baseToFlange = Multiply(baseToTcp, Inverse(flangeToTcp));
            snap.flangePose = PoseFromTransform(baseToFlange, snap.tcpPose);
        }
        else
            snap.flangePose = snap.tcpPose;

        snap.flangeQuaternion = QuaternionFromRotation(
            RotationFromRpy(snap.flangePose.rx, snap.flangePose.ry, snap.flangePose.rz));

        if (status.jointActualPosition.size() >= 6)
        {
            snap.joints = {{
                RadToDeg(status.jointActualPosition[0]),
                RadToDeg(status.jointActualPosition[1]),
                RadToDeg(status.jointActualPosition[2]),
                RadToDeg(status.jointActualPosition[3]),
                RadToDeg(status.jointActualPosition[4]),
                RadToDeg(status.jointActualPosition[5])}};
        }
        return snap;
    }

private:
    AppConfig config_;
    std::unique_ptr<DucoRPC::DucoCobot> robot_;
};

} // namespace

// =========================================================================
// Robot (public API)
// =========================================================================

struct Robot::Impl
{
    std::unique_ptr<RobotBackend> backend;
};

Robot::Robot(const AppConfig& config)
    : impl_(std::make_unique<Impl>())
{
    const std::string type = ToLower(config.robotType);
    if (type == "duco")
        impl_->backend = std::make_unique<DucoBackend>(config);
    else
        impl_->backend = std::make_unique<HuayanBackend>(config);
}

Robot::~Robot() = default;

RobotSnapshot Robot::ReadSnapshot(const std::string& toolName) const
{
    return impl_->backend->ReadSnapshot(toolName);
}

} // namespace handeye
