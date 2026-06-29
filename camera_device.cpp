#include "camera_device.h"

#include <VZNL_Common.h>
#include <VZNL_DetectConfig.h>
#include <VZNL_DetectLaser.h>
#include <VZNL_EyeConfig.h>
#include <VZNL_Graphics.h>
#include <VZNL_SwingMotor.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace handeye
{
namespace
{
std::string CameraError(int code)
{
    if (code == 0)
    {
        return "OK";
    }

    char error[256] = {};
    VzNL_GetErrorInfo(code, error);
    std::ostringstream oss;
    oss << "[" << code << "] " << error;
    return oss.str();
}

void CheckCamera(int code, const std::string& action)
{
    if (code != 0)
    {
        throw std::runtime_error(action + " failed: " + CameraError(code));
    }
}

std::vector<SVzNLEyeCBInfo> ResearchDevices()
{
    CheckCamera(VzNL_ResearchDevice(keSearchDeviceFlag_EthLaserRobotEye), "Search camera");

    int count = 0;
    CheckCamera(VzNL_GetEyeCBDeviceInfo(nullptr, &count), "Get camera count");
    if (count <= 0)
    {
        return {};
    }

    std::vector<SVzNLEyeCBInfo> devices(static_cast<size_t>(count));
    CheckCamera(VzNL_GetEyeCBDeviceInfo(devices.data(), &count), "Get camera info");
    devices.resize(static_cast<size_t>(count));
    return devices;
}

void SearchAndBindDevices(std::vector<SVzNLEyeCBInfo>& devices)
{
    bool canResearch = false;
    do
    {
        canResearch = false;
        devices = ResearchDevices();

        for (const auto& device : devices)
        {
            if (device.bValidDevice == VzTrue)
            {
                continue;
            }

            SVzNLEyeCBInfo bindDevice = device;
            SVzNLEthernetEyeConfigInfo ethernetInfo{};
            const int infoRet = VzNL_GetEthernetEyeConfigInfo(&bindDevice, &ethernetInfo);
            if (infoRet != 0)
            {
                std::cerr << "Get camera ethernet info failed: " << CameraError(infoRet) << "\n";
                continue;
            }

            const int bindRet = VzNL_BindEthernetEye(&bindDevice);
            if (bindRet != 0)
            {
                std::cerr << "Bind camera failed: " << CameraError(bindRet) << "\n";
                continue;
            }
            canResearch = true;
        }
    } while (canResearch);
}

SVzNLEyeCBInfo SelectCamera(const std::vector<SVzNLEyeCBInfo>& devices, const std::string& cameraIp)
{
    for (const auto& device : devices)
    {
        if (device.bValidDevice != VzTrue)
        {
            continue;
        }

        if (cameraIp.empty() || cameraIp == device.byServerIP)
        {
            return device;
        }
    }

    if (cameraIp.empty())
    {
        throw std::runtime_error("No valid camera found.");
    }
    throw std::runtime_error("Camera IP not found or invalid: " + cameraIp);
}
} // namespace

VizumSdk::VizumSdk()
{
    SVzNLConfigParam config{};
    config.nDeviceTimeOut = 0;
    CheckCamera(VzNL_Init(&config), "Init camera SDK");
    VzNL_SetLogLevel(keNLLogLevel_Information, keNLLogType_File);
    initialized_ = true;
}

VizumSdk::~VizumSdk()
{
    if (initialized_)
    {
        VzNL_Destroy();
    }
}

Camera::Camera(const AppConfig& config)
    : config_(config)
{
    std::vector<SVzNLEyeCBInfo> devices;
    SearchAndBindDevices(devices);

    std::cout << "Found " << devices.size() << " camera(s).\n";
    for (const auto& device : devices)
    {
        std::cout << "  IP=" << device.byServerIP
                  << " name=" << device.szDeviceName
                  << " valid=" << static_cast<int>(device.bValidDevice) << "\n";
    }

    deviceInfo_ = SelectCamera(devices, config.cameraIp);

    int errorCode = 0;
    SVzNLOpenDeviceParam openParam{};
    handle_ = VzNL_OpenDevice(&deviceInfo_, &openParam, &errorCode);
    CheckCamera(errorCode, "Open camera");
    if (handle_ == nullptr)
    {
        throw std::runtime_error("Open camera returned null handle.");
    }

    std::cout << "Camera connected: " << deviceInfo_.byServerIP << "\n";

    CheckCamera(
        VzNL_ConfigEyeExpose(handle_, keVzNLExposeMode_Fix, config_.cameraExposureTime),
        "Set camera exposure time");
    CheckCamera(
        VzNL_SetCameraGain(handle_, keEyeSensorType_Left, config_.cameraGain),
        "Set left camera gain");
    CheckCamera(
        VzNL_SetCameraGain(handle_, keEyeSensorType_Right, config_.cameraGain),
        "Set right camera gain");
    std::cout << "Camera exposure time: " << config_.cameraExposureTime
              << ", gain: " << config_.cameraGain << "\n";
    CheckCamera(VzNL_SetFrameRate(handle_, config_.cameraFrameRate), "Set camera frame rate");
    std::cout << "Set frame rate: " << config_.cameraFrameRate << "\n";

    VzNL_SetOutputImageFormat(keVzNLImageType_BGR888);
    CheckCamera(VzNL_BeginDetectLaser(handle_), "Begin laser detection");
    detectStarted_ = true;

    CheckCamera(VzNL_SetTriggerMode(handle_, keEyeTriggerMode_Master), "Set camera trigger master mode");

    const VzBool supportMotor = VzNL_IsSupportSwingMotor(handle_, nullptr);
    if (supportMotor == VzTrue)
    {
        CheckCamera(VzNL_EnableCalibROI(handle_, VzFalse), "Disable calib ROI");
        CheckCamera(VzNL_EnableSwingMotor(handle_, VzTrue), "Enable swing motor");
    }
}

Camera::~Camera()
{
    if (handle_ != nullptr)
    {
        if (detectStarted_)
        {
            VzNL_EndDetectLaser(handle_);
        }
        VzNL_CloseDevice(handle_);
    }
}

CameraPointCloud Camera::Snap()
{
    CameraPointCloud cloud;
    cloud.type = config_.resultType;

    const int ret = VzNL_PointCloudSnapShort(
        handle_,
        config_.resultType,
        config_.flipType,
        config_.mapToRgb,
        config_.clipRgb,
        &cloud.image,
        &cloud.count,
        &cloud.data);
    CheckCamera(ret, "Point cloud snapshot");

    if (cloud.data == nullptr || cloud.count == 0)
    {
        Release(cloud);
        throw std::runtime_error("Point cloud snapshot returned no points.");
    }

    return cloud;
}

void Camera::Release(CameraPointCloud& cloud)
{
    if (cloud.data != nullptr || cloud.image != nullptr)
    {
        VzNL_ReleasePointCloudSnapShortResult(&cloud.image, &cloud.data);
        cloud.count = 0;
    }
}

} // namespace handeye
