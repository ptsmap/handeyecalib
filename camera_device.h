#pragma once

#include "app_config.h"

#include <VZNL_Types.h>

namespace handeye
{
class __declspec(dllexport) VizumSdk
{
public:
    VizumSdk();
    ~VizumSdk();

    VizumSdk(const VizumSdk&) = delete;
    VizumSdk& operator=(const VizumSdk&) = delete;

private:
    bool initialized_ = false;
};

struct CameraPointCloud
{
    EVzResultDataType type = keResultDataType_PointXYZRGBA;
    unsigned int count = 0;
    void* data = nullptr;
    SVzNLImageData* image = nullptr;
};

class __declspec(dllexport) Camera
{
public:
    explicit Camera(const AppConfig& config);
    ~Camera();

    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;

    CameraPointCloud Snap();
    static void Release(CameraPointCloud& cloud);

private:
    AppConfig config_;
    SVzNLEyeCBInfo deviceInfo_{};
    VZNLHANDLE handle_ = nullptr;
    bool detectStarted_ = false;
};

} // namespace handeye
