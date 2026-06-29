#pragma once

#include <array>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace handeye
{
struct DeviceProfile
{
    std::string name;
    std::string type;
    std::string key;
    std::string ip;
    unsigned short port = 0;
    unsigned int boxId = 0;
    unsigned int robotId = 0;
    std::string flangeName;
    std::string welderName;
    unsigned int exposureTime = 200;
    std::array<int, 2> gain{{1, 1}};
    unsigned short frameRate = 160;
    int swingStart = 0;
    int swingEnd = 68;
    float swingSpeed = 30.0f;
};

struct __declspec(dllexport) AppConfig
{
    std::string activeRobot = "robot1";
    std::string activeCamera = "camera1";
    std::map<std::string, DeviceProfile> robots;
    std::map<std::string, DeviceProfile> cameras;

    std::filesystem::path outputDir = "output";
    bool exportFeatureCloud = true;

    const DeviceProfile& robotProfile() const;
    const DeviceProfile& cameraProfile() const;
};

std::string __declspec(dllexport) Trim(const std::string& value);
std::string __declspec(dllexport) ToLower(std::string value);
AppConfig __declspec(dllexport) LoadConfig(const std::filesystem::path& configPath);

} // namespace handeye
