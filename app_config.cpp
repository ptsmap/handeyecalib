#include "app_config.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace handeye
{
namespace fs = std::filesystem;

const DeviceProfile& AppConfig::robotProfile() const
{
    const auto it = robots.find(activeRobot);
    if (it != robots.end())
    {
        return it->second;
    }
    throw std::runtime_error("Active robot profile not found: " + activeRobot);
}

const DeviceProfile& AppConfig::cameraProfile() const
{
    const auto it = cameras.find(activeCamera);
    if (it != cameras.end())
    {
        return it->second;
    }
    throw std::runtime_error("Active camera profile not found: " + activeCamera);
}

std::string Trim(const std::string& value)
{
    const auto begin = std::find_if_not(value.begin(), value.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    });
    const auto end = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    }).base();
    if (begin >= end)
    {
        return {};
    }
    return std::string(begin, end);
}

std::string ToLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

namespace
{
std::vector<std::string> SplitCsv(const std::string& value)
{
    std::vector<std::string> parts;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ','))
    {
        item = Trim(item);
        if (!item.empty())
        {
            parts.push_back(item);
        }
    }
    return parts;
}

bool ParseBool(const std::string& value)
{
    const std::string lower = ToLower(Trim(value));
    return lower == "1" || lower == "true" || lower == "yes" || lower == "on";
}

DeviceProfile MakeDefaultRobotProfile(const std::string& name)
{
    DeviceProfile profile;
    profile.name = name;
    profile.type = "hans";
    profile.ip = "192.168.100.11";
    profile.port = 10003;
    profile.boxId = 0;
    profile.robotId = 0;
    profile.flangeName = "TCP";
    profile.welderName = "welder";
    return profile;
}

DeviceProfile MakeDefaultCameraProfile(const std::string& name)
{
    DeviceProfile profile;
    profile.name = name;
    profile.type = "VzNL";
    profile.exposureTime = 200;
    profile.gain = {{1, 1}};
    profile.frameRate = 160;
    profile.swingStart = 0;
    profile.swingEnd = 68;
    profile.swingSpeed = 30.0f;
    return profile;
}

DeviceProfile& EnsureRobotProfile(AppConfig& config, const std::string& name)
{
    auto [it, inserted] = config.robots.emplace(name, MakeDefaultRobotProfile(name));
    if (inserted)
    {
        it->second.name = name;
    }
    return it->second;
}

DeviceProfile& EnsureCameraProfile(AppConfig& config, const std::string& name)
{
    auto [it, inserted] = config.cameras.emplace(name, MakeDefaultCameraProfile(name));
    if (inserted)
    {
        it->second.name = name;
    }
    return it->second;
}

void LoadLegacyRobotKeys(const std::map<std::string, std::string>& values, AppConfig& config)
{
    DeviceProfile& profile = EnsureRobotProfile(config, config.activeRobot);
    if (values.count("robot_ip"))
    {
        profile.ip = values.at("robot_ip");
    }
    if (values.count("robot_port"))
    {
        profile.port = static_cast<unsigned short>(std::stoul(values.at("robot_port")));
    }
    if (values.count("robot_box_id"))
    {
        profile.boxId = static_cast<unsigned int>(std::stoul(values.at("robot_box_id")));
    }
    if (values.count("robot_id"))
    {
        profile.robotId = static_cast<unsigned int>(std::stoul(values.at("robot_id")));
    }
    if (values.count("robot_type"))
    {
        profile.type = values.at("robot_type");
    }
    if (values.count("robot_flange_name"))
    {
        profile.flangeName = values.at("robot_flange_name");
    }
    if (values.count("robot_welder_name"))
    {
        profile.welderName = values.at("robot_welder_name");
    }
}

void LoadLegacyCameraKeys(const std::map<std::string, std::string>& values, AppConfig& config)
{
    DeviceProfile& profile = EnsureCameraProfile(config, config.activeCamera);
    if (values.count("camera_ip"))
    {
        profile.ip = values.at("camera_ip");
        profile.key = profile.ip;
    }
    if (values.count("camera_exposure_time"))
    {
        profile.exposureTime = static_cast<unsigned int>(std::stoul(values.at("camera_exposure_time")));
    }
    if (values.count("camera_gain"))
    {
        const int gain = static_cast<int>(std::stoul(values.at("camera_gain")));
        profile.gain = {{gain, gain}};
    }
    if (values.count("camera_frame_rate"))
    {
        profile.frameRate = static_cast<unsigned short>(std::stoul(values.at("camera_frame_rate")));
    }
}

void LoadProfileKeys(const std::map<std::string, std::string>& values, AppConfig& config)
{
    for (const auto& [key, value] : values)
    {
        const auto firstDot = key.find('.');
        const auto secondDot = key.find('.', firstDot == std::string::npos ? firstDot : firstDot + 1);
        if (firstDot == std::string::npos || secondDot == std::string::npos)
        {
            continue;
        }

        const std::string group = key.substr(0, firstDot);
        const std::string profileName = key.substr(firstDot + 1, secondDot - firstDot - 1);
        const std::string field = key.substr(secondDot + 1);

        DeviceProfile* profile = nullptr;
        if (group == "robot")
        {
            profile = &EnsureRobotProfile(config, profileName);
        }
        else if (group == "camera")
        {
            profile = &EnsureCameraProfile(config, profileName);
        }
        else
        {
            continue;
        }

        if (field == "type")
        {
            profile->type = value;
        }
        else if (field == "key")
        {
            profile->key = value;
        }
        else if (field == "ip")
        {
            profile->ip = value;
        }
        else if (field == "port")
        {
            profile->port = static_cast<unsigned short>(std::stoul(value));
        }
        else if (field == "box_id")
        {
            profile->boxId = static_cast<unsigned int>(std::stoul(value));
        }
        else if (field == "robot_id")
        {
            profile->robotId = static_cast<unsigned int>(std::stoul(value));
        }
        else if (field == "flange_name")
        {
            profile->flangeName = value;
        }
        else if (field == "welder_name")
        {
            profile->welderName = value;
        }
        else if (field == "exposure_time")
        {
            profile->exposureTime = static_cast<unsigned int>(std::stoul(value));
        }
        else if (field == "gain_left")
        {
            profile->gain[0] = static_cast<int>(std::stoul(value));
        }
        else if (field == "gain_right")
        {
            profile->gain[1] = static_cast<int>(std::stoul(value));
        }
        else if (field == "frame_rate")
        {
            profile->frameRate = static_cast<unsigned short>(std::stoul(value));
        }
        else if (field == "swing_start")
        {
            profile->swingStart = static_cast<int>(std::stoi(value));
        }
        else if (field == "swing_end")
        {
            profile->swingEnd = static_cast<int>(std::stoi(value));
        }
        else if (field == "swing_speed")
        {
            profile->swingSpeed = std::stof(value);
        }
    }
}
} // namespace

AppConfig LoadConfig(const fs::path& configPath)
{
    AppConfig config;
    config.robots.emplace(config.activeRobot, MakeDefaultRobotProfile(config.activeRobot));
    config.cameras.emplace(config.activeCamera, MakeDefaultCameraProfile(config.activeCamera));

    std::ifstream in(configPath);
    if (!in)
    {
        std::cout << "Config not found, using defaults: " << configPath.string() << "\n";
        return config;
    }

    std::map<std::string, std::string> values;
    std::string line;
    while (std::getline(in, line))
    {
        const auto comment = line.find('#');
        if (comment != std::string::npos)
        {
            line = line.substr(0, comment);
        }

        const auto sep = line.find('=');
        if (sep == std::string::npos)
        {
            continue;
        }

        const std::string key = ToLower(Trim(line.substr(0, sep)));
        const std::string value = Trim(line.substr(sep + 1));
        if (!key.empty())
        {
            values[key] = value;
        }
    }

    if (values.count("active_robot"))
    {
        config.activeRobot = values["active_robot"];
    }
    if (values.count("active_camera"))
    {
        config.activeCamera = values["active_camera"];
    }
    if (values.count("robot_profiles"))
    {
        for (const std::string& name : SplitCsv(values["robot_profiles"]))
        {
            EnsureRobotProfile(config, name);
        }
    }
    if (values.count("camera_profiles"))
    {
        for (const std::string& name : SplitCsv(values["camera_profiles"]))
        {
            EnsureCameraProfile(config, name);
        }
    }

    LoadLegacyRobotKeys(values, config);
    LoadLegacyCameraKeys(values, config);
    LoadProfileKeys(values, config);

    if (values.count("camera_exposure_time"))
    {
        const unsigned long exposureTime = std::stoul(values["camera_exposure_time"]);
        if (exposureTime > 65535)
        {
            throw std::runtime_error("camera_exposure_time out of range: " + values["camera_exposure_time"] + ". Use 0..65535.");
        }
        EnsureCameraProfile(config, config.activeCamera).exposureTime = static_cast<unsigned int>(exposureTime);
    }
    if (values.count("camera_gain"))
    {
        const unsigned long gain = std::stoul(values["camera_gain"]);
        if (gain > 255)
        {
            throw std::runtime_error("camera_gain out of range: " + values["camera_gain"] + ". Use 0..255.");
        }
        EnsureCameraProfile(config, config.activeCamera).gain = {{static_cast<int>(gain), static_cast<int>(gain)}};
    }
    if (values.count("camera_frame_rate"))
    {
        const unsigned long frameRate = std::stoul(values["camera_frame_rate"]);
        if (frameRate > 255)
        {
            throw std::runtime_error("camera_frame_rate out of range: " + values["camera_frame_rate"] + ". Use 0..255.");
        }
        EnsureCameraProfile(config, config.activeCamera).frameRate = static_cast<unsigned short>(frameRate);
    }
    
    if (values.count("export_feature_cloud"))
    {
        config.exportFeatureCloud = ParseBool(values["export_feature_cloud"]);
    }
    
    if (values.count("output_dir"))
    {
        config.outputDir = values["output_dir"];
    }

    if (!config.outputDir.is_absolute())
    {
        config.outputDir = fs::absolute(configPath.parent_path() / config.outputDir);
    }

    EnsureRobotProfile(config, config.activeRobot);
    EnsureCameraProfile(config, config.activeCamera);

    return config;
}

} // namespace handeye
