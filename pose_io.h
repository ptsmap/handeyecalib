#pragma once

#include <Eigen/Geometry>

#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

namespace handeye
{
class __declspec(dllexport) PoseIo
{
public:
    struct Pose
    {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        double rx = 0.0;
        double ry = 0.0;
        double rz = 0.0;
    };

    struct Quaternion
    {
        double w = 1.0;
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
    };

    struct Record
    {
        std::string timestamp;
        std::filesystem::path cloudPath;
        Eigen::Isometry3d baseFromFlange = Eigen::Isometry3d::Identity();
    };

    struct ReadOptions
    {
        bool requireExistingCloud = true;
        bool resolveSiblingTcpCloud = true;
        bool individualFilesOverrideIndex = false;
        std::ostream* errorStream = nullptr;
    };

    static std::vector<Record> ReadPoseFile(
        const std::filesystem::path& path,
        const std::filesystem::path& dataDir,
        std::ostream* errorStream = nullptr);

    static std::vector<Record> ReadPoseRecords(
        const std::filesystem::path& dataDir,
        const ReadOptions& options = ReadOptions{});

    static const Record* FindForCloud(
        const std::vector<Record>& records,
        const std::filesystem::path& cloudPath);

    static void WriteSinglePoseFile(
        const std::filesystem::path& path,
        const std::string& stamp,
        const std::filesystem::path& cloudFile,
        const Pose& flangePose,
        const Quaternion& flangeQuaternion,
        const Pose& tcpPose);

    static void AppendPoseIndex(
        const std::filesystem::path& path,
        const std::string& stamp,
        const std::filesystem::path& cloudFile,
        const Pose& flangePose,
        const Quaternion& flangeQuaternion,
        const Pose& tcpPose);
};

} // namespace handeye
