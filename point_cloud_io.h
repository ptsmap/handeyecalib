#pragma once

#include <Eigen/Geometry>
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <array>
#include <filesystem>
#include <iosfwd>

namespace handeye
{
class __declspec(dllexport) PointCloudIo
{
public:
    struct Transform
    {
        std::array<std::array<double, 3>, 3> r{};
        std::array<double, 3> t{};
    };

    struct CameraCloud
    {
        unsigned int count = 0;
        const void* data = nullptr;
    };

    static void WriteCameraCloud(const std::filesystem::path& path, const CameraCloud& cloud);

    static void WriteCameraCloudToBase(
        const std::filesystem::path& path,
        const CameraCloud& cloud,
        const Transform& baseFromCamera);

    static bool WritePointXYZI(
        const std::filesystem::path& path,
        const pcl::PointCloud<pcl::PointXYZI>& cloud,
        std::ostream* errorStream = nullptr);

    static bool LoadPointXYZ(
        const std::filesystem::path& path,
        pcl::PointCloud<pcl::PointXYZ>& cloud,
        std::ostream* errorStream = nullptr);

    static bool TransformPcdFileToBase(
        const std::filesystem::path& inputPath,
        const std::filesystem::path& outputPath,
        const Transform& baseFromCamera,
        std::ostream* errorStream = nullptr);

    static bool TransformPcdFileToBase(
        const std::filesystem::path& inputPath,
        const std::filesystem::path& outputPath,
        const Eigen::Isometry3d& baseFromCamera,
        std::ostream* errorStream = nullptr);
};

} // namespace handeye
