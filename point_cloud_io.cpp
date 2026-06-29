#include "point_cloud_io.h"

#include <pcl/PCLPointField.h>
#include <pcl/io/pcd_io.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace handeye
{
namespace
{
std::array<double, 3> TransformPoint(const PointCloudIo::Transform& transform, double x, double y, double z)
{
    return {{
        transform.r[0][0] * x + transform.r[0][1] * y + transform.r[0][2] * z + transform.t[0],
        transform.r[1][0] * x + transform.r[1][1] * y + transform.r[1][2] * z + transform.t[1],
        transform.r[2][0] * x + transform.r[2][1] * y + transform.r[2][2] * z + transform.t[2],
    }};
}

PointCloudIo::Transform ToPointCloudTransform(const Eigen::Isometry3d& transform)
{
    PointCloudIo::Transform out;
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            out.r[row][col] = transform.linear()(row, col);
        }
        out.t[row] = transform.translation()(row);
    }
    return out;
}

void SaveBinaryPcd(const std::filesystem::path& path, const pcl::PCLPointCloud2& cloud)
{
    if (pcl::PCDWriter().writeBinary(path.string(), cloud) != 0)
    {
        throw std::runtime_error("Cannot open PCD file for writing: " + path.string());
    }
}

pcl::PointCloud<pcl::PointXYZ> MakePointXYZCloud(
    const Eigen::Vector3f* points,
    unsigned int count,
    const PointCloudIo::Transform* baseFromCamera = nullptr)
{
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.width = count;
    cloud.height = 1;
    cloud.is_dense = false;
    cloud.points.resize(count);

    for (unsigned int i = 0; i < count; ++i)
    {
        float x = static_cast<float>(points[i].x());
        float y = static_cast<float>(points[i].y());
        float z = static_cast<float>(points[i].z());
        if (baseFromCamera)
        {
            const std::array<double, 3> basePoint = TransformPoint(*baseFromCamera, x, y, z);
            x = static_cast<float>(basePoint[0]);
            y = static_cast<float>(basePoint[1]);
            z = static_cast<float>(basePoint[2]);
        }
        cloud.points[i].x = x;
        cloud.points[i].y = y;
        cloud.points[i].z = z;
    }
    return cloud;
}

void WritePcdPosition(const std::filesystem::path& path, const Eigen::Vector3f* points, unsigned int count)
{
    if (pcl::PCDWriter().writeBinary(path.string(), MakePointXYZCloud(points, count)) != 0)
    {
        throw std::runtime_error("Cannot open PCD file for writing: " + path.string());
    }
}

const pcl::PCLPointField* FindFloatField(const pcl::PCLPointCloud2& cloud, const std::string& name)
{
    for (const pcl::PCLPointField& field : cloud.fields)
    {
        if (field.name == name && field.datatype == pcl::PCLPointField::FLOAT32 && field.count >= 1)
        {
            return &field;
        }
    }
    return nullptr;
}

void WriteError(std::ostream* errorStream, const std::string& message)
{
    if (errorStream)
    {
        *errorStream << message << "\n";
    }
}
} // namespace

void PointCloudIo::WriteCameraCloud(const std::filesystem::path& path, const CameraCloud& cloud)
{
    WritePcdPosition(path, static_cast<const Eigen::Vector3f*>(cloud.data), cloud.count);
}

void PointCloudIo::WriteCameraCloudToBase(
    const std::filesystem::path& path,
    const CameraCloud& cloud,
    const Transform& baseFromCamera)
{
    WritePcdPosition(path, static_cast<const Eigen::Vector3f*>(cloud.data), cloud.count);
}

bool PointCloudIo::WritePointXYZI(
    const std::filesystem::path& path,
    const pcl::PointCloud<pcl::PointXYZI>& cloud,
    std::ostream* errorStream)
{
    if (pcl::PCDWriter().writeBinary(path.string(), cloud) == 0)
    {
        return true;
    }

    WriteError(errorStream, "Cannot save PCD: " + path.string());
    return false;
}

bool PointCloudIo::LoadPointXYZ(
    const std::filesystem::path& path,
    pcl::PointCloud<pcl::PointXYZ>& cloud,
    std::ostream* errorStream)
{
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(path.string(), cloud) == 0)
    {
        return true;
    }

    WriteError(errorStream, "Cannot load PCD: " + path.string());
    return false;
}

bool PointCloudIo::TransformPcdFileToBase(
    const std::filesystem::path& inputPath,
    const std::filesystem::path& outputPath,
    const Transform& baseFromCamera,
    std::ostream* errorStream)
{
    pcl::PCLPointCloud2 cloud;
    if (pcl::io::loadPCDFile(inputPath.string(), cloud) != 0)
    {
        WriteError(errorStream, "Cannot load PCD: " + inputPath.string());
        return false;
    }

    const pcl::PCLPointField* xField = FindFloatField(cloud, "x");
    const pcl::PCLPointField* yField = FindFloatField(cloud, "y");
    const pcl::PCLPointField* zField = FindFloatField(cloud, "z");
    if (!xField || !yField || !zField)
    {
        WriteError(errorStream, "PCD does not contain float x/y/z fields: " + inputPath.string());
        return false;
    }
    if (xField->offset + sizeof(float) > cloud.point_step ||
        yField->offset + sizeof(float) > cloud.point_step ||
        zField->offset + sizeof(float) > cloud.point_step)
    {
        WriteError(errorStream, "Invalid PCD x/y/z field layout: " + inputPath.string());
        return false;
    }

    const size_t pointCount = static_cast<size_t>(cloud.width) * static_cast<size_t>(cloud.height);
    for (size_t i = 0; i < pointCount; ++i)
    {
        const size_t offset = i * static_cast<size_t>(cloud.point_step);
        if (offset + cloud.point_step > cloud.data.size())
        {
            WriteError(errorStream, "Invalid PCD point data size: " + inputPath.string());
            return false;
        }

        std::uint8_t* pointData = cloud.data.data() + offset;
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        std::memcpy(&x, pointData + xField->offset, sizeof(float));
        std::memcpy(&y, pointData + yField->offset, sizeof(float));
        std::memcpy(&z, pointData + zField->offset, sizeof(float));
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
        {
            continue;
        }

        const std::array<double, 3> basePoint = TransformPoint(
            baseFromCamera,
            static_cast<double>(x),
            static_cast<double>(y),
            static_cast<double>(z));
        const float baseX = static_cast<float>(basePoint[0]);
        const float baseY = static_cast<float>(basePoint[1]);
        const float baseZ = static_cast<float>(basePoint[2]);
        std::memcpy(pointData + xField->offset, &baseX, sizeof(float));
        std::memcpy(pointData + yField->offset, &baseY, sizeof(float));
        std::memcpy(pointData + zField->offset, &baseZ, sizeof(float));
    }

    if (pcl::PCDWriter().writeBinary(outputPath.string(), cloud) != 0)
    {
        WriteError(errorStream, "Cannot save base PCD: " + outputPath.string());
        return false;
    }
    return true;
}

bool PointCloudIo::TransformPcdFileToBase(
    const std::filesystem::path& inputPath,
    const std::filesystem::path& outputPath,
    const Eigen::Isometry3d& baseFromCamera,
    std::ostream* errorStream)
{
    return TransformPcdFileToBase(inputPath, outputPath, ToPointCloudTransform(baseFromCamera), errorStream);
}

} // namespace handeye
