#pragma once

#include "point_cloud_io.h"
#include "pose_io.h"

#include <Eigen/Geometry>

#include <array>
#include <filesystem>

namespace handeye
{
using Quaternion = PoseIo::Quaternion;
using Pose = PoseIo::Pose;
using Transform = PointCloudIo::Transform;

std::array<std::array<double, 3>, 3> __declspec(dllexport) RotationFromRpy(double rxDeg, double ryDeg, double rzDeg);
Quaternion __declspec(dllexport) QuaternionFromRotation(const std::array<std::array<double, 3>, 3>& r);
Transform __declspec(dllexport) ToTransform(const Pose& pose);
Transform __declspec(dllexport) ToTransform(const Eigen::Isometry3d& transform);
Transform __declspec(dllexport) TransformFromPoseAndQuaternion(const Pose& pose, Quaternion q);
Transform __declspec(dllexport) Inverse(const Transform& transform);
Transform __declspec(dllexport) Multiply(const Transform& lhs, const Transform& rhs);
std::array<double, 3> __declspec(dllexport) TransformPoint(const Transform& transform, double x, double y, double z);
Pose __declspec(dllexport) PoseFromTransform(const Transform& transform, const Pose& orientationHint);
bool __declspec(dllexport) ReadHandEyeMatrix(const std::filesystem::path& matrixPath, Transform& flangeFromCamera);

} // namespace handeye
