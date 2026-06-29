#include "transform_utils.h"

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace handeye
{
namespace
{
constexpr double kPi = 3.14159265358979323846;

double DegToRad(double degree)
{
    return degree * kPi / 180.0;
}
} // namespace

std::array<std::array<double, 3>, 3> RotationFromRpy(double rxDeg, double ryDeg, double rzDeg)
{
    const double rx = DegToRad(rxDeg);
    const double ry = DegToRad(ryDeg);
    const double rz = DegToRad(rzDeg);

    const double cx = std::cos(rx);
    const double sx = std::sin(rx);
    const double cy = std::cos(ry);
    const double sy = std::sin(ry);
    const double cz = std::cos(rz);
    const double sz = std::sin(rz);

    return {{
        {{cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx}},
        {{sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx}},
        {{-sy, cy * sx, cy * cx}},
    }};
}

Quaternion QuaternionFromRotation(const std::array<std::array<double, 3>, 3>& r)
{
    Quaternion q;
    const double trace = r[0][0] + r[1][1] + r[2][2];
    if (trace > 0.0)
    {
        const double s = std::sqrt(trace + 1.0) * 2.0;
        q.w = 0.25 * s;
        q.x = (r[2][1] - r[1][2]) / s;
        q.y = (r[0][2] - r[2][0]) / s;
        q.z = (r[1][0] - r[0][1]) / s;
    }
    else if (r[0][0] > r[1][1] && r[0][0] > r[2][2])
    {
        const double s = std::sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0;
        q.w = (r[2][1] - r[1][2]) / s;
        q.x = 0.25 * s;
        q.y = (r[0][1] + r[1][0]) / s;
        q.z = (r[0][2] + r[2][0]) / s;
    }
    else if (r[1][1] > r[2][2])
    {
        const double s = std::sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0;
        q.w = (r[0][2] - r[2][0]) / s;
        q.x = (r[0][1] + r[1][0]) / s;
        q.y = 0.25 * s;
        q.z = (r[1][2] + r[2][1]) / s;
    }
    else
    {
        const double s = std::sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0;
        q.w = (r[1][0] - r[0][1]) / s;
        q.x = (r[0][2] + r[2][0]) / s;
        q.y = (r[1][2] + r[2][1]) / s;
        q.z = 0.25 * s;
    }

    const double norm = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    if (norm > 0.0)
    {
        q.w /= norm;
        q.x /= norm;
        q.y /= norm;
        q.z /= norm;
    }
    return q;
}

Transform ToTransform(const Pose& pose)
{
    Transform transform;
    transform.r = RotationFromRpy(pose.rx, pose.ry, pose.rz);
    transform.t = {{pose.x, pose.y, pose.z}};
    return transform;
}

Transform ToTransform(const Eigen::Isometry3d& transform)
{
    Transform out;
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

Transform TransformFromPoseAndQuaternion(const Pose& pose, Quaternion q)
{
    const double norm = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    if (norm > 0.0)
    {
        q.w /= norm;
        q.x /= norm;
        q.y /= norm;
        q.z /= norm;
    }
    else
    {
        q = Quaternion{};
    }

    const double xx = q.x * q.x;
    const double yy = q.y * q.y;
    const double zz = q.z * q.z;
    const double xy = q.x * q.y;
    const double xz = q.x * q.z;
    const double yz = q.y * q.z;
    const double wx = q.w * q.x;
    const double wy = q.w * q.y;
    const double wz = q.w * q.z;

    Transform transform;
    transform.r = {{
        {{1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)}},
        {{2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)}},
        {{2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)}},
    }};
    transform.t = {{pose.x, pose.y, pose.z}};
    return transform;
}

Transform Inverse(const Transform& transform)
{
    Transform inv;
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            inv.r[row][col] = transform.r[col][row];
        }
    }

    for (int row = 0; row < 3; ++row)
    {
        inv.t[row] = 0.0;
        for (int col = 0; col < 3; ++col)
        {
            inv.t[row] -= inv.r[row][col] * transform.t[col];
        }
    }
    return inv;
}

Transform Multiply(const Transform& lhs, const Transform& rhs)
{
    Transform out;
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            out.r[row][col] = 0.0;
            for (int k = 0; k < 3; ++k)
            {
                out.r[row][col] += lhs.r[row][k] * rhs.r[k][col];
            }
        }
    }

    for (int row = 0; row < 3; ++row)
    {
        out.t[row] = lhs.t[row];
        for (int k = 0; k < 3; ++k)
        {
            out.t[row] += lhs.r[row][k] * rhs.t[k];
        }
    }
    return out;
}

std::array<double, 3> TransformPoint(const Transform& transform, double x, double y, double z)
{
    return {{
        transform.r[0][0] * x + transform.r[0][1] * y + transform.r[0][2] * z + transform.t[0],
        transform.r[1][0] * x + transform.r[1][1] * y + transform.r[1][2] * z + transform.t[1],
        transform.r[2][0] * x + transform.r[2][1] * y + transform.r[2][2] * z + transform.t[2],
    }};
}

Pose PoseFromTransform(const Transform& transform, const Pose& orientationHint)
{
    Pose pose;
    pose.x = transform.t[0];
    pose.y = transform.t[1];
    pose.z = transform.t[2];
    pose.rx = orientationHint.rx;
    pose.ry = orientationHint.ry;
    pose.rz = orientationHint.rz;
    return pose;
}

bool ReadHandEyeMatrix(const std::filesystem::path& matrixPath, Transform& flangeFromCamera)
{
    if (!std::filesystem::exists(matrixPath))
    {
        return false;
    }

    std::ifstream in(matrixPath);
    if (!in)
    {
        throw std::runtime_error("Cannot open hand-eye matrix: " + matrixPath.string());
    }

    std::array<double, 16> values{};
    for (double& value : values)
    {
        if (!(in >> value) || !std::isfinite(value))
        {
            throw std::runtime_error("Invalid hand-eye matrix, expected 16 finite values: " + matrixPath.string());
        }
    }

    std::cout << "Hand-eye matrix from: " << matrixPath.string() << "\n";
    std::cout << "flangeFromCamera (4x4):\n";
    std::cout << std::fixed << std::setprecision(4);
    for (int row = 0; row < 4; ++row)
    {
        std::cout << "  ";
        for (int col = 0; col < 4; ++col)
        {
            if (col > 0)
            {
                std::cout << ' ';
            }
            std::cout << values[row * 4 + col];
        }
        std::cout << "\n";
    }
    
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            flangeFromCamera.r[row][col] = values[static_cast<size_t>(row) * 4 + static_cast<size_t>(col)];
        }
        flangeFromCamera.t[row] = values[static_cast<size_t>(row) * 4 + 3];
    }
    return true;
}

} // namespace handeye
