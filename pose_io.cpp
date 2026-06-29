#include "pose_io.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace handeye
{
namespace fs = std::filesystem;

namespace
{
struct HeaderTable
{
    std::vector<std::string> names;
    std::map<std::string, size_t> index;
};

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

std::vector<std::string> SplitWhitespace(const std::string& line)
{
    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::string token;
    while (iss >> token)
    {
        tokens.push_back(token);
    }
    return tokens;
}

HeaderTable ParseHeader(const std::string& line)
{
    std::string text = Trim(line);
    if (!text.empty() && text.front() == '#')
    {
        text.erase(text.begin());
    }

    HeaderTable header;
    header.names = SplitWhitespace(text);
    for (size_t i = 0; i < header.names.size(); ++i)
    {
        header.index[ToLower(header.names[i])] = i;
    }
    return header;
}

bool FieldString(const HeaderTable& header, const std::vector<std::string>& row, const std::string& name, std::string& value)
{
    const auto it = header.index.find(ToLower(name));
    if (it == header.index.end() || it->second >= row.size())
    {
        return false;
    }
    value = row[it->second];
    return true;
}

bool FieldDouble(const HeaderTable& header, const std::vector<std::string>& row, const std::string& name, double& value)
{
    std::string text;
    if (!FieldString(header, row, name, text))
    {
        return false;
    }

    try
    {
        size_t parsed = 0;
        value = std::stod(text, &parsed);
        return parsed == text.size() && std::isfinite(value);
    }
    catch (const std::exception&)
    {
        return false;
    }
}

Eigen::Isometry3d MakeTransform(const Eigen::Vector3d& translation, Eigen::Quaterniond quaternion)
{
    if (quaternion.norm() > 0.0)
    {
        quaternion.normalize();
    }
    else
    {
        quaternion = Eigen::Quaterniond::Identity();
    }
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.linear() = quaternion.toRotationMatrix();
    transform.translation() = translation;
    return transform;
}

std::optional<PoseIo::Record> ParsePoseRow(
    const fs::path& dataDir,
    const HeaderTable& header,
    const std::vector<std::string>& row)
{
    std::string timestamp;
    std::string cloudFile;
    double flangeX = 0.0;
    double flangeY = 0.0;
    double flangeZ = 0.0;
    double flangeQw = 1.0;
    double flangeQx = 0.0;
    double flangeQy = 0.0;
    double flangeQz = 0.0;
    if (!FieldString(header, row, "timestamp", timestamp) ||
        !FieldString(header, row, "point_cloud_file", cloudFile) ||
        !FieldDouble(header, row, "flange_x", flangeX) ||
        !FieldDouble(header, row, "flange_y", flangeY) ||
        !FieldDouble(header, row, "flange_z", flangeZ) ||
        !FieldDouble(header, row, "flange_qw", flangeQw) ||
        !FieldDouble(header, row, "flange_qx", flangeQx) ||
        !FieldDouble(header, row, "flange_qy", flangeQy) ||
        !FieldDouble(header, row, "flange_qz", flangeQz))
    {
        return std::nullopt;
    }

    fs::path cloudPath = cloudFile;
    if (cloudPath.is_relative())
    {
        cloudPath = dataDir / cloudPath;
    }

    PoseIo::Record record;
    record.timestamp = timestamp;
    record.cloudPath = cloudPath;
    record.baseFromFlange = MakeTransform(
        Eigen::Vector3d(flangeX, flangeY, flangeZ),
        Eigen::Quaterniond(flangeQw, flangeQx, flangeQy, flangeQz));
    return record;
}

void WritePoseHeader(std::ofstream& out)
{
    out << "timestamp point_cloud_file flange_x flange_y flange_z flange_qw flange_qx flange_qy flange_qz "
        << "flange_rx_deg flange_ry_deg flange_rz_deg tcp_x tcp_y tcp_z tcp_rx_deg tcp_ry_deg tcp_rz_deg\n";
}

void WritePoseRow(
    std::ofstream& out,
    const std::string& stamp,
    const fs::path& cloudFile,
    const PoseIo::Pose& flangePose,
    const PoseIo::Quaternion& flangeQuaternion,
    const PoseIo::Pose& tcpPose)
{
    out << std::fixed << std::setprecision(4)
        << stamp << ' '
        << cloudFile.filename().string() << ' '
        << flangePose.x << ' '
        << flangePose.y << ' '
        << flangePose.z << ' '
        << flangeQuaternion.w << ' '
        << flangeQuaternion.x << ' '
        << flangeQuaternion.y << ' '
        << flangeQuaternion.z << ' '
        << flangePose.rx << ' '
        << flangePose.ry << ' '
        << flangePose.rz << ' '
        << tcpPose.x << ' '
        << tcpPose.y << ' '
        << tcpPose.z << ' '
        << tcpPose.rx << ' '
        << tcpPose.ry << ' '
        << tcpPose.rz << '\n';
}

std::string TimestampFromCloudPath(const fs::path& cloudPath)
{
    const std::string stem = cloudPath.stem().string();
    const auto firstDigit = std::find_if(stem.begin(), stem.end(), [](unsigned char ch) {
        return std::isdigit(ch) != 0;
    });
    if (firstDigit == stem.end())
    {
        return stem;
    }
    return std::string(firstDigit, stem.end());
}
} // namespace

std::vector<PoseIo::Record> PoseIo::ReadPoseFile(
    const fs::path& path,
    const fs::path& dataDir,
    std::ostream* errorStream)
{
    std::ifstream in(path);
    if (!in)
    {
        if (errorStream)
        {
            *errorStream << "Cannot open pose file: " << path.string() << "\n";
        }
        return {};
    }

    HeaderTable header;
    bool haveHeader = false;
    std::vector<Record> records;
    std::string line;
    while (std::getline(in, line))
    {
        line = Trim(line);
        if (line.empty())
        {
            continue;
        }

        const std::string lower = ToLower(line);
        if (lower.find("timestamp") != std::string::npos && lower.find("point_cloud_file") != std::string::npos)
        {
            header = ParseHeader(line);
            haveHeader = true;
            continue;
        }

        if (!haveHeader)
        {
            if (errorStream)
            {
                *errorStream << "Pose file has no header: " << path.string() << "\n";
            }
            return records;
        }

        std::optional<Record> record = ParsePoseRow(dataDir, header, SplitWhitespace(line));
        if (record)
        {
            records.push_back(*record);
        }
        else if (errorStream)
        {
            *errorStream << "Skip invalid pose row in: " << path.string() << "\n";
        }
    }
    return records;
}

std::vector<PoseIo::Record> PoseIo::ReadPoseRecords(const fs::path& dataDir, const ReadOptions& options)
{
    std::map<std::string, Record> byTimestamp;

    const fs::path indexPath = dataDir / "robot_records.pose";
    if (fs::exists(indexPath))
    {
        for (const Record& record : ReadPoseFile(indexPath, dataDir, options.errorStream))
        {
            byTimestamp[record.timestamp] = record;
        }
    }

    for (const fs::directory_entry& entry : fs::directory_iterator(dataDir))
    {
        if (!entry.is_regular_file() || entry.path().extension() != ".pose")
        {
            continue;
        }
        if (entry.path().filename() == "robot_records.pose")
        {
            continue;
        }

        for (const Record& record : ReadPoseFile(entry.path(), dataDir, options.errorStream))
        {
            if (options.individualFilesOverrideIndex)
            {
                byTimestamp[record.timestamp] = record;
            }
            else
            {
                byTimestamp.emplace(record.timestamp, record);
            }
        }
    }

    std::vector<Record> records;
    records.reserve(byTimestamp.size());
    for (const auto& item : byTimestamp)
    {
        Record resolved = item.second;
        if (options.resolveSiblingTcpCloud && !fs::exists(resolved.cloudPath))
        {
            const fs::path siblingTcpCloud = dataDir.parent_path() / "TCP" / resolved.cloudPath.filename();
            if (fs::exists(siblingTcpCloud))
            {
                resolved.cloudPath = siblingTcpCloud;
            }
        }

        if (!options.requireExistingCloud || fs::exists(resolved.cloudPath))
        {
            records.push_back(resolved);
        }
        else if (options.errorStream)
        {
            *options.errorStream << "Skip " << resolved.timestamp << ": missing point cloud "
                                 << resolved.cloudPath.string() << "\n";
        }
    }
    return records;
}

const PoseIo::Record* PoseIo::FindForCloud(const std::vector<Record>& records, const fs::path& cloudPath)
{
    const std::string filename = ToLower(cloudPath.filename().string());
    for (const Record& record : records)
    {
        if (ToLower(record.cloudPath.filename().string()) == filename)
        {
            return &record;
        }
    }

    const std::string timestamp = TimestampFromCloudPath(cloudPath);
    for (const Record& record : records)
    {
        if (record.timestamp == timestamp)
        {
            return &record;
        }
    }
    return nullptr;
}

void PoseIo::WriteSinglePoseFile(
    const fs::path& path,
    const std::string& stamp,
    const fs::path& cloudFile,
    const Pose& flangePose,
    const Quaternion& flangeQuaternion,
    const Pose& tcpPose)
{
    std::ofstream out(path);
    if (!out)
    {
        throw std::runtime_error("Cannot open pose file for writing: " + path.string());
    }

    WritePoseHeader(out);
    WritePoseRow(out, stamp, cloudFile, flangePose, flangeQuaternion, tcpPose);
}

void PoseIo::AppendPoseIndex(
    const fs::path& path,
    const std::string& stamp,
    const fs::path& cloudFile,
    const Pose& flangePose,
    const Quaternion& flangeQuaternion,
    const Pose& tcpPose)
{
    const bool exists = fs::exists(path) && fs::file_size(path) > 0;
    std::ofstream out(path, std::ios::app);
    if (!out)
    {
        throw std::runtime_error("Cannot open pose index for writing: " + path.string());
    }

    if (!exists)
    {
        WritePoseHeader(out);
    }
    WritePoseRow(out, stamp, cloudFile, flangePose, flangeQuaternion, tcpPose);
}

} // namespace handeye
