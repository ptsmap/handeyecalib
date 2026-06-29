#include "app_config.h"
#include "device_session.h"
#include "handeye_auto_calibrator.h"
#include "handeye_calibrator.h"
#include "handeye_workpiece_corner_calibrator.h"
#include "point_cloud_io.h"
#include "pose_io.h"
#include "transform_utils.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace fs = std::filesystem;
using namespace handeye;

namespace
{
std::string NowStamp()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

std::vector<fs::path> CollectInputPcdFiles(const fs::path& inputPath, fs::path& dataDir)
{
    std::vector<fs::path> files;
    if (fs::is_regular_file(inputPath) && inputPath.extension() == ".pcd")
    {
        dataDir = inputPath.parent_path();
        files.push_back(inputPath);
        return files;
    }

    if (!fs::is_directory(inputPath))
    {
        throw std::runtime_error("Input must be a PCD file or directory: " + inputPath.string());
    }

    dataDir = inputPath;
    for (const fs::directory_entry& entry : fs::directory_iterator(inputPath))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".pcd")
        {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

fs::path ResolveDataDir(int argc, char* argv[], const fs::path& exeDir)
{
    if (argc >= 3)
    {
        return argv[2];
    }

    const std::array<fs::path, 4> candidates = {
        fs::current_path() / "data",
        fs::current_path() / ".." / "data",
        exeDir / ".." / ".." / "data",
        exeDir / ".." / "data",
    };
    const auto found = std::find_if(candidates.begin(), candidates.end(), [](const fs::path& path) {
        return fs::exists(path) && fs::is_directory(path);
    });
    return found != candidates.end() ? *found : (fs::current_path() / "data");
}
int RunCameraCloudToBaseMode(const fs::path& inputArg, const fs::path& matrixArg)
{
    const fs::path inputPath = fs::absolute(inputArg);
    const fs::path matrixPath = fs::absolute(matrixArg);

    Transform flangeFromCamera;
    if (!ReadHandEyeMatrix(matrixPath, flangeFromCamera))
    {
        std::cerr << "Hand-eye matrix not found: " << matrixPath.string() << "\n";
        return 1;
    }

    fs::path dataDir;
    const std::vector<fs::path> pcdFiles = CollectInputPcdFiles(inputPath, dataDir);
    if (pcdFiles.empty())
    {
        std::cerr << "No PCD files found: " << inputPath.string() << "\n";
        return 1;
    }

    PoseIo::ReadOptions poseOptions;
    poseOptions.requireExistingCloud = false;
    poseOptions.individualFilesOverrideIndex = true;
    poseOptions.errorStream = &std::cerr;
    const std::vector<PoseIo::Record> poseRecords = PoseIo::ReadPoseRecords(dataDir, poseOptions);
    if (poseRecords.empty())
    {
        std::cerr << "No robot pose records found in: " << dataDir.string() << "\n";
        std::cerr << "Expected robot_records.pose or robot_*.pose next to the input PCD files.\n";
        return 1;
    }

    const fs::path outputDir = inputPath / "base";
    fs::create_directories(outputDir);

    size_t exportedCount = 0;
    size_t skippedCount = 0;
    size_t failedCount = 0;
    std::cout << "Camera-to-base PCD mode\n";
    std::cout << "Input: " << inputPath.string() << "\n";
    std::cout << "Output: " << outputDir.string() << "\n";

    for (const fs::path& pcdPath : pcdFiles)
    {
        const PoseIo::Record* record = PoseIo::FindForCloud(poseRecords, pcdPath);
        if (!record)
        {
            std::cerr << "Skip PCD without matching pose: " << pcdPath.filename().string() << "\n";
            ++skippedCount;
            continue;
        }

        const Transform baseFromCamera = Multiply(ToTransform(record->baseFromFlange), flangeFromCamera);
        const fs::path outputPath = outputDir / pcdPath.filename();
        if (PointCloudIo::TransformPcdFileToBase(pcdPath, outputPath, baseFromCamera, &std::cerr))
        {
            ++exportedCount;
            std::cout << "Wrote: " << outputPath.string() << "\n";
        }
        else
        {
            ++failedCount;
        }
    }

    std::cout << "Exported " << exportedCount << "/" << pcdFiles.size()
              << " PCD files to base coordinates."
              << " skipped=" << skippedCount
              << " failed=" << failedCount << "\n";

    return (exportedCount > 0 && skippedCount == 0 && failedCount == 0) ? 0 : 1;
}

void PrintUsage(const fs::path& exePath)
{
    const std::string exe = exePath.filename().empty() ? "handeye_collector.exe" : exePath.filename().string();
    std::cout << "Usage:\n";
    std::cout << "  " << exe << " calibrate [data_dir]         # Full calibration pipeline\n";
    std::cout << "  " << exe << " autocalib [data_dir]         # Calibration without taught corner.xyz\n";
    std::cout << "  " << exe << " workpiece [data_dir]         # Calibration from a fixed workpiece inner corner\n";
    std::cout << "  " << exe << " multiblock [data_dir] [init]  # Multi-block calibration\n";
    std::cout << "  " << exe << " base <pcd_dir_or_file> [mat]  # Transform PCD to base coords\n";
    std::cout << "  " << exe << " feature [data_dir]            # Feature extraction only\n";
    std::cout << "\n";
    std::cout << "base mode uses exe_dir/handeye_matrix.txt by default and writes to <input_dir>/../base.\n";
}

int RunFeatureExtractionMode(const fs::path& dataDir, const fs::path& exeDir)
{
    FeatureExtractionParams params;
    params.dataDir = dataDir;
    params.exportFeatureCloud = true;
    
    const FeatureExtractionResult result = RunFeatureExtraction(params);
    if (result.code != 0)
    {
        std::cerr << result.message << "\n";
        return 1;
    }

    std::cout << "Feature extraction completed.\n";
    std::cout << "  Valid stations: " << result.stationCount << "\n";
    std::cout << "  Skipped stations: " << result.skippedStationCount << "\n";
    for (size_t i = 0; i < result.stations.size(); ++i)
    {
        const StationFeatureInfo& station = result.stations[i];
        std::cout << "  [" << i << "] " << station.timestamp
                  << " planes=" << station.detectedPlaneCount
                  << "/" << station.mergedPlaneCount
                  << " corners=" << station.cornerCandidates.size();
        if (!station.cornerCandidates.empty())
        {
            const TrihedralCornerInfo& best = station.cornerCandidates.front();
            std::cout << " best_score=" << best.score
                      << " angle_deg=" << best.angleResidualDeg;
        }
        std::cout << "\n";
    }
    return 0;
}


int RunAutoCalibrationMode(const fs::path& dataDir, const fs::path& exeDir, const AppConfig& config)
{
    AutoCalibrateParams params;
    params.dataDir = dataDir;
    params.outputDir = dataDir;
    params.featureParams.dataDir = dataDir;
    params.featureParams.exportFeatureCloud = config.exportFeatureCloud;
    params.exportBaseClouds = true;
    params.baseCloudOutputDir = fs::absolute(dataDir) / "base";
    params.exportFeatureBaseClouds = config.exportFeatureCloud;
    params.featureBaseOutputDir = fs::absolute(dataDir) / "featureBase";

    const HandEyeAutoCalibrator calibrator;
    const AutoCalibrateResult result = calibrator.Run(params);
    if (result.code != 0)
    {
        std::cerr << result.message << "\n";
        return 1;
    }

    std::cout << "Auto calibration completed.\n";
    std::cout << "Report written: " << result.reportPath.string() << "\n";
    std::cout << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < result.estimatedTargetBases.size(); ++i)
    {
        const Eigen::Vector3d& targetBase = result.estimatedTargetBases[i];
        std::cout << "Estimated target corner group " << (i + 1) << " in base (mm): "
                  << targetBase.x() << " "
                  << targetBase.y() << " "
                  << targetBase.z() << "\n";
    }
    std::cout << "RMS error: " << result.rmsErrorMm << " mm\n";
    std::cout << "flangeFromCamera (4x4):\n";
    for (int row = 0; row < 4; ++row)
    {
        std::cout << "  ";
        for (int col = 0; col < 4; ++col)
        {
            if (col > 0)
            {
                std::cout << ' ';
            }
            std::cout << result.flangeFromCamera.matrix()(row, col);
        }
        std::cout << "\n";
    }
    std::cout << "Hand-eye matrix written: " << result.matrixPath.string() << "\n";
    if (config.exportFeatureCloud)
    {
        std::cout << "Base-coordinate feature PCD files written: "
                  << params.featureBaseOutputDir << "\n";
    }
    return 0;
}
int RunWorkpieceCornerCalibrationMode(const fs::path& dataDir, const fs::path& exeDir, const AppConfig& config)
{
    WorkpieceCornerCalibrateParams params;
    params.dataDir = dataDir;
    params.outputDir = dataDir;
    params.featureParams.exportFeatureCloud = config.exportFeatureCloud;
    params.exportBaseClouds = true;
    params.baseCloudOutputDir = fs::absolute(dataDir) / "base_workpiece";

    const HandEyeWorkpieceCornerCalibrator calibrator;
    const WorkpieceCornerCalibrateResult result = calibrator.Run(params);
    if (result.code != 0)
    {
        std::cerr << result.message << "\n";
        return 1;
    }

    std::cout << "Workpiece corner calibration completed.\n";
    std::cout << "Report written: " << result.reportPath.string() << "\n";
    std::cout << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < result.estimatedWorkpieceCornerBases.size(); ++i)
    {
        const Eigen::Vector3d& cornerBase = result.estimatedWorkpieceCornerBases[i];
        std::cout << "Estimated workpiece corner group " << (i + 1) << " in base (mm): "
                  << cornerBase.x() << " "
                  << cornerBase.y() << " "
                  << cornerBase.z() << "\n";
    }
    std::cout << "RMS error: " << result.rmsErrorMm << " mm\n";
    std::cout << "flangeFromCamera (4x4):\n";
    for (int row = 0; row < 4; ++row)
    {
        std::cout << "  ";
        for (int col = 0; col < 4; ++col)
        {
            if (col > 0)
            {
                std::cout << ' ';
            }
            std::cout << result.flangeFromCamera.matrix()(row, col);
        }
        std::cout << "\n";
    }
    std::cout << "Hand-eye matrix written: " << result.matrixPath.string() << "\n";
    return 0;
}

void CaptureOnce(DeviceSession& session, const AppConfig& config, const fs::path& exeDir, int groupIndex)
{
    const std::string stamp = NowStamp();

    std::cout << "Scanning...\n";
    std::cout << "Reading robot pose...\n";
    DeviceCapture capture = session.captureOnce(stamp);

    // Use the new ScanSaveStation API
    ScanSaveParams saveParams;
    saveParams.timestamp = stamp;
    const fs::path groupDir = config.outputDir / ("group_" + std::to_string(groupIndex));
    saveParams.outputDir = groupDir;
    saveParams.cloudData = capture.points.data();
    saveParams.cloudPointCount = static_cast<unsigned int>(capture.points.size());
    //saveParams.cloudDataType = static_cast<int>(capture.type);
    saveParams.flangeTranslation = Eigen::Vector3d(
        capture.flangePose.x, capture.flangePose.y, capture.flangePose.z);
    saveParams.flangeRotation = Eigen::Vector3d(
        capture.flangePose.rx, capture.flangePose.ry, capture.flangePose.rz);
    saveParams.flangeQuaternion = Eigen::Quaterniond(
        capture.flangeQuaternion.w,
        capture.flangeQuaternion.x,
        capture.flangeQuaternion.y,
        capture.flangeQuaternion.z);
    saveParams.tcpTranslation = Eigen::Vector3d(
        capture.tcpPose.x, capture.tcpPose.y, capture.tcpPose.z);
    saveParams.baseCloudOutputDir = config.outputDir.parent_path() / "base" / ("group_" + std::to_string(groupIndex));

    const ScanSaveResult result = SaveScanStation(saveParams);

    if (result.code == 0)
    {
        std::cout << "Saved group " << groupIndex << " scan with " << result.savedPointCount
                  << " points: " << result.cloudPath.string() << "\n";
        if (!result.baseCloudPath.empty())
        {
            std::cout << "Saved base cloud: " << result.baseCloudPath.string() << "\n";
        }
        std::cout << "Saved pose: " << result.posePath.string() << "\n";
    }
    else
    {
        std::cerr << "Save failed: " << result.message << "\n";
    }
}

fs::path ExecutableDir(const char* argv0)
{
    fs::path exePath = fs::absolute(argv0);
    if (fs::exists(exePath))
    {
        return exePath.parent_path();
    }
    return fs::current_path();
}
} // namespace

int main(int argc, char* argv[])
{
    try
    {
        const fs::path exeDir = ExecutableDir(argc > 0 ? argv[0] : ".");
        if (argc >= 2)
        {
            const std::string command = ToLower(Trim(argv[1]));
            if (command == "help" || command == "--help" || command == "-h")
            {
                PrintUsage(argc > 0 ? fs::path(argv[0]) : fs::path("calibAlgo.exe"));
                return 0;
            }

            if (command == "calibrate" || command == "--calibrate" ||
                command == "calib" || command == "--calib")
            {
                const fs::path dataDir = ResolveDataDir(argc, argv, exeDir);

                const fs::path configPath = fs::exists(exeDir / "config.ini")
                    ? (exeDir / "config.ini")
                    : (fs::current_path() / "config.ini");
                const AppConfig config = LoadConfig(configPath);

                // Auto-generate corner.xyz if not present: read robot welder TCP
                const fs::path cornerPath = dataDir / "corner.xyz";
                if (!fs::exists(cornerPath))
                {
                    std::cout << cornerPath.filename().string()
                              << " not found in " << dataDir.string() << "\n";
                    std::cout << "Reading robot welder coordinates...\n";
                    try
                    {
                        DeviceSession session(config);
                        const PoseIo::Pose snapshot = session.readWelderPose();
                        std::ofstream out(cornerPath);
                        if (!out)
                        {
                            std::cerr << "Failed to create " << cornerPath.string() << "\n";
                            return 1;
                        }
                        out << std::fixed << std::setprecision(3)
                            << snapshot.x << " "
                            << snapshot.y << " "
                            << snapshot.z << "\n";
                        std::cout << "Saved welder TCP → " << cornerPath.string() << "\n";
                    }
                    catch (const std::exception& ex)
                    {
                        std::cerr << "Failed to read robot welder: " << ex.what() << "\n";
                        std::cerr << "Please manually create " << cornerPath.string() << "\n";
                        return 1;
                    }
                }
                
                return RunHandEyeCalibration(dataDir, exeDir, config);
            }

            if (command == "autocalib" || command == "--autocalib" ||
                command == "auto-calib" || command == "auto_calib")
            {
                const fs::path dataDir = ResolveDataDir(argc, argv, exeDir);
                const fs::path configPath = fs::exists(exeDir / "config.ini")
                    ? (exeDir / "config.ini")
                    : (fs::current_path() / "config.ini");
                const AppConfig config = LoadConfig(configPath);
                return RunAutoCalibrationMode(dataDir, exeDir, config);
            }
            if (command == "workpiece" || command == "--workpiece" ||
                command == "workpiece-calib" || command == "workpiece_corner" ||
                command == "corner-workpiece")
            {
                const fs::path dataDir = ResolveDataDir(argc, argv, exeDir);
                const fs::path configPath = fs::exists(exeDir / "config.ini")
                    ? (exeDir / "config.ini")
                    : (fs::current_path() / "config.ini");
                const AppConfig config = LoadConfig(configPath);
                return RunWorkpieceCornerCalibrationMode(dataDir, exeDir, config);
            }
            if (command == "multiblock" || command == "--multiblock" ||
                command == "calibrate-multiblock" || command == "multi-block")
            {
                const fs::path dataDir = ResolveDataDir(argc, argv, exeDir);

                const fs::path initialMatrixPath = argc >= 4 ? fs::path(argv[3]) : fs::path();
                return RunHandEyeMultiBlockCalibration(dataDir, exeDir, initialMatrixPath);
            }

            if (command == "feature" || command == "--feature" ||
                command == "extract" || command == "extract-features")
            {
                const fs::path dataDir = ResolveDataDir(argc, argv, exeDir);

                return RunFeatureExtractionMode(dataDir, exeDir);
            }

            if (command == "base" || command == "--base" ||
                command == "camera-to-base" || command == "solve-base" || command == "pcd-base")
            {
                if (argc < 3)
                {
                    PrintUsage(argc > 0 ? fs::path(argv[0]) : fs::path("calibAlgo.exe"));
                    return 1;
                }
                const fs::path matrixPath = argc >= 4 ? fs::path(argv[3]) : (fs::path(argv[2]) / "handeye_matrix.txt");
                return RunCameraCloudToBaseMode(argv[2], matrixPath);
            }
        }

        const fs::path configPath = fs::exists(exeDir / "config.ini")
            ? (exeDir / "config.ini")
            : (fs::current_path() / "config.ini");

        AppConfig config = LoadConfig(configPath);
        
        std::cout << "Hand-eye calibration collector\n";
        std::cout << "Config: " << configPath.string() << "\n";
        std::cout << "Output: " << config.outputDir.string() << "\n";
        std::cout << "Robot profile: " << config.activeRobot << " type=" << config.robotProfile().type << "\n";
        std::cout << "Camera profile: " << config.activeCamera << " type=" << config.cameraProfile().type << "\n";
        std::cout << "Connecting devices...\n";

        DeviceSession session(config);

        config.outputDir = NowStamp() + "/scans";
        fs::create_directories(config.outputDir);

        std::cout << "\nReady. Type group index number + Enter to scan (1=group_1, 2=group_2, ...), q + Enter to quit.\n";
        std::string command;
        while (true)
        {
            std::cout << "> ";
            if (!std::getline(std::cin, command))
            {
                break;
            }

            command = ToLower(Trim(command));
            if (command == "q" || command == "quit" || command == "exit")
            {
                break;
            }

            const bool isGroupCommand = !command.empty()
                && std::all_of(command.begin(), command.end(), [](unsigned char ch) {
                    return std::isdigit(ch) != 0;
                });
            if (isGroupCommand)
            {
                try
                {
                    const int groupIndex = std::stoi(command);
                    if (groupIndex <= 0)
                    {
                        std::cout << "Group index must be >= 1.\n";
                        continue;
                    }
                    CaptureOnce(session, config, exeDir, groupIndex);
                }
                catch (const std::exception& ex)
                {
                    std::cerr << "Capture failed: " << ex.what() << "\n";
                }
                continue;
            }

            if (!command.empty())
            {
                std::cout << "Unknown command. Type a group index number to scan or q to quit.\n";
            }
        }

        std::cout << "Bye.\n";
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Fatal error: " << ex.what() << "\n";
        return 1;
    }
}
