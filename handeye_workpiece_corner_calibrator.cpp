#include "handeye_workpiece_corner_calibrator.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{
std::string ToLowerAscii(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool LooksLikeGroupedDataDir(const fs::path& path)
{
    if (!fs::is_directory(path))
    {
        return false;
    }

    const std::string name = ToLowerAscii(path.filename().string());
    if (name.rfind("group", 0) != 0 && name.rfind("pos", 0) != 0 && name.rfind("position", 0) != 0)
    {
        return false;
    }

    if (fs::exists(path / "robot_records.pose"))
    {
        return true;
    }

    for (const fs::directory_entry& entry : fs::directory_iterator(path))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".pose")
        {
            return true;
        }
    }
    return false;
}

std::vector<fs::path> FindGroupedDataDirs(const fs::path& dataDir)
{
    std::vector<fs::path> dirs;
    for (const fs::directory_entry& entry : fs::directory_iterator(dataDir))
    {
        if (entry.is_directory() && LooksLikeGroupedDataDir(entry.path()))
        {
            dirs.push_back(entry.path());
        }
    }
    std::sort(dirs.begin(), dirs.end(), [](const fs::path& lhs, const fs::path& rhs) {
        return lhs.filename().string() < rhs.filename().string();
    });
    return dirs;
}

void AppendFeatures(
    handeye::FeatureExtractionResult& aggregate,
    const handeye::FeatureExtractionResult& current)
{
    aggregate.stationCount += current.stationCount;
    aggregate.skippedStationCount += current.skippedStationCount;
    aggregate.stations.insert(aggregate.stations.end(), current.stations.begin(), current.stations.end());
    if (aggregate.tcpFilePath.empty())
    {
        aggregate.tcpFilePath = current.tcpFilePath;
    }
}

void LimitInnerCornerCandidates(
    std::vector<handeye::StationFeatureInfo>& stations,
    size_t maxCandidates)
{
    if (maxCandidates == 0)
    {
        return;
    }

    for (handeye::StationFeatureInfo& station : stations)
    {
        if (station.cornerCandidates.size() > maxCandidates)
        {
            station.cornerCandidates.resize(maxCandidates);
        }
    }
}
} // namespace

namespace handeye
{

HandEyeWorkpieceCornerCalibrator::HandEyeWorkpieceCornerCalibrator(WorkpieceCornerCalibrateParams params)
    : params_(std::move(params))
{
}

WorkpieceCornerCalibrateResult HandEyeWorkpieceCornerCalibrator::Run() const
{
    return Run(params_);
}

WorkpieceCornerCalibrateResult HandEyeWorkpieceCornerCalibrator::Run(
    const WorkpieceCornerCalibrateParams& params) const
{
    WorkpieceCornerCalibrateResult result;
    try
    {
        if (params.dataDir.empty())
        {
            result.code = -1;
            result.message = "Data directory is empty";
            return result;
        }

        const fs::path absoluteDataDir = fs::absolute(params.dataDir);
        if (!fs::exists(absoluteDataDir) || !fs::is_directory(absoluteDataDir))
        {
            result.code = -1;
            result.message = "Data directory does not exist: " + absoluteDataDir.string();
            return result;
        }

        FeatureExtractionResult features;
        std::vector<StationFeatureInfo> stations = params.stations;
        std::vector<std::vector<StationFeatureInfo>> stationGroups = params.stationGroups;

        if (!stationGroups.empty())
        {
            features.code = 0;
            features.message = "Using pre-extracted grouped workpiece corner features";
            for (std::vector<StationFeatureInfo>& groupStations : stationGroups)
            {
                LimitInnerCornerCandidates(groupStations, params.maxCandidatesPerStation);
                features.stationCount += static_cast<int>(groupStations.size());
                features.stations.insert(features.stations.end(), groupStations.begin(), groupStations.end());
            }
        }
        else if (stations.empty())
        {
            std::vector<fs::path> dataDirs;
            if (params.useGroupedDataDirs)
            {
                dataDirs = FindGroupedDataDirs(absoluteDataDir);
            }
            if (dataDirs.empty())
            {
                dataDirs.push_back(absoluteDataDir);
            }

            features.code = 0;
            stationGroups.reserve(dataDirs.size());
            for (const fs::path& dataDir : dataDirs)
            {
                FeatureExtractionParams featureParams = params.featureParams;
                featureParams.dataDir = dataDir;
                featureParams.maxCandidatesPerStation = params.maxCandidatesPerStation;
                if (featureParams.featureOutputDir.empty())
                {
                    featureParams.featureOutputDir = absoluteDataDir / "feature_workpiece" / dataDir.filename();
                }

                FeatureExtractionResult current = RunWorkpieceCornerFeatureExtraction(featureParams);
                if (current.code != 0)
                {
                    result.code = current.code;
                    result.message = current.message;
                    result.featureExtraction = current;
                    return result;
                }

                LimitInnerCornerCandidates(current.stations, params.maxCandidatesPerStation);
                stationGroups.push_back(current.stations);
                AppendFeatures(features, current);
            }
            features.message = "Workpiece corner feature extraction completed for "
                + std::to_string(stationGroups.size()) + " group(s)";
        }
        else
        {
            features.code = 0;
            features.message = "Using pre-extracted workpiece corner features";
            LimitInnerCornerCandidates(stations, params.maxCandidatesPerStation);
            features.stationCount = static_cast<int>(stations.size());
            features.stations = stations;
            stationGroups.push_back(stations);
        }

        AutoCalibrateParams autoParams;
        autoParams.dataDir = absoluteDataDir;
        autoParams.stationGroups = stationGroups;
        autoParams.featureParams = params.featureParams;
        autoParams.maxCandidatesPerStation = params.maxCandidatesPerStation;
        autoParams.candidateBeamWidth = params.candidateBeamWidth;
        autoParams.minStationCount = params.minStationCount;
        autoParams.cornerHuberLossMm = params.cornerHuberLossMm;
        autoParams.cornerMaxIterations = params.cornerMaxIterations;
        autoParams.cornerMinimizerProgressToStdout = params.cornerMinimizerProgressToStdout;
        autoParams.candidateGeometryScoreWeight = params.candidateGeometryScoreWeight;
        autoParams.maxStationErrorMm = params.maxStationErrorMm;
        autoParams.outputDir = params.outputDir.empty() ? absoluteDataDir : params.outputDir;
        autoParams.writeOutputs = params.writeOutputs;
        autoParams.exportBaseClouds = params.exportBaseClouds;
        autoParams.baseCloudOutputDir = params.baseCloudOutputDir.empty()
            ? absoluteDataDir / "base_workpiece"
            : params.baseCloudOutputDir;
        autoParams.exportFeatureBaseClouds = false;

        const HandEyeAutoCalibrator autoCalibrator;
        AutoCalibrateResult autoResult = autoCalibrator.Run(autoParams);
        result = WorkpieceCornerCalibrateResult{};
        static_cast<AutoCalibrateResult&>(result) = std::move(autoResult);
        result.featureExtraction = features;
        result.estimatedWorkpieceCornerBases = result.estimatedTargetBases;
        result.estimatedWorkpieceCornerBase = result.estimatedWorkpieceCornerBases.empty()
            ? Eigen::Vector3d::Zero()
            : result.estimatedWorkpieceCornerBases.front();

        if (result.code == 0)
        {
            std::ostringstream message;
            message << "Workpiece corner calibration completed: groups="
                << result.estimatedWorkpieceCornerBases.size()
                << ", rms=" << result.rmsErrorMm << " mm";
            result.message = message.str();
        }
    }
    catch (const std::exception& ex)
    {
        result.code = -99;
        result.message = std::string("HandEyeWorkpieceCornerCalibrator exception: ") + ex.what();
    }
    return result;
}

} // namespace handeye
