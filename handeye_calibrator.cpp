#include "handeye_calibrator.h"
#include "app_config.h"
//#include "robot_device.h"
#include "point_cloud_io.h"
#include "pose_io.h"
#include "handeye_multi_block_plane_optimizer.h"
#include "handeye_nonlinear_optimizer.h"
#include "handeye_plane_optimizer.h"
#include "handeye_trihedral_bundle_optimizer.h"
#include "fastGICP/inc/fast_gicp.hpp"

#include <RansacShapeDetector.h>
#include <PlanePrimitiveShape.h>
#include <PlanePrimitiveShapeConstructor.h>
#include <CylinderPrimitiveShape.h>
#include <CylinderPrimitiveShapeConstructor.h>
#include <SpherePrimitiveShape.h>
#include <SpherePrimitiveShapeConstructor.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <pcl/console/time.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/common/pca.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;
using handeye::PointCloudIo;
using handeye::PoseIo;

namespace
{

typedef pcl::PointCloud<pcl::PointNormal>::Ptr PTRN;
typedef pcl::PointCloud<pcl::PointNormal> PCDN;
typedef pcl::PointCloud<pcl::Normal>::Ptr NPTR;
typedef pcl::PointCloud<pcl::Normal> NORMAL;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PTR;
typedef pcl::PointCloud<pcl::PointXYZ> PCD;
typedef pcl::PointCloud<pcl::PointXYZI>::Ptr PTRI;
typedef pcl::PointCloud<pcl::PointXYZI> PCDI;

constexpr float F_MAX = (std::numeric_limits<float>::max)();
constexpr float F_MIN = -(std::numeric_limits<float>::max)();

constexpr double kPi = 3.14159265358979323846;
constexpr double kOrthogonalToleranceDeg = 8.0;
constexpr double kCentroidDistanceMm = 100.0;
constexpr double kCentroidDistanceToleranceMm = 15.0;

using PoseRecord = PoseIo::Record;

struct HeaderTable
{
    std::vector<std::string> names;
    std::map<std::string, size_t> index;
};

struct PlaneObservation
{
    Eigen::Vector3d normal = Eigen::Vector3d::UnitX();
    double d = 0.0;
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    size_t pointCount = 0;
    std::vector<Eigen::Vector3d> points;
};

struct PlaneFitResult {
    int id = -1;
    float a = 0.0;
    float b = 0.0;
    float c = 1.0;
    float d = 0.0;
    float dX = 0.0;
    float dY = 0.0;
    float hullArea = 0.0;
    size_t pointCount = 0;
    std::vector<Eigen::Vector3d> points;
    PlaneFitResult& operator*=(float scalar) {
        if (scalar == -1.0f)
        {
            a *= scalar;
            b *= scalar;
            c *= scalar;
            d *= scalar;
        }
        return *this;
    }
    PTR inliers_cloud;
    PTR inliers_cloud_densy;
    PTR hull_cloud;
    PTRN plane_cloud_normal;
    pcl::PointXYZ centroid;
    pcl::PointXYZ intersect;
    Eigen::Matrix4f transMat;

    float getWidthScale() const
    {
        return std::min(dX, dY) / std::max(dX, dY);
    }
};

struct CornerCandidate
{
    bool usesMergedPlanes = false;
    std::array<int, 3> planeIndices{};
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    double score = std::numeric_limits<double>::max();
    size_t supportPointCount = 0;
    double angleResidualDeg = 0.0;
    double centroidPairResidualMm = 0.0;
    double centroidCornerResidualMm = 0.0;
    bool centroidDistanceOk = false;
};

struct StationCandidates
{
    std::string timestamp;
    fs::path cloudPath;
    Eigen::Isometry3d baseFromFlange = Eigen::Isometry3d::Identity();
    int detectedPlaneCount = 0;
    int mergedPlaneCount = 0;
    std::vector<PlaneObservation> planes;
    std::vector<PlaneObservation> mergedPlanes;
    std::vector<CornerCandidate> candidates;
};

enum class CornerExtractionMode
{
    GeometryScore,
    WorkpieceSupportPointCount,
};

struct FeatureObservation
{
    std::string timestamp;
    fs::path cloudPath;
    Eigen::Isometry3d baseFromFlange = Eigen::Isometry3d::Identity();
    int targetIndex = 0;
    Eigen::Vector3d cameraPoint = Eigen::Vector3d::Zero();
    std::array<int, 3> planeIndices{};
    std::array<PlaneObservation, 3> featurePlanes;
    double candidateScore = 0.0;
    bool centroidDistanceOk = false;
};

struct CalibrationResult
{
    Eigen::Isometry3d flangeFromCamera = Eigen::Isometry3d::Identity();
    std::vector<Eigen::Vector3d> basePredictions;
    std::vector<Eigen::Vector3d> errors;
    std::string optimizerBriefReport;
    double optimizerInitialCost = 0.0;
    double optimizerFinalCost = 0.0;
    double meanError = 0.0;
    double rmsError = 0.0;
    double maxError = 0.0;
    bool planeOptimizationAttempted = true;
    bool planeOptimizationSuccess = false;
    size_t planeResidualCount = 0;
    std::string planeOptimizerBriefReport;
    double planeOptimizerInitialCost = 0.0;
    double planeOptimizerFinalCost = 0.0;
    double planeMeanAbsDistanceMm = 0.0;
    double planeRmsDistanceMm = 0.0;
    double planeMaxAbsDistanceMm = 0.0;
    bool trihedralOptimizationAttempted = false;
    bool trihedralOptimizationSuccess = false;
    size_t trihedralStationCount = 0;
    size_t trihedralPlaneResidualCount = 0;
    size_t trihedralCornerResidualCount = 0;
    std::string trihedralOptimizerBriefReport;
    double trihedralOptimizerInitialCost = 0.0;
    double trihedralOptimizerFinalCost = 0.0;
    double trihedralPlaneMeanAbsDistanceMm = 0.0;
    double trihedralPlaneRmsDistanceMm = 0.0;
    double trihedralPlaneMaxAbsDistanceMm = 0.0;
    Eigen::Vector3d trihedralCornerBase = Eigen::Vector3d::Zero();
    Eigen::Vector3d trihedralTargetAlignmentTranslationDeltaMm = Eigen::Vector3d::Zero();
    bool trihedralCornerBaseAvailable = false;
    double trihedralCornerMeanErrorMm = 0.0;
    double trihedralCornerRmsErrorMm = 0.0;
    double trihedralCornerMaxErrorMm = 0.0;
    size_t targetBaseCount = 1;
    std::vector<Eigen::Vector3d> targetBases;
    double trihedralTargetAlignmentRotationDeltaDeg = 0.0;
    bool trihedralTargetPoseOptimized = false;
};

int computeOBB(PTR pcd, float cube_size[])
{
    if (!pcd || pcd->size() < 2) {
        return 0;
    }
    pcl::PCA<pcl::PointXYZ> pca;
    PTR pcd_xoy(new PCD);
    *pcd_xoy = *pcd;
    for (auto& pt : pcd_xoy->points)
    {
        pt.z = 0;
    }
    pca.setInputCloud(pcd_xoy);

    Eigen::Vector4f centroid = pca.getMean();
    Eigen::Matrix3f eigen_vectors = pca.getEigenVectors();
    Eigen::Vector3f min_pca(FLT_MAX, FLT_MAX, FLT_MAX);
    Eigen::Vector3f max_pca(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    Eigen::Vector3f min_pt(FLT_MAX, FLT_MAX, FLT_MAX);
    Eigen::Vector3f max_pt(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (const auto& point : pcd->points) {
        Eigen::Vector3f p(point.x, point.y, point.z);
        Eigen::Vector3f centered_p = p - centroid.head<3>();
        Eigen::Vector3f pca_coords = eigen_vectors.transpose() * centered_p;

        for (int i = 0; i < 3; ++i) {
            min_pca[i] = (std::min)(min_pca[i], pca_coords[i]);
            max_pca[i] = (std::max)(max_pca[i], pca_coords[i]);
        }
        for (int i = 0; i < 3; ++i) {
            min_pt[i] = (std::min)(min_pt[i], p[i]);
            max_pt[i] = (std::max)(max_pt[i], p[i]);
        }
    }
    cube_size[0] = fabs(max_pca.x() - min_pca.x());
    cube_size[1] = fabs(max_pca.y() - min_pca.y());
    cube_size[2] = fabs(max_pca.z() - min_pca.z());
    return 1;
}

PTR downSampling(const PTR cloud_in, float leaf_size)
{
    if (!cloud_in || cloud_in->empty())
    {
        return 0;
    }

    PTR cloud_out(new PCD);
    pcl::VoxelGrid<pcl::PointXYZ> vox_grid;
    vox_grid.setInputCloud(cloud_in);
    vox_grid.setLeafSize(leaf_size, leaf_size, leaf_size);
    vox_grid.filter(*cloud_out);

    return cloud_out;
}

enum RANSAC_PRIMITIVE_TYPES
{
    RPT_PLANE = 0,
    RPT_SPHERE = 1,
    RPT_CYLINDER = 2,
    RPT_CONE = 3,
    RPT_TORUS = 4,
};

typedef std::pair< MiscLib::RefCountPtr< PrimitiveShape >, size_t > DetectedShape;
int rdDetection(PTR cloud, MiscLib::Vector<DetectedShape>& shapes, PointCloud& rsd_cloud,
    int minPoints = 200, double epsilon = 5.0, double bitmapEpsilon = 10.0)
{
    if (!cloud || cloud->size() < 200)
    {
        return 0;
    }

    int count = cloud->size();
    rsd_cloud.clear();
    rsd_cloud.reserve(cloud->size());
    const float f_max = std::numeric_limits<float>::max();
    pcl::PointXYZ cbbMin(f_max, f_max, f_max), cbbMax(-f_max, -f_max, -f_max);
    Point rsd_pt;
    rsd_pt.normal[0] = 0.0;
    rsd_pt.normal[1] = 0.0;
    rsd_pt.normal[2] = 0.0;
    for (size_t i = 0; i < cloud->size(); i++)
    {
        const pcl::PointXYZ& pt = cloud->points[i];
        cbbMin.x = std::min(cbbMin.x, pt.x);
        cbbMin.y = std::min(cbbMin.y, pt.y);
        cbbMin.z = std::min(cbbMin.z, pt.z);
        cbbMax.x = std::max(cbbMax.x, pt.x);
        cbbMax.y = std::max(cbbMax.y, pt.y);
        cbbMax.z = std::max(cbbMax.z, pt.z);
        rsd_pt.pos.setValue(pt.x, pt.y, pt.z);
        rsd_pt.index = i;
        rsd_cloud.push_back(rsd_pt);
    }
    rsd_cloud.setBBox(Vec3f(cbbMin.x, cbbMin.y, cbbMin.z), Vec3f(cbbMax.x, cbbMax.y, cbbMax.z));

    const float scale = rsd_cloud.getScale();
    rsd_cloud.calcNormals(0.01 * scale);
    PCL_INFO("calcNormals:%lf\n", .01f * scale);

    RansacShapeDetector::Options option;
    option.m_minSupport = minPoints;
    option.m_normalThresh = static_cast<float>(cos(DEG2RAD(25)));
    option.m_probability = 0.01;
    option.m_epsilon = epsilon;//.004 * scale;
    option.m_bitmapEpsilon = bitmapEpsilon;// .08 * scale;
    option.m_allowSimplification = true;
    RansacShapeDetector detector(option);
    detector.Add(new PlanePrimitiveShapeConstructor());
    shapes.clear();
    detector.Detect(rsd_cloud, 0, rsd_cloud.size(), &shapes);

    // PCL_INFO("detected planes:%zd\n", shapes.size());

    return shapes.size() > 0 ? 1 : 0;
}

int doDetection(PTR cloud_sample, /*PTR cloud,*/ std::vector<PlaneFitResult>& out_planes, pcl::PointXYZ origin,
    int minPoints = 200, double epsilon = 5.0, double bitmapEpsilon = 10.0,
    double plane_angle_tol = 25.0, double concave_alpha = 0.01)
{
    pcl::console::TicToc tt;
    tt.tic();
    MiscLib::Vector<DetectedShape> shapes;

    PointCloud rsd_cloud;
    rdDetection(cloud_sample, shapes, rsd_cloud, minPoints, epsilon, bitmapEpsilon);
    PCL_WARN("detected planes:%zd ,use time:%lf\n", shapes.size(), tt.toc());
    tt.tic();

    if (shapes.size() < 2)
    {
        return 0;
    }

    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud_sample, centroid);

    int count = cloud_sample->size();
    out_planes.clear();

    int planeId = 0;
#ifdef _DEBUG_PCD
    PTR planes_cloud(new PCD);
#endif
    for (MiscLib::Vector<DetectedShape>::const_iterator it = shapes.begin(); it != shapes.end(); ++it)
    {
        const PrimitiveShape* shape = it->first;
        unsigned shapePointsCount = static_cast<unsigned>(it->second);
        if (count <= 0 || shapePointsCount > static_cast<unsigned>(count))
        {
            PCL_WARN("Skipping inconsistent detected shape: support=%u remaining=%d\n", shapePointsCount, count);
            break;
        }
        if (shape->Identifier() != RPT_PLANE)
        {
            count -= shapePointsCount;
            continue;
        }
        if (shapePointsCount < minPoints)
        {
            count -= shapePointsCount;
            continue;
        }

        const PlanePrimitiveShape* plane = static_cast<const PlanePrimitiveShape*>(shape);
        if (!plane)
        {
            count -= shapePointsCount;
            continue;
        }

        int shapeCloudIndex = count - 1;
        PlaneFitResult fit_plane;
        fit_plane.a = plane->Internal().getNormal()[0];
        fit_plane.b = plane->Internal().getNormal()[1];
        fit_plane.c = plane->Internal().getNormal()[2];
        fit_plane.d = -plane->Internal().SignedDistToOrigin();

        float signedDist = origin.x * fit_plane.a + origin.y * fit_plane.b + origin.z * fit_plane.c + fit_plane.d;
        if (signedDist < 0.0f)
        {
            fit_plane *= -1.0f;
        }

        fit_plane.pointCount = shapePointsCount;
        fit_plane.points.reserve(shapePointsCount);
        fit_plane.centroid.getVector3fMap().setZero();
        Vec3f X = plane->getXDim();
        Vec3f Y = plane->getYDim();
        Vec3f G = plane->Internal().getPosition();
        Vec3f N = plane->Internal().getNormal();

        //we look for real plane extents
        float minX, maxX, minY, maxY, minZ, maxZ;
        for (unsigned j = 0; j < shapePointsCount; ++j)
        {
            std::pair<float, float> param;
            plane->Parameters(rsd_cloud[shapeCloudIndex - j].pos, &param);
            if (j != 0)
            {
                minX = (std::min)(minX, param.first);
                maxX = (std::max)(maxX, param.first);
                minY = (std::min)(minY, param.second);
                maxY = (std::max)(maxY, param.second);
            }
            else
            {
                minX = maxX = param.first;
                minY = maxY = param.second;
            }
        }

        //we recenter plane (as it is not always the case!)
        fit_plane.dX = maxX - minX;
        fit_plane.dY = maxY - minY;
        G += X * (minX + fit_plane.dX / 2);
        G += Y * (minY + fit_plane.dY / 2);

        //we build matrix from these vectors
        Eigen::Matrix4f transMat;
        transMat << X[0], Y[0], N[0], G[0],
            X[1], Y[1], N[1], G[1],
            X[2], Y[2], N[2], G[2],
            0, 0, 0, 1;
        fit_plane.centroid.getVector3fMap() = transMat.block<3, 1>(0, 3);
        fit_plane.transMat = transMat;
        minX = maxX = minY = maxY = minZ = maxZ = 0.0f;
        pcl::PointXYZ point;
        // pcl::PointNormal pointN;
        point.data[3] = 1.0f;
        for (int j = 0; j < shapePointsCount; ++j)
        {
            point.x = rsd_cloud[shapeCloudIndex - j].pos[0];
            point.y = rsd_cloud[shapeCloudIndex - j].pos[1];
            point.z = rsd_cloud[shapeCloudIndex - j].pos[2];
            fit_plane.points.emplace_back(
                static_cast<double>(point.x),
                static_cast<double>(point.y),
                static_cast<double>(point.z));
            // RANSAC inliers are counted by shapePointsCount.
            if (j != 0)
            {
                maxX = std::max(maxX, point.x);
                minX = std::min(minX, point.x);
                maxY = std::max(maxY, point.y);
                minY = std::min(minY, point.y);
                maxZ = std::max(maxZ, point.z);
                minZ = std::min(minZ, point.z);
            }
            else
            {
                minX = maxX = point.x;
                minY = maxY = point.y;
                minZ = maxZ = point.z;
            }
        }

        //fit_plane.inliers_cloud->clear();
        //fit_plane.inliers_cloud->reserve(shapePointsCount * 3);
        //for (const auto& pt : cloud->points)
        //{
        //    if (pt.x < minX || pt.y < minY || pt.z < minZ ||
        //        pt.x > maxX || pt.y > maxY || pt.z > maxZ)
        //    {
        //        continue;
        //    }
        //    float dist = pt.x * fit_plane.a + pt.y * fit_plane.b + pt.z * fit_plane.c + fit_plane.d;
        //    //epsilon,bitmapEpsilon
        //    if (dist < bitmapEpsilon * 1.2 && dist > -bitmapEpsilon * 1.2)
        //    {
        //        fit_plane.inliers_cloud->push_back(pt);
        //    }
        //}
#ifdef _DEBUG_PCD
        * planes_cloud += *fit_plane.inliers_cloud;
#endif
        /*fit_plane.inliers_cloud_densy = fit_plane.inliers_cloud;
        fit_plane.inliers_cloud.reset(new PCD);
        *fit_plane.inliers_cloud = *fit_plane.inliers_cloud_densy;
        downSampling(fit_plane.inliers_cloud, 1.0f);*/

        fit_plane.id = planeId;
        out_planes.push_back(fit_plane);
        count -= shapePointsCount;
        planeId++;

    }
    PCL_INFO("extract planes:%zd ,use time:%lf\n", out_planes.size(), tt.toc());

    /*for (size_t k = 0; k < out_planes.size(); k++)
    {
        const PlaneFitResult& cur_plane = out_planes[k];
        PCL_INFO("plane_%zd:%f,%f,%f,%f\n", k, cur_plane.a, cur_plane.b, cur_plane.c, cur_plane.d);
    }*/
    return 1;
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
        header.index[header.names[i]] = i;
    }
    return header;
}

std::optional<double> FieldDouble(const HeaderTable& header, const std::vector<std::string>& row, const std::string& name)
{
    const auto it = header.index.find(name);
    if (it == header.index.end() || it->second >= row.size())
    {
        PCL_ERROR("Missing field: %s\n", name.c_str());
        return std::nullopt;
    }

    try
    {
        return std::stod(row[it->second]);
    }
    catch (const std::exception&)
    {
        PCL_ERROR("Invalid numeric field %s: %s\n", name.c_str(), row[it->second].c_str());
        return std::nullopt;
    }
}

std::optional<std::string> FieldString(const HeaderTable& header, const std::vector<std::string>& row, const std::string& name)
{
    const auto it = header.index.find(name);
    if (it == header.index.end() || it->second >= row.size())
    {
        PCL_ERROR("Missing field: %s\n", name.c_str());
        return std::nullopt;
    }
    return row[it->second];
}
std::vector<Eigen::Vector3d> ReadTargetTcps(const fs::path& tcpPath)
{
    std::ifstream in(tcpPath);
    if (!in)
    {
        PCL_ERROR("Cannot open tcp file: %s\n", tcpPath.string().c_str());
        return {};
    }

    std::vector<Eigen::Vector3d> targets;
    HeaderTable header;
    bool haveHeader = false;
    std::string line;
    while (std::getline(in, line))
    {
        line = Trim(line);
        if (line.empty())
        {
            continue;
        }

        if (line.find("timestamp") != std::string::npos)
        {
            header = ParseHeader(line);
            haveHeader = true;
            continue;
        }

        const std::vector<std::string> row = SplitWhitespace(line);
        if ((!haveHeader && row.size() < 3) || row.empty())
        {
            continue;
        }

        const auto tcpX = haveHeader ? FieldDouble(header, row, "tcp_x") : std::stod(row[0]);
        const auto tcpY = haveHeader ? FieldDouble(header, row, "tcp_y") : std::stod(row[1]);
        const auto tcpZ = haveHeader ? FieldDouble(header, row, "tcp_z") : std::stod(row[2]);
        if (!tcpX || !tcpY || !tcpZ)
        {
            PCL_WARN("Skip invalid TCP row in %s: %s\n", tcpPath.string().c_str(), line.c_str());
            continue;
        }
        targets.emplace_back(*tcpX, *tcpY, *tcpZ);
    }

    if (targets.empty())
    {
        PCL_ERROR("TCP file has no data row: %s\n", tcpPath.string().c_str());
    }
    return targets;
}

std::optional<Eigen::Vector3d> ReadTargetTcp(const fs::path& tcpPath)
{
    const std::vector<Eigen::Vector3d> targets = ReadTargetTcps(tcpPath);
    if (targets.empty())
    {
        return std::nullopt;
    }
    return targets.front();
}

Eigen::Vector3d ToVector(const pcl::PointXYZ& point)
{
    return Eigen::Vector3d(point.x, point.y, point.z);
}

bool IsOrthogonal(const Eigen::Vector3d& a, const Eigen::Vector3d& b, double toleranceDeg = kOrthogonalToleranceDeg)
{
    const double dot = std::abs(a.normalized().dot(b.normalized()));
    return dot <= std::sin(toleranceDeg * kPi / 180.0);
}

double AngleResidualDeg(const Eigen::Vector3d& a, const Eigen::Vector3d& b)
{
    const double dot = std::clamp(std::abs(a.normalized().dot(b.normalized())), 0.0, 1.0);
    return std::abs(90.0 - std::acos(dot) * 180.0 / kPi);
}

bool IsExpectedCentroidDistance(double distance, double expectedMm = kCentroidDistanceMm, double toleranceMm = kCentroidDistanceToleranceMm)
{
    return std::abs(distance - expectedMm) <= toleranceMm;
}

std::optional<Eigen::Vector3d> IntersectPlanes(
    const PlaneObservation& a,
    const PlaneObservation& b,
    const PlaneObservation& c)
{
    Eigen::Matrix3d matrix;
    matrix.row(0) = a.normal.transpose();
    matrix.row(1) = b.normal.transpose();
    matrix.row(2) = c.normal.transpose();

    const double det = matrix.determinant();
    if (std::abs(det) < 1e-8)
    {
        return std::nullopt;
    }

    const Eigen::Vector3d rhs(-a.d, -b.d, -c.d);
    return matrix.colPivHouseholderQr().solve(rhs);
}

double PlaneDistance(const PlaneObservation& a, const PlaneObservation& b)
{
    const double same = std::abs(a.d - b.d);
    const double opposite = std::abs(a.d + b.d);
    return std::min(same, opposite);
}

std::vector<PlaneObservation> MergeSimilarPlanes(const std::vector<PlaneObservation>& planes,
    double mergeAngleDeg = 5.0, double mergeDistanceMm = 12.0)
{
    struct Group
    {
        Eigen::Vector3d normalSum = Eigen::Vector3d::Zero();
        Eigen::Vector3d weightedCentroid = Eigen::Vector3d::Zero();
        double weightedD = 0.0;
        size_t pointCount = 0;
        int partCount = 0;
        std::vector<Eigen::Vector3d> points;
    };

    const double mergeCos = std::cos(mergeAngleDeg * kPi / 180.0);

    std::vector<Group> groups;
    for (const PlaneObservation& plane : planes)
    {
        bool merged = false;
        for (Group& group : groups)
        {
            if (group.pointCount == 0)
            {
                continue;
            }

            Eigen::Vector3d groupNormal = group.normalSum.normalized();
            Eigen::Vector3d normal = plane.normal;
            double d = plane.d;
            if (groupNormal.dot(normal) < 0.0)
            {
                normal = -normal;
                d = -d;
            }

            if (groupNormal.dot(normal) >= mergeCos && std::abs(group.weightedD / static_cast<double>(group.pointCount) - d) <= mergeDistanceMm)
            {
                const double weight = static_cast<double>(plane.pointCount);
                group.normalSum += normal * weight;
                group.weightedCentroid += plane.centroid * weight;
                group.weightedD += d * weight;
                group.pointCount += plane.pointCount;
                group.partCount += 1;
                group.points.insert(group.points.end(), plane.points.begin(), plane.points.end());
                merged = true;
                break;
            }
        }

        if (!merged)
        {
            Group group;
            const double weight = static_cast<double>(plane.pointCount);
            group.normalSum = plane.normal * weight;
            group.weightedCentroid = plane.centroid * weight;
            group.weightedD = plane.d * weight;
            group.pointCount = plane.pointCount;
            group.partCount = 1;
            group.points = plane.points;
            groups.push_back(group);
        }
    }

    std::vector<PlaneObservation> merged;
    merged.reserve(groups.size());
    for (const Group& group : groups)
    {
        if (group.pointCount == 0)
        {
            continue;
        }
        PlaneObservation plane;
        const double invWeight = 1.0 / static_cast<double>(group.pointCount);
        plane.normal = group.normalSum.normalized();
        plane.centroid = group.weightedCentroid * invWeight;
        plane.d = group.weightedD * invWeight;
        plane.pointCount = group.pointCount;
        plane.points = group.points;
        merged.push_back(plane);
    }

    std::sort(merged.begin(), merged.end(), [](const PlaneObservation& lhs, const PlaneObservation& rhs) {
        return lhs.pointCount > rhs.pointCount;
    });
    return merged;
}

std::vector<PlaneObservation> FilterPlanesWithOrthogonalMate(const std::vector<PlaneObservation>& planes,
    double orthogonalToleranceDeg = kOrthogonalToleranceDeg, double maxCentroidDistanceMm = 200.0)
{
    std::vector<PlaneObservation> filtered;
    filtered.reserve(planes.size());
    for (size_t i = 0; i < planes.size(); ++i)
    {
        bool hasOrthogonalMate = false;
        for (size_t j = 0; j < planes.size(); ++j)
        {
            if (i == j)
            {
                continue;
            }
            if (IsOrthogonal(planes[i].normal, planes[j].normal, orthogonalToleranceDeg) && (planes[i].centroid - planes[j].centroid).norm() < maxCentroidDistanceMm)
            {
                hasOrthogonalMate = true;
                break;
            }
        }

        if (hasOrthogonalMate)
        {
            filtered.push_back(planes[i]);
        }
    }
    return filtered;
}
std::vector<CornerCandidate> BuildCornerCandidates(const std::vector<PlaneObservation>& planes, bool usesMergedPlanes,
    double orthoToleranceDeg = kOrthogonalToleranceDeg,
    double expectedCentroidDistanceMm = kCentroidDistanceMm,
    double centroidDistanceToleranceMm = kCentroidDistanceToleranceMm,
    size_t maxCandidates = 120)
{
    std::vector<CornerCandidate> candidates;

    for (size_t i = 0; i < planes.size(); ++i)
    {
        for (size_t j = i + 1; j < planes.size(); ++j)
        {
            for (size_t k = j + 1; k < planes.size(); ++k)
            {
                const PlaneObservation& p0 = planes[i];
                const PlaneObservation& p1 = planes[j];
                const PlaneObservation& p2 = planes[k];
                if (!IsOrthogonal(p0.normal, p1.normal, orthoToleranceDeg) ||
                    !IsOrthogonal(p0.normal, p2.normal, orthoToleranceDeg) ||
                    !IsOrthogonal(p1.normal, p2.normal, orthoToleranceDeg))
                {
                    continue;
                }

                const auto intersection = IntersectPlanes(p0, p1, p2);
                if (!intersection)
                {
                    continue;
                }

                const std::array<double, 3> pairDistances = {
                    (p0.centroid - p1.centroid).norm(),
                    (p0.centroid - p2.centroid).norm(),
                    (p1.centroid - p2.centroid).norm(),
                };
                const std::array<double, 3> cornerDistances = {
                    (p0.centroid - *intersection).norm(),
                    (p1.centroid - *intersection).norm(),
                    (p2.centroid - *intersection).norm(),
                };

                const bool pairDistanceOk =
                    IsExpectedCentroidDistance(pairDistances[0], expectedCentroidDistanceMm, centroidDistanceToleranceMm) &&
                    IsExpectedCentroidDistance(pairDistances[1], expectedCentroidDistanceMm, centroidDistanceToleranceMm) &&
                    IsExpectedCentroidDistance(pairDistances[2], expectedCentroidDistanceMm, centroidDistanceToleranceMm);
                const bool cornerDistanceOk =
                    IsExpectedCentroidDistance(cornerDistances[0], expectedCentroidDistanceMm, centroidDistanceToleranceMm) &&
                    IsExpectedCentroidDistance(cornerDistances[1], expectedCentroidDistanceMm, centroidDistanceToleranceMm) &&
                    IsExpectedCentroidDistance(cornerDistances[2], expectedCentroidDistanceMm, centroidDistanceToleranceMm);

                CornerCandidate candidate;
                candidate.usesMergedPlanes = usesMergedPlanes;
                candidate.planeIndices = {
                    static_cast<int>(i),
                    static_cast<int>(j),
                    static_cast<int>(k),
                };
                candidate.point = *intersection;
                candidate.supportPointCount = p0.pointCount + p1.pointCount + p2.pointCount;
                candidate.angleResidualDeg =
                    AngleResidualDeg(p0.normal, p1.normal) +
                    AngleResidualDeg(p0.normal, p2.normal) +
                    AngleResidualDeg(p1.normal, p2.normal);
                candidate.centroidPairResidualMm =
                    std::abs(pairDistances[0] - expectedCentroidDistanceMm) +
                    std::abs(pairDistances[1] - expectedCentroidDistanceMm) +
                    std::abs(pairDistances[2] - expectedCentroidDistanceMm);
                candidate.centroidCornerResidualMm =
                    std::abs(cornerDistances[0] - expectedCentroidDistanceMm) +
                    std::abs(cornerDistances[1] - expectedCentroidDistanceMm) +
                    std::abs(cornerDistances[2] - expectedCentroidDistanceMm);
                candidate.centroidDistanceOk = pairDistanceOk || cornerDistanceOk;
                const double sizeResidual = std::min(candidate.centroidPairResidualMm, candidate.centroidCornerResidualMm);
                candidate.score = candidate.angleResidualDeg + 0.02 * sizeResidual;
                if (!candidate.centroidDistanceOk)
                {
                    candidate.score += 10.0;
                }
                candidates.push_back(candidate);
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const CornerCandidate& lhs, const CornerCandidate& rhs) {
        if (lhs.centroidDistanceOk != rhs.centroidDistanceOk)
        {
            return lhs.centroidDistanceOk > rhs.centroidDistanceOk;
        }
        return lhs.score < rhs.score;
    });

    if (maxCandidates > 0 && candidates.size() > maxCandidates)
    {
        candidates.resize(maxCandidates);
    }
    return candidates;
}

std::vector<CornerCandidate> BuildWorkpieceCornerCandidates(
    const std::vector<PlaneObservation>& planes,
    bool usesMergedPlanes,
    double orthoToleranceDeg = kOrthogonalToleranceDeg,
    double expectedCentroidDistanceMm = kCentroidDistanceMm,
    double centroidDistanceToleranceMm = kCentroidDistanceToleranceMm,
    size_t maxCandidates = 120)
{
    std::vector<CornerCandidate> candidates = BuildCornerCandidates(
        planes,
        usesMergedPlanes,
        orthoToleranceDeg,
        expectedCentroidDistanceMm,
        centroidDistanceToleranceMm,
        0);

    std::sort(candidates.begin(), candidates.end(), [](const CornerCandidate& lhs, const CornerCandidate& rhs) {
        if (lhs.supportPointCount != rhs.supportPointCount)
        {
            return lhs.supportPointCount > rhs.supportPointCount;
        }
        if (lhs.angleResidualDeg != rhs.angleResidualDeg)
        {
            return lhs.angleResidualDeg < rhs.angleResidualDeg;
        }
        return lhs.score < rhs.score;
    });

    if (maxCandidates > 0 && candidates.size() > maxCandidates)
    {
        candidates.resize(maxCandidates);
    }
    return candidates;
}
StationCandidates ExtractStationCandidates(const PoseRecord& record,
    const handeye::FeatureExtractionParams& params = handeye::FeatureExtractionParams{},
    CornerExtractionMode mode = CornerExtractionMode::GeometryScore)
{
    PCL_INFO("extract station: %s corners\n", record.timestamp.c_str());
    StationCandidates station;
    station.timestamp = record.timestamp;
    station.cloudPath = record.cloudPath;
    station.baseFromFlange = record.baseFromFlange;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    if (!PointCloudIo::LoadPointXYZ(record.cloudPath, *cloud, &std::cerr))
    {
        return station;
    }

    std::vector<int> validIndices;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, validIndices);
    if (cloud->empty())
    {
        PCL_ERROR("Point cloud has no valid XYZ points: %s\n", record.cloudPath.string().c_str());
        return station;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_sample(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> vox_grid;
    vox_grid.setInputCloud(cloud);
    vox_grid.setLeafSize(1.0, 1.0, 1.0);
    vox_grid.filter(*cloud_sample);
    if (cloud_sample->empty())
    {
        PCL_ERROR("Downsampled point cloud is empty: %s\n", record.cloudPath.string().c_str());
        return station;
    }

    std::vector<PlaneFitResult> detectedPlanes;
    const int detectionOk = doDetection(cloud_sample, detectedPlanes, pcl::PointXYZ(),
        params.ransacMinPoints, params.ransacEpsilon, params.ransacBitmapEpsilon,
        params.planeAngleTolerance, 0.01);
    if (detectionOk == 0 || detectedPlanes.size() < 3)
    {
        PCL_ERROR("Detected fewer than 3 planes in %s\n", record.cloudPath.string().c_str());
        station.detectedPlaneCount = static_cast<int>(detectedPlanes.size());
        return station;
    }

    std::vector<PlaneObservation> planes;
    planes.reserve(detectedPlanes.size());
    for (const PlaneFitResult& shape : detectedPlanes)
    {
        if (shape.pointCount == 0)
        {
            PCL_WARN("Skip plane with zero support in %s\n", record.cloudPath.string().c_str());
            continue;
        }

        if (std::fabs(shape.dX) > params.planeMaxExtentMm || std::fabs(shape.dY) > params.planeMaxExtentMm ||
            std::fabs(shape.dX) < params.planeMinExtentMm || std::fabs(shape.dY) < params.planeMinExtentMm)
        {
            continue;
        }

        Eigen::Vector3d normal(
            static_cast<double>(shape.a),
            static_cast<double>(shape.b),
            static_cast<double>(shape.c));
        const double normalNorm = normal.norm();
        if (normalNorm <= std::numeric_limits<double>::epsilon())
        {
            PCL_WARN("Skip plane with invalid normal in %s\n", record.cloudPath.string().c_str());
            continue;
        }
        normal /= normalNorm;
        double d = static_cast<double>(shape.d) / normalNorm;
        if (d < 0.0)
        {
            normal = -normal;
            d = -d;
        }

        PlaneObservation planeObs;
        planeObs.centroid = shape.centroid.getVector3fMap().cast<double>();
        planeObs.normal = normal;
        planeObs.d = d;
        planeObs.pointCount = shape.pointCount;
        planeObs.points = shape.points;
        planes.push_back(planeObs);
    }

    station.detectedPlaneCount = static_cast<int>(planes.size());
    if (planes.size() < 3)
    {
        PCL_ERROR("Fewer than 3 usable planes in %s\n", record.cloudPath.string().c_str());
        return station;
    }

    std::vector<PlaneObservation> mergedPlanes = MergeSimilarPlanes(planes,
        params.mergeAngleDeg, params.mergeDistanceMm);
    station.mergedPlaneCount = static_cast<int>(mergedPlanes.size());
    std::vector<CornerCandidate> candidates;
    std::vector<CornerCandidate> rawCandidates;
    if (mode == CornerExtractionMode::WorkpieceSupportPointCount)
    {
        candidates = BuildWorkpieceCornerCandidates(mergedPlanes, true,
            params.orthogonalToleranceDeg,
            params.expectedCentroidDistanceMm, params.centroidDistanceToleranceMm,
            0);
        rawCandidates = BuildWorkpieceCornerCandidates(planes, false,
            params.orthogonalToleranceDeg,
            params.expectedCentroidDistanceMm, params.centroidDistanceToleranceMm,
            0);
    }
    else
    {
        candidates = BuildCornerCandidates(mergedPlanes, true,
            params.orthogonalToleranceDeg,
            params.expectedCentroidDistanceMm, params.centroidDistanceToleranceMm,
            0);
        rawCandidates = BuildCornerCandidates(planes, false,
            params.orthogonalToleranceDeg,
            params.expectedCentroidDistanceMm, params.centroidDistanceToleranceMm,
            0);
    }
    candidates.insert(candidates.end(), rawCandidates.begin(), rawCandidates.end());
    if (mode == CornerExtractionMode::WorkpieceSupportPointCount)
    {
        std::sort(candidates.begin(), candidates.end(), [](const CornerCandidate& lhs, const CornerCandidate& rhs) {
            if (lhs.supportPointCount != rhs.supportPointCount)
            {
                return lhs.supportPointCount > rhs.supportPointCount;
            }
            if (lhs.angleResidualDeg != rhs.angleResidualDeg)
            {
                return lhs.angleResidualDeg < rhs.angleResidualDeg;
            }
            return lhs.score < rhs.score;
        });
    }
    else
    {
        std::sort(candidates.begin(), candidates.end(), [](const CornerCandidate& lhs, const CornerCandidate& rhs) {
            if (lhs.centroidDistanceOk != rhs.centroidDistanceOk)
            {
                return lhs.centroidDistanceOk > rhs.centroidDistanceOk;
            }
            return lhs.score < rhs.score;
        });
    }
    /*for (const auto& corner : candidates)
    {
        PCL_INFO("%f,%f,%f\n", corner.point.x(), corner.point.y(), corner.point.z());
    }*/
    candidates.erase(std::unique(candidates.begin(), candidates.end(), [](const CornerCandidate& lhs, const CornerCandidate& rhs) {
        return (lhs.point - rhs.point).norm() < 1.0;
    }), candidates.end());
    if (params.maxCandidatesPerStation > 0 && candidates.size() > params.maxCandidatesPerStation)
    {
        candidates.resize(params.maxCandidatesPerStation);
    }
    if (candidates.empty())
    {
        PCL_WARN("Cannot find three mutually perpendicular planes in %s\n", record.cloudPath.string().c_str());
        return station;
    }
    PCL_INFO("found %zd corners:\n", candidates.size());
    for (const auto& corner : candidates)
    {
        PCL_INFO("points:%zd,%f,%f,%f\n", corner.supportPointCount,
            corner.point.x(), corner.point.y(), corner.point.z());
    }
    PCL_INFO("\n");
    station.planes = std::move(planes);
    station.mergedPlanes = std::move(mergedPlanes);
    station.candidates = std::move(candidates);
    return station;
}


void AddFeaturePoint(
    pcl::PointCloud<pcl::PointXYZI>& cloud,
    const Eigen::Vector3d& point,
    float id,
    const Eigen::Isometry3d* outputFromCamera = nullptr)
{
    const Eigen::Vector3d outputPoint = outputFromCamera ? ((*outputFromCamera) * point) : point;
    pcl::PointXYZI out;
    out.x = static_cast<float>(outputPoint.x());
    out.y = static_cast<float>(outputPoint.y());
    out.z = static_cast<float>(outputPoint.z());
    out.intensity = id;
    cloud.push_back(out);
}

bool ExportFeatureCloud(
    const StationCandidates& station,
    const fs::path& featureDir,
    const Eigen::Isometry3d* outputFromCamera = nullptr)
{
    constexpr size_t kMaxExportTrihedrals = 5;
    if (station.candidates.empty())
    {
        return true;
    }

    pcl::PointCloud<pcl::PointXYZI> featureCloud;
    featureCloud.reserve(4096);
    const size_t trihedralCount = std::min(kMaxExportTrihedrals, station.candidates.size());
    for (size_t trihedralIndex = 0; trihedralIndex < trihedralCount; ++trihedralIndex)
    {
        const CornerCandidate& candidate = station.candidates[trihedralIndex];
        const std::vector<PlaneObservation>& sourcePlanes = candidate.usesMergedPlanes ? station.mergedPlanes : station.planes;
        const int trihedralId = static_cast<int>(trihedralIndex) + 1;
        for (size_t faceIndex = 0; faceIndex < candidate.planeIndices.size(); ++faceIndex)
        {
            const int planeIndex = candidate.planeIndices[faceIndex];
            if (planeIndex < 0 || static_cast<size_t>(planeIndex) >= sourcePlanes.size())
            {
                continue;
            }

            const float id = static_cast<float>(trihedralId * 10 + static_cast<int>(faceIndex) + 1);
            const PlaneObservation& plane = sourcePlanes[static_cast<size_t>(planeIndex)];
            for (const Eigen::Vector3d& point : plane.points)
            {
                AddFeaturePoint(featureCloud, point, id, outputFromCamera);
            }
        }

        AddFeaturePoint(featureCloud, candidate.point, static_cast<float>(trihedralId * 100), outputFromCamera);
    }

    if (featureCloud.empty())
    {
        return true;
    }

    featureCloud.width = static_cast<uint32_t>(featureCloud.size());
    featureCloud.height = 1;
    featureCloud.is_dense = false;
    fs::create_directories(featureDir);
    fs::path outputPath = featureDir / station.cloudPath.filename();
    if (outputPath.empty() || outputPath.filename().empty())
    {
        outputPath = featureDir / (station.timestamp + ".pcd");
    }
    return PointCloudIo::WritePointXYZI(outputPath, featureCloud, &std::cerr);
}

bool ExportFeatureCloudsToBase(
    const std::vector<StationCandidates>& stations,
    const CalibrationResult& result,
    const fs::path& featureBaseDir)
{
    bool allOk = true;
    size_t exportedCount = 0;
    for (const StationCandidates& station : stations)
    {
        const Eigen::Isometry3d baseFromCamera = station.baseFromFlange * result.flangeFromCamera;
        if (ExportFeatureCloud(station, featureBaseDir, &baseFromCamera))
        {
            ++exportedCount;
        }
        else
        {
            allOk = false;
        }
    }

    PCL_INFO("Exported %zu/%zu base-coordinate feature PCD files to %s\n",
        exportedCount, stations.size(), featureBaseDir.string().c_str());
    return allOk;
}
FeatureObservation MakeFeatureObservation(
    const StationCandidates& station,
    const CornerCandidate& candidate,
    int targetIndex = 0)
{
    FeatureObservation feature;
    feature.timestamp = station.timestamp;
    feature.cloudPath = station.cloudPath;
    feature.baseFromFlange = station.baseFromFlange;
    feature.targetIndex = targetIndex;
    feature.cameraPoint = candidate.point;
    feature.planeIndices = candidate.planeIndices;
    const std::vector<PlaneObservation>& sourcePlanes = candidate.usesMergedPlanes ? station.mergedPlanes : station.planes;
    for (size_t i = 0; i < feature.planeIndices.size(); ++i)
    {
        const int planeIndex = feature.planeIndices[i];
        if (planeIndex >= 0 && static_cast<size_t>(planeIndex) < sourcePlanes.size())
        {
            feature.featurePlanes[i] = sourcePlanes[static_cast<size_t>(planeIndex)];
        }
        else
        {
            PCL_WARN("Invalid feature plane index %d for station %s\n", planeIndex, station.timestamp.c_str());
        }
    }
    feature.candidateScore = candidate.score;
    feature.centroidDistanceOk = candidate.centroidDistanceOk;
    return feature;
}

std::optional<Eigen::Isometry3d> EstimateFlangeFromCamera(
    const std::vector<FeatureObservation>& observations,
    const std::vector<Eigen::Vector3d>& targetBases)
{
    if (observations.size() < 3)
    {
        PCL_ERROR("At least 3 valid stations are required for 6D calibration.\n");
        return std::nullopt;
    }
    if (targetBases.empty())
    {
        PCL_ERROR("No target base point for 6D calibration.\n");
        return std::nullopt;
    }

    std::vector<Eigen::Vector3d> cameraPoints;
    std::vector<Eigen::Vector3d> flangePoints;
    cameraPoints.reserve(observations.size());
    flangePoints.reserve(observations.size());
    for (const FeatureObservation& observation : observations)
    {
        if (observation.targetIndex < 0 || static_cast<size_t>(observation.targetIndex) >= targetBases.size())
        {
            continue;
        }
        cameraPoints.push_back(observation.cameraPoint);
        flangePoints.push_back(observation.baseFromFlange.inverse() * targetBases[static_cast<size_t>(observation.targetIndex)]);
    }

    if (cameraPoints.size() < 3)
    {
        PCL_ERROR("At least 3 valid target-indexed observations are required for 6D calibration.\n");
        return std::nullopt;
    }

    Eigen::Vector3d cameraMean = Eigen::Vector3d::Zero();
    Eigen::Vector3d flangeMean = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < cameraPoints.size(); ++i)
    {
        cameraMean += cameraPoints[i];
        flangeMean += flangePoints[i];
    }
    cameraMean /= static_cast<double>(cameraPoints.size());
    flangeMean /= static_cast<double>(flangePoints.size());

    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < cameraPoints.size(); ++i)
    {
        covariance += (cameraPoints[i] - cameraMean) * (flangePoints[i] - flangeMean).transpose();
    }

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
    if (svd.info() != Eigen::Success)
    {
        PCL_ERROR("SVD failed while estimating hand-eye transform.\n");
        return std::nullopt;
    }

    Eigen::Matrix3d v = svd.matrixV();
    const Eigen::Matrix3d u = svd.matrixU();
    Eigen::Matrix3d rotation = v * u.transpose();
    if (rotation.determinant() < 0.0)
    {
        v.col(2) *= -1.0;
        rotation = v * u.transpose();
    }

    Eigen::Isometry3d flangeFromCamera = Eigen::Isometry3d::Identity();
    flangeFromCamera.linear() = rotation;
    flangeFromCamera.translation() = flangeMean - rotation * cameraMean;
    return flangeFromCamera;
}
std::optional<Eigen::Isometry3d> EstimateFlangeFromCamera(
    const std::vector<FeatureObservation>& observations,
    const Eigen::Vector3d& targetBase)
{
    return EstimateFlangeFromCamera(observations, std::vector<Eigen::Vector3d>{targetBase});
}


void RecomputeCornerMetrics(
    const std::vector<FeatureObservation>& observations,
    const std::vector<Eigen::Vector3d>& targetBases,
    CalibrationResult& result)
{
    result.basePredictions.clear();
    result.errors.clear();
    result.basePredictions.reserve(observations.size());
    result.errors.reserve(observations.size());
    result.meanError = 0.0;
    result.rmsError = 0.0;
    result.maxError = 0.0;

    double sum = 0.0;
    double sumSq = 0.0;
    for (const FeatureObservation& observation : observations)
    {
        if (observation.targetIndex < 0 || static_cast<size_t>(observation.targetIndex) >= targetBases.size())
        {
            continue;
        }
        const Eigen::Vector3d flangePoint = result.flangeFromCamera * observation.cameraPoint;
        const Eigen::Vector3d basePrediction = observation.baseFromFlange * flangePoint;
        const Eigen::Vector3d error = basePrediction - targetBases[static_cast<size_t>(observation.targetIndex)];
        const double norm = error.norm();
        result.basePredictions.push_back(basePrediction);
        result.errors.push_back(error);
        sum += norm;
        sumSq += norm * norm;
        result.maxError = std::max(result.maxError, norm);
    }

    if (!result.errors.empty())
    {
        result.meanError = sum / static_cast<double>(result.errors.size());
        result.rmsError = std::sqrt(sumSq / static_cast<double>(result.errors.size()));
    }
}
void RecomputeCornerMetrics(
    const std::vector<FeatureObservation>& observations,
    const Eigen::Vector3d& targetBase,
    CalibrationResult& result)
{
    result.basePredictions.clear();
    result.errors.clear();
    result.basePredictions.reserve(observations.size());
    result.errors.reserve(observations.size());
    result.meanError = 0.0;
    result.rmsError = 0.0;
    result.maxError = 0.0;

    double sum = 0.0;
    double sumSq = 0.0;
    for (const FeatureObservation& observation : observations)
    {
        const Eigen::Vector3d flangePoint = result.flangeFromCamera * observation.cameraPoint;
        const Eigen::Vector3d basePrediction = observation.baseFromFlange * flangePoint;
        const Eigen::Vector3d error = basePrediction - targetBase;
        const double norm = error.norm();
        result.basePredictions.push_back(basePrediction);
        result.errors.push_back(error);
        sum += norm;
        sumSq += norm * norm;
        result.maxError = std::max(result.maxError, norm);
    }

    if (!observations.empty())
    {
        result.meanError = sum / static_cast<double>(observations.size());
        result.rmsError = std::sqrt(sumSq / static_cast<double>(observations.size()));
    }
}

HandEyePlaneOptimizer::PlaneObservation ToPlaneOptimizerObservation(const PlaneObservation& plane)
{
    HandEyePlaneOptimizer::PlaneObservation out;
    out.normal = plane.normal;
    out.d = plane.d;
    out.points = plane.points;
    return out;
}

std::vector<HandEyePlaneOptimizer::StationObservation> BuildPlaneOptimizerStations(
    const std::vector<FeatureObservation>& observations)
{
    std::vector<HandEyePlaneOptimizer::StationObservation> stations;
    stations.reserve(observations.size());
    for (const FeatureObservation& observation : observations)
    {
        HandEyePlaneOptimizer::StationObservation station;
        station.timestamp = observation.timestamp;
        station.baseFromFlange = observation.baseFromFlange;
        bool valid = true;
        for (size_t i = 0; i < observation.featurePlanes.size(); ++i)
        {
            station.planes[i] = ToPlaneOptimizerObservation(observation.featurePlanes[i]);
            if (station.planes[i].points.empty())
            {
                valid = false;
            }
        }
        if (valid)
        {
            stations.push_back(std::move(station));
        }
        else
        {
            PCL_WARN("Skip station for plane optimization because plane points are empty: %s\n",
                observation.timestamp.c_str());
        }
    }
    return stations;
}

void ApplyPlaneOptimizationResult(
    const std::vector<FeatureObservation>& observations,
    const Eigen::Vector3d& targetBase,
    const HandEyePlaneOptimizer::Result& planeResult,
    CalibrationResult& result)
{
    result.planeOptimizationAttempted = planeResult.attempted;
    result.planeOptimizationSuccess = planeResult.success;
    result.planeResidualCount = planeResult.residualCount;
    result.planeOptimizerBriefReport = planeResult.briefReport;
    result.planeOptimizerInitialCost = planeResult.initialCost;
    result.planeOptimizerFinalCost = planeResult.finalCost;
    result.planeMeanAbsDistanceMm = planeResult.meanAbsDistanceMm;
    result.planeRmsDistanceMm = planeResult.rmsDistanceMm;
    result.planeMaxAbsDistanceMm = planeResult.maxAbsDistanceMm;

    if (!planeResult.success)
    {
        return;
    }

    result.flangeFromCamera = planeResult.flangeFromCamera;
    RecomputeCornerMetrics(observations, targetBase, result);
}

HandEyeTrihedralBundleOptimizer::PlaneObservation ToTrihedralBundleObservation(const PlaneObservation& plane)
{
    HandEyeTrihedralBundleOptimizer::PlaneObservation out;
    out.normal = plane.normal;
    out.d = plane.d;
    out.points = plane.points;
    return out;
}

std::vector<HandEyeTrihedralBundleOptimizer::StationObservation> BuildTrihedralBundleStations(
    const std::vector<FeatureObservation>& observations)
{
    std::vector<HandEyeTrihedralBundleOptimizer::StationObservation> stations;
    stations.reserve(observations.size());
    for (const FeatureObservation& observation : observations)
    {
        HandEyeTrihedralBundleOptimizer::StationObservation station;
        station.timestamp = observation.timestamp;
        station.baseFromFlange = observation.baseFromFlange;
        station.targetIndex = observation.targetIndex;
        station.cornerCamera = observation.cameraPoint;
        bool valid = true;
        for (size_t i = 0; i < observation.featurePlanes.size(); ++i)
        {
            station.planes[i] = ToTrihedralBundleObservation(observation.featurePlanes[i]);
            if (station.planes[i].points.empty())
            {
                valid = false;
            }
        }
        if (valid)
        {
            stations.push_back(std::move(station));
        }
        else
        {
            PCL_WARN("Skip station for trihedral bundle because plane points are empty: %s\n",
                observation.timestamp.c_str());
        }
    }
    return stations;
}

void ApplyTrihedralBundleResult(
    const std::vector<FeatureObservation>& observations,
    const std::vector<Eigen::Vector3d>& targetBases,
    const HandEyeTrihedralBundleOptimizer::Result& trihedralResult,
    CalibrationResult& result)
{
    result.trihedralOptimizationAttempted = trihedralResult.attempted;
    result.trihedralOptimizationSuccess = trihedralResult.success;
    result.trihedralStationCount = trihedralResult.stationCount;
    result.trihedralPlaneResidualCount = trihedralResult.planeResidualCount;
    result.trihedralCornerResidualCount = trihedralResult.cornerResidualCount;
    result.trihedralOptimizerBriefReport = trihedralResult.briefReport;
    result.trihedralOptimizerInitialCost = trihedralResult.initialCost;
    result.trihedralOptimizerFinalCost = trihedralResult.finalCost;
    result.trihedralPlaneMeanAbsDistanceMm = trihedralResult.planeMeanAbsDistanceMm;
    result.trihedralPlaneRmsDistanceMm = trihedralResult.planeRmsDistanceMm;
    result.trihedralPlaneMaxAbsDistanceMm = trihedralResult.planeMaxAbsDistanceMm;
    result.trihedralCornerBase = trihedralResult.baseFromBlock.translation();
    result.trihedralTargetAlignmentTranslationDeltaMm = trihedralResult.targetAlignmentTranslationDeltaMm;
    result.trihedralTargetAlignmentRotationDeltaDeg = trihedralResult.targetAlignmentRotationDeltaDeg;
    result.trihedralTargetPoseOptimized = trihedralResult.targetPoseOptimized;
    result.trihedralCornerBaseAvailable = trihedralResult.success;
    result.trihedralCornerMeanErrorMm = trihedralResult.cornerMeanErrorMm;
    result.trihedralCornerRmsErrorMm = trihedralResult.cornerRmsErrorMm;
    result.trihedralCornerMaxErrorMm = trihedralResult.cornerMaxErrorMm;

    if (!trihedralResult.success)
    {
        return;
    }

    result.flangeFromCamera = trihedralResult.flangeFromCamera;
    RecomputeCornerMetrics(observations, targetBases, result);
}
void ApplyTrihedralBundleResult(
    const std::vector<FeatureObservation>& observations,
    const Eigen::Vector3d& targetBase,
    const HandEyeTrihedralBundleOptimizer::Result& trihedralResult,
    CalibrationResult& result)
{
    result.trihedralOptimizationAttempted = trihedralResult.attempted;
    result.trihedralOptimizationSuccess = trihedralResult.success;
    result.trihedralStationCount = trihedralResult.stationCount;
    result.trihedralPlaneResidualCount = trihedralResult.planeResidualCount;
    result.trihedralCornerResidualCount = trihedralResult.cornerResidualCount;
    result.trihedralOptimizerBriefReport = trihedralResult.briefReport;
    result.trihedralOptimizerInitialCost = trihedralResult.initialCost;
    result.trihedralOptimizerFinalCost = trihedralResult.finalCost;
    result.trihedralPlaneMeanAbsDistanceMm = trihedralResult.planeMeanAbsDistanceMm;
    result.trihedralPlaneRmsDistanceMm = trihedralResult.planeRmsDistanceMm;
    result.trihedralPlaneMaxAbsDistanceMm = trihedralResult.planeMaxAbsDistanceMm;
    result.trihedralCornerBase = trihedralResult.baseFromBlock.translation();
    result.trihedralTargetAlignmentTranslationDeltaMm = trihedralResult.targetAlignmentTranslationDeltaMm;
    result.trihedralCornerBaseAvailable = trihedralResult.success;
    result.trihedralCornerMeanErrorMm = trihedralResult.cornerMeanErrorMm;
    result.trihedralCornerRmsErrorMm = trihedralResult.cornerRmsErrorMm;
    result.trihedralCornerMaxErrorMm = trihedralResult.cornerMaxErrorMm;

    if (!trihedralResult.success)
    {
        return;
    }

    result.flangeFromCamera = trihedralResult.flangeFromCamera;
    RecomputeCornerMetrics(observations, targetBase, result);
}

std::optional<CalibrationResult> Calibrate(
    const std::vector<FeatureObservation>& observations,
    const std::vector<Eigen::Vector3d>& targetBases,
    bool enableTrihedralOptimization)
{
    if (targetBases.empty())
    {
        return std::nullopt;
    }

    const auto initialFlangeFromCamera = EstimateFlangeFromCamera(observations, targetBases);
    if (!initialFlangeFromCamera)
    {
        return std::nullopt;
    }

    std::vector<HandEyeNonlinearOptimizer::Observation> optimizerObservations;
    optimizerObservations.reserve(observations.size());
    for (const FeatureObservation& observation : observations)
    {
        if (observation.targetIndex < 0 || static_cast<size_t>(observation.targetIndex) >= targetBases.size())
        {
            continue;
        }
        HandEyeNonlinearOptimizer::Observation optimizerObservation;
        optimizerObservation.baseFromFlange = observation.baseFromFlange;
        optimizerObservation.cameraPoint = observation.cameraPoint;
        optimizerObservation.targetBase = targetBases[static_cast<size_t>(observation.targetIndex)];
        optimizerObservation.weight = 1.0;
        optimizerObservations.push_back(optimizerObservation);
    }

    HandEyeNonlinearOptimizer::Options options;
    options.huberLossMm = 1.0;
    options.maxNumIterations = 1000;
    options.minimizerProgressToStdout = false;
    const HandEyeNonlinearOptimizer optimizer(options);
    const HandEyeNonlinearOptimizer::Result optimized = optimizer.Optimize(
        optimizerObservations,
        *initialFlangeFromCamera);
    if (!optimized.success)
    {
        PCL_WARN("Ceres hand-eye optimization did not report a usable solution: %s\n",
            optimized.briefReport.c_str());
    }

    CalibrationResult result;
    result.flangeFromCamera = optimized.flangeFromCamera;
    result.basePredictions = optimized.basePredictions;
    result.errors = optimized.errors;
    result.optimizerBriefReport = optimized.briefReport;
    result.optimizerInitialCost = optimized.initialCost;
    result.optimizerFinalCost = optimized.finalCost;
    result.meanError = optimized.meanError;
    result.rmsError = optimized.rmsError;
    result.maxError = optimized.maxError;
    result.targetBaseCount = targetBases.size();
    result.targetBases = targetBases;

    PCL_INFO("hand-eye optimization: success=%d observations=%zu "
             "initial_cost=%.4f final_cost=%.4f "
             "mean_error_mm=%.4f rms_error_mm=%.4f max_error_mm=%.4f\n",
             optimized.success ? 1 : 0,
             optimizerObservations.size(),
             optimized.initialCost,
             optimized.finalCost,
             optimized.meanError,
             optimized.rmsError,
             optimized.maxError);

    if (enableTrihedralOptimization)
    {
        HandEyeTrihedralBundleOptimizer::Options trihedralOptions;
        trihedralOptions.maxCornerErrorForTrihedralStageMm = 5.0;
        trihedralOptions.planeHuberLossMm = 0.5;
        trihedralOptions.targetAlignmentHuberLossMm = 0.5;
        trihedralOptions.targetAlignmentSigmaMm = 0.5;
        trihedralOptions.minMatchedNormalDot = 0.94;
        trihedralOptions.maxNumIterations = 5000;
        trihedralOptions.minimizerProgressToStdout = false;
        const HandEyeTrihedralBundleOptimizer trihedralOptimizer(trihedralOptions);
        const std::vector<HandEyeTrihedralBundleOptimizer::StationObservation> trihedralStations =
            BuildTrihedralBundleStations(observations);
        const bool optimizeTargetPose = targetBases.size() >= 2;
        const HandEyeTrihedralBundleOptimizer::Result trihedralResult = trihedralOptimizer.Optimize(
            trihedralStations,
            targetBases,
            result.flangeFromCamera,
            result.maxError,
            optimizeTargetPose);
        if (trihedralResult.attempted)
        {
            PCL_INFO("Trihedral bundle optimization: success=%d stations=%zu plane_residuals=%zu plane_rms=%lf corner_rms=%lf report=%s\n",
                trihedralResult.success ? 1 : 0,
                trihedralResult.stationCount,
                trihedralResult.planeResidualCount,
                trihedralResult.planeRmsDistanceMm,
                trihedralResult.cornerRmsErrorMm,
                trihedralResult.briefReport.c_str());
            if (trihedralResult.success)
            {
                const Eigen::Vector3d optimizedCornerBase = trihedralResult.baseFromBlock.translation();
                const Eigen::Vector3d targetDelta = trihedralResult.targetAlignmentTranslationDeltaMm;
                PCL_INFO("Optimized cornerBase in base frame (mm): %lf %lf %lf\n",
                    optimizedCornerBase.x(),
                    optimizedCornerBase.y(),
                    optimizedCornerBase.z());
                PCL_INFO("Target alignment hand-eye delta dx/dy/dz/rdeg: %lf %lf %lf %lf\n",
                    targetDelta.x(),
                    targetDelta.y(),
                    targetDelta.z(),
                    trihedralResult.targetAlignmentRotationDeltaDeg);
            }
        }
        ApplyTrihedralBundleResult(observations, targetBases, trihedralResult, result);
    }
    return result;
}
std::optional<CalibrationResult> Calibrate(
    const std::vector<FeatureObservation>& observations,
    const Eigen::Vector3d& targetBase,
    bool enableTrihedralOptimization);

std::vector<FeatureObservation> SelectGloballyConsistentObservations(
    const std::vector<StationCandidates>& stations,
    const Eigen::Vector3d& targetBase)
{
    if (stations.size() < 3)
    {
        PCL_ERROR("At least 3 valid stations are required for global candidate selection.\n");
        return {};
    }

    struct BeamState
    {
        std::vector<int> candidateIndices;
        double rmsError = std::numeric_limits<double>::max();
        double score = std::numeric_limits<double>::max();
    };

    std::vector<BeamState> beam(1);
    constexpr size_t kBeamWidth = 10000;
    constexpr size_t kCandidatesPerStation = 2;

    for (size_t stationIndex = 0; stationIndex < stations.size(); ++stationIndex)
    {
        std::vector<BeamState> next;
        const size_t candidateCount = std::min(kCandidatesPerStation, stations[stationIndex].candidates.size());
        if (candidateCount == 0)
        {
            PCL_ERROR("Station %s has no candidates.\n", stations[stationIndex].timestamp.c_str());
            return {};
        }

        for (const BeamState& state : beam)
        {
            for (size_t candidateIndex = 0; candidateIndex < candidateCount; ++candidateIndex)
            {
                BeamState candidateState = state;
                candidateState.candidateIndices.push_back(static_cast<int>(candidateIndex));

                std::vector<FeatureObservation> observations;
                observations.reserve(candidateState.candidateIndices.size());
                for (size_t i = 0; i < candidateState.candidateIndices.size(); ++i)
                {
                    observations.push_back(MakeFeatureObservation(
                        stations[i],
                        stations[i].candidates[static_cast<size_t>(candidateState.candidateIndices[i])]));
                }

                double geometryPenalty = 0.0;
                for (const FeatureObservation& observation : observations)
                {
                    geometryPenalty += observation.candidateScore;
                }
                geometryPenalty /= static_cast<double>(observations.size());

                if (observations.size() >= 3)
                {
                    const auto result = Calibrate(observations, targetBase, false);
                    if (!result)
                    {
                        continue;
                    }
                    candidateState.rmsError = result->rmsError;
                    candidateState.score = result->rmsError;
                }
                else
                {
                    candidateState.rmsError = geometryPenalty;
                    candidateState.score = geometryPenalty;
                }
                next.push_back(std::move(candidateState));
            }
        }

        if (next.empty())
        {
            PCL_ERROR("Global candidate selection produced no solution at station %s.\n", stations[stationIndex].timestamp.c_str());
            return {};
        }

        std::sort(next.begin(), next.end(), [](const BeamState& lhs, const BeamState& rhs) {
            return lhs.score < rhs.score;
        });
        if (next.size() > kBeamWidth)
        {
            next.resize(kBeamWidth);
        }
        beam = std::move(next);
    }

    if (beam.empty())
    {
        PCL_ERROR("Global candidate selection produced no solution.\n");
        return {};
    }

    const BeamState& best = beam.front();
    std::vector<FeatureObservation> observations;
    observations.reserve(stations.size());
    for (size_t i = 0; i < stations.size(); ++i)
    {
        observations.push_back(MakeFeatureObservation(
            stations[i],
            stations[i].candidates[static_cast<size_t>(best.candidateIndices[i])]));
    }
    return observations;
}

std::optional<CalibrationResult> Calibrate(
    const std::vector<FeatureObservation>& observations,
    const Eigen::Vector3d& targetBase,
    bool enableTrihedralOptimization)
{
    return Calibrate(observations, std::vector<Eigen::Vector3d>{targetBase}, enableTrihedralOptimization);
}

std::vector<FeatureObservation> SelectGloballyConsistentMultiTargetObservations(
    const std::vector<StationCandidates>& stations,
    const std::vector<Eigen::Vector3d>& targetBases)
{
    std::vector<FeatureObservation> allObservations;
    for (size_t targetIndex = 0; targetIndex < targetBases.size(); ++targetIndex)
    {
        std::vector<FeatureObservation> observations = SelectGloballyConsistentObservations(
            stations,
            targetBases[targetIndex]);
        if (observations.size() < 3)
        {
            PCL_WARN("No valid observation set for target %zu.\n", targetIndex);
            continue;
        }
        for (FeatureObservation& observation : observations)
        {
            observation.targetIndex = static_cast<int>(targetIndex);
            allObservations.push_back(std::move(observation));
        }
    }
    return allObservations;
}

std::array<double, 3> RpyFromRotation(const Eigen::Matrix3d& r)
{
    double rx = 0.0;
    double ry = 0.0;
    double rz = 0.0;

    const double sy = -r(2, 0);
    if (std::abs(sy) < 1.0 - 1e-12)
    {
        ry = std::asin(sy);
        rx = std::atan2(r(2, 1), r(2, 2));
        rz = std::atan2(r(1, 0), r(0, 0));
    }
    else
    {
        ry = std::copysign(kPi / 2.0, sy);
        rx = std::atan2(-r(0, 1), r(1, 1));
        rz = 0.0;
    }

    return {
        rx * 180.0 / kPi,
        ry * 180.0 / kPi,
        rz * 180.0 / kPi,
    };
}

void PrintVector(std::ostream& out, const Eigen::Vector3d& value)
{
    out << value.x() << ' ' << value.y() << ' ' << value.z();
}


bool ExportObservationCloudsToBase(
    const std::vector<FeatureObservation>& observations,
    const CalibrationResult& result,
    const fs::path& baseDir)
{
    std::error_code ec;
    fs::create_directories(baseDir, ec);
    if (ec)
    {
        PCL_ERROR("Cannot create base cloud directory %s: %s\n", baseDir.string().c_str(), ec.message().c_str());
        return false;
    }

    bool allOk = true;
    size_t exportedCount = 0;
    for (const FeatureObservation& observation : observations)
    {
        fs::path outputPath = baseDir / observation.cloudPath.filename();
        if (outputPath.empty() || outputPath.filename().empty())
        {
            outputPath = baseDir / (observation.timestamp + ".pcd");
        }

        const Eigen::Isometry3d baseFromCamera = observation.baseFromFlange * result.flangeFromCamera;
        if (PointCloudIo::TransformPcdFileToBase(observation.cloudPath, outputPath, baseFromCamera, &std::cerr))
        {
            ++exportedCount;
        }
        else
        {
            allOk = false;
        }
    }

    PCL_INFO("Exported %zu/%zu base-coordinate PCD files to %s\n",
        exportedCount, observations.size(), baseDir.string().c_str());
    return allOk;
}
bool WriteHandEyeMatrix(const Eigen::Isometry3d& flangeFromCamera, const fs::path& matrixPath)
{
    std::ofstream out(matrixPath);
    if (!out)
    {
        PCL_ERROR("Cannot write hand-eye matrix: %s\n", matrixPath.string().c_str());
        return false;
    }

    const Eigen::Matrix4d matrix = flangeFromCamera.matrix();
    out << std::fixed << std::setprecision(6);
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            if (row != 0 || col != 0)
            {
                out << ' ';
            }
            out << matrix(row, col);
        }
    }
    out << '\n';
    return true;
}

void WriteReport(
    std::ostream& out,
    const fs::path& dataDir,
    const Eigen::Vector3d& targetBase,
    const std::vector<FeatureObservation>& observations,
    const CalibrationResult& result)
{
    const Eigen::Matrix3d rotation = result.flangeFromCamera.linear();
    const Eigen::Vector3d translation = result.flangeFromCamera.translation();
    const Eigen::Quaterniond quaternion(rotation);
    const auto rpy = RpyFromRotation(rotation);

    out << std::fixed << std::setprecision(6);
    out << "Hand-eye calibration report\n";
    out << "Data dir: " << dataDir.string() << "\n";
    out << "Target corner in robot base (mm): ";
    PrintVector(out, targetBase);
    out << "\n";
    out << "Target corner count: " << result.targetBaseCount << "\n";
    for (size_t targetIndex = 0; targetIndex < result.targetBases.size(); ++targetIndex)
    {
        out << "target_" << targetIndex << "_base_mm: ";
        PrintVector(out, result.targetBases[targetIndex]);
        out << "\n";
    }
    out << "Valid stations: " << observations.size() << "\n\n";

    out << "T_flange_camera maps camera coordinates to flange coordinates.\n";
    out << "camera_center_in_flange_mm: ";
    PrintVector(out, translation);
    out << "\n";
    out << "quaternion_wxyz: " << quaternion.w() << ' ' << quaternion.x() << ' '
        << quaternion.y() << ' ' << quaternion.z() << "\n";
    out << "rpy_xyz_deg: " << rpy[0] << ' ' << rpy[1] << ' ' << rpy[2] << "\n";
    out << "rotation_matrix:\n";
    for (int row = 0; row < 3; ++row)
    {
        out << "  " << rotation(row, 0) << ' ' << rotation(row, 1) << ' ' << rotation(row, 2) << "\n";
    }
    out << "ceres_brief_report: " << result.optimizerBriefReport << "\n";
    out << "ceres_initial_cost: " << result.optimizerInitialCost << "\n";
    out << "ceres_final_cost: " << result.optimizerFinalCost << "\n";
    out << "plane_optimization_attempted: " << (result.planeOptimizationAttempted ? "yes" : "no") << "\n";
    out << "plane_optimization_success: " << (result.planeOptimizationSuccess ? "yes" : "no") << "\n";
    out << "plane_ceres_brief_report: " << result.planeOptimizerBriefReport << "\n";
    out << "plane_ceres_initial_cost: " << result.planeOptimizerInitialCost << "\n";
    out << "plane_ceres_final_cost: " << result.planeOptimizerFinalCost << "\n";
    out << "plane_residual_count: " << result.planeResidualCount << "\n";
    out << "plane_mean_abs_distance_mm: " << result.planeMeanAbsDistanceMm << "\n";
    out << "plane_rms_distance_mm: " << result.planeRmsDistanceMm << "\n";
    out << "plane_max_abs_distance_mm: " << result.planeMaxAbsDistanceMm << "\n";
    out << "trihedral_optimization_attempted: " << (result.trihedralOptimizationAttempted ? "yes" : "no") << "\n";
    out << "trihedral_optimization_success: " << (result.trihedralOptimizationSuccess ? "yes" : "no") << "\n";
    out << "trihedral_ceres_brief_report: " << result.trihedralOptimizerBriefReport << "\n";
    out << "trihedral_ceres_initial_cost: " << result.trihedralOptimizerInitialCost << "\n";
    out << "trihedral_ceres_final_cost: " << result.trihedralOptimizerFinalCost << "\n";
    out << "trihedral_station_count: " << result.trihedralStationCount << "\n";
    out << "trihedral_plane_residual_count: " << result.trihedralPlaneResidualCount << "\n";
    out << "trihedral_corner_residual_count: " << result.trihedralCornerResidualCount << "\n";
    out << "trihedral_plane_mean_abs_distance_mm: " << result.trihedralPlaneMeanAbsDistanceMm << "\n";
    out << "trihedral_plane_rms_distance_mm: " << result.trihedralPlaneRmsDistanceMm << "\n";
    out << "trihedral_plane_max_abs_distance_mm: " << result.trihedralPlaneMaxAbsDistanceMm << "\n";
    out << "trihedral_target_alignment_translation_delta_mm: ";
    PrintVector(out, result.trihedralTargetAlignmentTranslationDeltaMm);
    out << "\n";
    out << "trihedral_target_alignment_translation_delta_norm_mm: "
        << result.trihedralTargetAlignmentTranslationDeltaMm.norm() << "\n";
    out << "trihedral_target_alignment_rotation_delta_deg: " << result.trihedralTargetAlignmentRotationDeltaDeg << "\n";
    out << "trihedral_target_pose_optimized: " << (result.trihedralTargetPoseOptimized ? "yes" : "no") << "\n";
    out << "trihedral_corner_base_available: " << (result.trihedralCornerBaseAvailable ? "yes" : "no") << "\n";
    if (result.trihedralCornerBaseAvailable)
    {
        out << "trihedral_corner_base_mm: ";
        PrintVector(out, result.trihedralCornerBase);
        out << "\n";
        out << "trihedral_corner_base_delta_from_teaching_mm: ";
        PrintVector(out, result.trihedralCornerBase - targetBase);
        out << "\n";
        out << "trihedral_corner_base_delta_norm_mm: " << (result.trihedralCornerBase - targetBase).norm() << "\n";
    }
    out << "trihedral_corner_mean_error_mm: " << result.trihedralCornerMeanErrorMm << "\n";
    out << "trihedral_corner_rms_error_mm: " << result.trihedralCornerRmsErrorMm << "\n";
    out << "trihedral_corner_max_error_mm: " << result.trihedralCornerMaxErrorMm << "\n";

    out << "\nPer-station corner error in robot base (mm):\n";
    out << "timestamp camera_x camera_y camera_z predicted_x predicted_y predicted_z "
        << "error_x error_y error_z error_norm target_index planes candidate_score centroid_distance_ok\n";
    for (size_t i = 0; i < observations.size(); ++i)
    {
        const FeatureObservation& observation = observations[i];
        out << observation.timestamp << ' ';
        PrintVector(out, observation.cameraPoint);
        out << ' ';
        PrintVector(out, result.basePredictions[i]);
        out << ' ';
        PrintVector(out, result.errors[i]);
        out << ' ' << result.errors[i].norm() << ' ' << observation.targetIndex << ' '
            << observation.planeIndices[0] << ','
            << observation.planeIndices[1] << ','
            << observation.planeIndices[2] << ' '
            << observation.candidateScore << ' '
            << (observation.centroidDistanceOk ? "yes" : "no") << "\n";
    }

    out << "\nmean_error_mm: " << result.meanError << "\n";
    out << "rms_error_mm: " << result.rmsError << "\n";
    out << "max_error_mm: " << result.maxError << "\n";
}

constexpr int kMultiBlockBlockCount = 2;
constexpr int kMultiBlockFacesPerBlock = 2;
constexpr double kMultiBlockMinMatchedNormalDot = 0.70;
constexpr size_t kMultiBlockMaxPairCandidates = 80;
constexpr size_t kMultiBlockMaxTwoBlockCandidates = 160;
constexpr size_t kMultiBlockGicpMinPoints = 80;
constexpr size_t kMultiBlockMaxGicpCandidateTests = 80;
constexpr double kMultiBlockGicpVoxelLeafMm = 3.0;
constexpr double kMultiBlockGicpMaxCorrespondenceDistanceMm = 2.0;
constexpr int kMultiBlockGicpMaxIterations = 10000;
constexpr double kMultiBlockGicpMaxFitnessMm2 = 200000.0;

using MultiBlockNormalGrid = std::array<std::array<Eigen::Vector3d, kMultiBlockFacesPerBlock>, kMultiBlockBlockCount>;
using MultiBlockIndexGrid = std::array<std::array<int, kMultiBlockFacesPerBlock>, kMultiBlockBlockCount>;

struct MultiBlockPlaneStation
{
    std::string timestamp;
    fs::path cloudPath;
    Eigen::Isometry3d baseFromFlange = Eigen::Isometry3d::Identity();
    int detectedPlaneCount = 0;
    int mergedPlaneCount = 0;
    std::vector<PlaneObservation> planes;
    std::vector<PlaneObservation> mergedPlanes;
};

struct MultiBlockPairCandidate
{
    std::array<int, kMultiBlockFacesPerBlock> planeIndices = {0, 1};
    double score = std::numeric_limits<double>::max();
    double angleResidualDeg = 0.0;
    double centroidDistanceResidualMm = 0.0;
    bool centroidDistanceOk = false;
};

struct MultiBlockTwoBlockCandidate
{
    std::array<MultiBlockPairCandidate, kMultiBlockBlockCount> blocks;
    double score = std::numeric_limits<double>::max();
};

struct MultiBlockMatchedStation
{
    HandEyeMultiBlockPlaneOptimizer::StationObservation optimizerStation;
    std::array<std::array<int, kMultiBlockFacesPerBlock>, kMultiBlockBlockCount> sourcePlaneIndices{};
    double score = std::numeric_limits<double>::max();
    double minMatchedNormalDot = 0.0;
    bool matchedByGicp = false;
    double gicpFitnessMm2 = 0.0;
    Eigen::Matrix4f gicpTargetFromSource = Eigen::Matrix4f::Identity();
};

bool ReadHandEyeInitialMatrix(const fs::path& matrixPath, Eigen::Isometry3d& flangeFromCamera)
{
    std::ifstream in(matrixPath);
    if (!in)
    {
        return false;
    }

    std::vector<double> values;
    double value = 0.0;
    while (in >> value)
    {
        values.push_back(value);
    }
    if (values.size() != 16)
    {
        PCL_WARN("Initial hand-eye matrix must contain 16 values: %s\n", matrixPath.string().c_str());
        return false;
    }

    Eigen::Matrix4d matrix;
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            matrix(row, col) = values[static_cast<size_t>(row * 4 + col)];
        }
    }

    flangeFromCamera = Eigen::Isometry3d::Identity();
    flangeFromCamera.linear() = matrix.block<3, 3>(0, 0);
    flangeFromCamera.translation() = matrix.block<3, 1>(0, 3);
    return true;
}

bool IsUsableInitialHandEye(const Eigen::Isometry3d& flangeFromCamera)
{
    if (!flangeFromCamera.matrix().allFinite())
    {
        return false;
    }

    const double rotationIdentityError = (flangeFromCamera.linear() - Eigen::Matrix3d::Identity()).norm();
    const double translationNorm = flangeFromCamera.translation().norm();
    return rotationIdentityError > 1.0e-6 || translationNorm > 1.0e-6;
}
Eigen::Vector3d TransformMultiBlockPlaneNormal(
    const MultiBlockPlaneStation& station,
    const PlaneObservation& plane,
    const Eigen::Isometry3d& flangeFromCamera)
{
    return (station.baseFromFlange.linear() * flangeFromCamera.linear() * plane.normal).normalized();
}

std::optional<MultiBlockPlaneStation> ExtractMultiBlockPlaneStation(const PoseRecord& record)
{
    MultiBlockPlaneStation station;
    station.timestamp = record.timestamp;
    station.cloudPath = record.cloudPath;
    station.baseFromFlange = record.baseFromFlange;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    if (!PointCloudIo::LoadPointXYZ(record.cloudPath, *cloud, &std::cerr))
    {
        return std::nullopt;
    }

    std::vector<int> validIndices;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, validIndices);
    if (cloud->empty())
    {
        PCL_ERROR("Point cloud has no valid XYZ points: %s\n", record.cloudPath.string().c_str());
        return std::nullopt;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudSample(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> voxGrid;
    voxGrid.setInputCloud(cloud);
    voxGrid.setLeafSize(1.0, 1.0, 1.0);
    voxGrid.filter(*cloudSample);
    if (cloudSample->empty())
    {
        PCL_ERROR("Downsampled point cloud is empty: %s\n", record.cloudPath.string().c_str());
        return std::nullopt;
    }

    std::vector<PlaneFitResult> detectedPlanes;
    const int detectionOk = doDetection(cloudSample, detectedPlanes, pcl::PointXYZ(), 100, 1.0, 2.0, 25.0, 0.01);
    station.detectedPlaneCount = static_cast<int>(detectedPlanes.size());
    if (detectionOk == 0 || detectedPlanes.size() < 4)
    {
        PCL_WARN("Detected fewer than 4 planes for multiblock in %s\n", record.cloudPath.string().c_str());
        return std::nullopt;
    }

    for (const PlaneFitResult& shape : detectedPlanes)
    {
        if (shape.pointCount == 0)
        {
            continue;
        }
        if (std::fabs(shape.dX) > 180.0 || std::fabs(shape.dY) > 180.0 ||
            std::fabs(shape.dX) < 45.0 || std::fabs(shape.dY) < 45.0)
        {
            continue;
        }

        Eigen::Vector3d normal(
            static_cast<double>(shape.a),
            static_cast<double>(shape.b),
            static_cast<double>(shape.c));
        const double normalNorm = normal.norm();
        if (normalNorm <= std::numeric_limits<double>::epsilon())
        {
            continue;
        }
        normal /= normalNorm;
        double d = static_cast<double>(shape.d) / normalNorm;
        if (d < 0.0)
        {
            normal = -normal;
            d = -d;
        }

        PlaneObservation plane;
        plane.normal = normal;
        plane.d = d;
        plane.centroid = shape.centroid.getVector3fMap().cast<double>();
        plane.pointCount = shape.pointCount;
        plane.points = shape.points;
        station.planes.push_back(std::move(plane));
    }

    if (station.planes.size() < 4)
    {
        PCL_WARN("Fewer than 4 usable planes for multiblock in %s\n", record.cloudPath.string().c_str());
        return std::nullopt;
    }

    station.mergedPlanes = FilterPlanesWithOrthogonalMate(MergeSimilarPlanes(station.planes));
    station.mergedPlaneCount = static_cast<int>(station.mergedPlanes.size());
    if (station.mergedPlanes.size() < 4)
    {
        PCL_WARN("Fewer than 4 merged planes with orthogonal mates for multiblock in %s\n", record.cloudPath.string().c_str());
        return std::nullopt;
    }
    return station;
}

std::vector<MultiBlockPairCandidate> BuildMultiBlockPairCandidates(const std::vector<PlaneObservation>& planes)
{
    std::vector<MultiBlockPairCandidate> candidates;
    for (size_t i = 0; i < planes.size(); ++i)
    {
        for (size_t j = i + 1; j < planes.size(); ++j)
        {
            if (!IsOrthogonal(planes[i].normal, planes[j].normal))
            {
                continue;
            }

            const double centroidDistance = (planes[i].centroid - planes[j].centroid).norm();
            MultiBlockPairCandidate candidate;
            candidate.planeIndices = {static_cast<int>(i), static_cast<int>(j)};
            candidate.angleResidualDeg = AngleResidualDeg(planes[i].normal, planes[j].normal);
            candidate.centroidDistanceResidualMm = std::abs(centroidDistance - kCentroidDistanceMm);
            candidate.centroidDistanceOk = IsExpectedCentroidDistance(centroidDistance);
            candidate.score = candidate.angleResidualDeg + 0.02 * candidate.centroidDistanceResidualMm;
            if (!candidate.centroidDistanceOk)
            {
                candidate.score += 8.0;
            }
            candidates.push_back(candidate);
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const MultiBlockPairCandidate& lhs, const MultiBlockPairCandidate& rhs) {
        if (lhs.centroidDistanceOk != rhs.centroidDistanceOk)
        {
            return lhs.centroidDistanceOk > rhs.centroidDistanceOk;
        }
        return lhs.score < rhs.score;
    });
    if (candidates.size() > kMultiBlockMaxPairCandidates)
    {
        candidates.resize(kMultiBlockMaxPairCandidates);
    }
    return candidates;
}

bool PairCandidateSharesPlane(const MultiBlockPairCandidate& lhs, const MultiBlockPairCandidate& rhs)
{
    for (const int lhsIndex : lhs.planeIndices)
    {
        for (const int rhsIndex : rhs.planeIndices)
        {
            if (lhsIndex == rhsIndex)
            {
                return true;
            }
        }
    }
    return false;
}

std::vector<MultiBlockTwoBlockCandidate> BuildMultiBlockTwoBlockCandidates(const std::vector<PlaneObservation>& planes)
{
    const std::vector<MultiBlockPairCandidate> pairs = BuildMultiBlockPairCandidates(planes);
    std::vector<MultiBlockTwoBlockCandidate> candidates;
    for (size_t i = 0; i < pairs.size(); ++i)
    {
        for (size_t j = i + 1; j < pairs.size(); ++j)
        {
            if (PairCandidateSharesPlane(pairs[i], pairs[j]))
            {
                continue;
            }

            MultiBlockTwoBlockCandidate candidate;
            candidate.blocks = {pairs[i], pairs[j]};
            candidate.score = pairs[i].score + pairs[j].score;
            candidates.push_back(candidate);
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const MultiBlockTwoBlockCandidate& lhs, const MultiBlockTwoBlockCandidate& rhs) {
        return lhs.score < rhs.score;
    });
    if (candidates.size() > kMultiBlockMaxTwoBlockCandidates)
    {
        candidates.resize(kMultiBlockMaxTwoBlockCandidates);
    }
    return candidates;
}

std::array<std::array<Eigen::Vector3d, kMultiBlockFacesPerBlock>, kMultiBlockBlockCount> CandidateNormalsBase(
    const MultiBlockPlaneStation& station,
    const MultiBlockTwoBlockCandidate& candidate,
    const Eigen::Isometry3d& flangeFromCamera)
{
    std::array<std::array<Eigen::Vector3d, kMultiBlockFacesPerBlock>, kMultiBlockBlockCount> normals;
    for (int blockIndex = 0; blockIndex < kMultiBlockBlockCount; ++blockIndex)
    {
        for (int faceIndex = 0; faceIndex < kMultiBlockFacesPerBlock; ++faceIndex)
        {
            const int planeIndex = candidate.blocks[blockIndex].planeIndices[faceIndex];
            normals[blockIndex][faceIndex] = TransformMultiBlockPlaneNormal(
                station,
                station.mergedPlanes[static_cast<size_t>(planeIndex)],
                flangeFromCamera);
        }
    }
    return normals;
}

MultiBlockNormalGrid CandidateNormalsCamera(
    const MultiBlockPlaneStation& station,
    const MultiBlockTwoBlockCandidate& candidate)
{
    MultiBlockNormalGrid normals;
    for (int blockIndex = 0; blockIndex < kMultiBlockBlockCount; ++blockIndex)
    {
        for (int faceIndex = 0; faceIndex < kMultiBlockFacesPerBlock; ++faceIndex)
        {
            const int planeIndex = candidate.blocks[blockIndex].planeIndices[faceIndex];
            normals[blockIndex][faceIndex] = station.mergedPlanes[static_cast<size_t>(planeIndex)].normal.normalized();
        }
    }
    return normals;
}

MultiBlockNormalGrid TransformMultiBlockNormals(
    const MultiBlockNormalGrid& normals,
    const Eigen::Matrix3d& rotationTargetFromSource)
{
    MultiBlockNormalGrid out;
    for (int blockIndex = 0; blockIndex < kMultiBlockBlockCount; ++blockIndex)
    {
        for (int faceIndex = 0; faceIndex < kMultiBlockFacesPerBlock; ++faceIndex)
        {
            out[blockIndex][faceIndex] = (rotationTargetFromSource * normals[blockIndex][faceIndex]).normalized();
        }
    }
    return out;
}

PTR BuildMultiBlockCandidateCloud(
    const MultiBlockPlaneStation& station,
    const MultiBlockTwoBlockCandidate& candidate)
{
    PTR cloud(new PCD);
    std::array<int, kMultiBlockBlockCount * kMultiBlockFacesPerBlock> usedPlaneIndices{};
    size_t usedPlaneCount = 0;

    for (int blockIndex = 0; blockIndex < kMultiBlockBlockCount; ++blockIndex)
    {
        for (int faceIndex = 0; faceIndex < kMultiBlockFacesPerBlock; ++faceIndex)
        {
            const int planeIndex = candidate.blocks[blockIndex].planeIndices[faceIndex];
            bool alreadyUsed = false;
            for (size_t usedIndex = 0; usedIndex < usedPlaneCount; ++usedIndex)
            {
                if (usedPlaneIndices[usedIndex] == planeIndex)
                {
                    alreadyUsed = true;
                    break;
                }
            }
            if (alreadyUsed || planeIndex < 0 || static_cast<size_t>(planeIndex) >= station.mergedPlanes.size())
            {
                continue;
            }

            usedPlaneIndices[usedPlaneCount++] = planeIndex;
            const PlaneObservation& plane = station.mergedPlanes[static_cast<size_t>(planeIndex)];
            for (const Eigen::Vector3d& point : plane.points)
            {
                if (!point.allFinite())
                {
                    continue;
                }
                cloud->push_back(pcl::PointXYZ(
                    static_cast<float>(point.x()),
                    static_cast<float>(point.y()),
                    static_cast<float>(point.z())));
            }
        }
    }

    cloud->width = static_cast<uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = false;
    if (cloud->size() < kMultiBlockGicpMinPoints)
    {
        return cloud;
    }

    PTR sampled = downSampling(cloud, static_cast<float>(kMultiBlockGicpVoxelLeafMm));
    if (sampled && !sampled->empty())
    {
        return sampled;
    }
    return cloud;
}

struct MultiBlockGicpResult
{
    bool success = false;
    double fitnessMm2 = std::numeric_limits<double>::infinity();
    Eigen::Matrix4f targetFromSource = Eigen::Matrix4f::Identity();
    size_t sourcePointCount = 0;
    size_t targetPointCount = 0;
};

Eigen::Matrix4f InitialMultiBlockGicpGuess(
    const MultiBlockPlaneStation& sourceStation,
    const MultiBlockPlaneStation& targetStation,
    const Eigen::Isometry3d& initialFlangeFromCamera)
{
    const Eigen::Isometry3d baseFromSourceCamera = sourceStation.baseFromFlange * initialFlangeFromCamera;
    const Eigen::Isometry3d baseFromTargetCamera = targetStation.baseFromFlange * initialFlangeFromCamera;
    return (baseFromTargetCamera.inverse() * baseFromSourceCamera).matrix().cast<float>();
}
MultiBlockGicpResult RunMultiBlockGicp(
    const PTR& sourceCloud,
    const PTR& targetCloud,
    const Eigen::Matrix4f& initialTargetFromSource)
{
    MultiBlockGicpResult result;
    result.sourcePointCount = sourceCloud ? sourceCloud->size() : 0;
    result.targetPointCount = targetCloud ? targetCloud->size() : 0;
    if (!sourceCloud || !targetCloud ||
        result.sourcePointCount < kMultiBlockGicpMinPoints ||
        result.targetPointCount < kMultiBlockGicpMinPoints)
    {
        return result;
    }

    try
    {
        fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> gicp;
        gicp.setInputSource(sourceCloud);
        gicp.setInputTarget(targetCloud);
        gicp.setNumThreads(8);
        //gicp.setCorrespondenceRandomness(20);
        gicp.setRegularizationMethod(fast_gicp::RegularizationMethod::PLANE);
        gicp.setMaximumIterations(kMultiBlockGicpMaxIterations);
        gicp.setMaxCorrespondenceDistance(kMultiBlockGicpMaxCorrespondenceDistanceMm);
        gicp.setTransformationEpsilon(1.0);
        //gicp.setEuclideanFitnessEpsilon(1.0e-6);

        PCD aligned;
        gicp.align(aligned, initialTargetFromSource);
        if (!gicp.hasConverged())
        {
            return result;
        }

        result.targetFromSource = gicp.getFinalTransformation();
        if (!result.targetFromSource.allFinite())
        {
            return result;
        }

        result.fitnessMm2 = gicp.getFitnessScore();//kMultiBlockGicpMaxCorrespondenceDistanceMm
        if (!std::isfinite(result.fitnessMm2) || result.fitnessMm2 > kMultiBlockGicpMaxFitnessMm2)
        {
            return result;
        }

        result.success = true;
    }
    catch (const std::exception& ex)
    {
        PCL_WARN("GICP matching failed with exception: %s\n", ex.what());
    }
    return result;
}

std::array<std::array<Eigen::Vector3d, kMultiBlockFacesPerBlock>, kMultiBlockBlockCount> ApplyMultiBlockPermutation(
    const std::array<std::array<Eigen::Vector3d, kMultiBlockFacesPerBlock>, kMultiBlockBlockCount>& normals,
    bool swapBlocks,
    const std::array<bool, kMultiBlockBlockCount>& swapFaces)
{
    std::array<std::array<Eigen::Vector3d, kMultiBlockFacesPerBlock>, kMultiBlockBlockCount> out;
    for (int targetBlock = 0; targetBlock < kMultiBlockBlockCount; ++targetBlock)
    {
        const int sourceBlock = swapBlocks ? 1 - targetBlock : targetBlock;
        for (int targetFace = 0; targetFace < kMultiBlockFacesPerBlock; ++targetFace)
        {
            const int sourceFace = swapFaces[targetBlock] ? 1 - targetFace : targetFace;
            out[targetBlock][targetFace] = normals[sourceBlock][sourceFace];
        }
    }
    return out;
}

std::array<std::array<int, kMultiBlockFacesPerBlock>, kMultiBlockBlockCount> ApplyMultiBlockIndexPermutation(
    const MultiBlockTwoBlockCandidate& candidate,
    bool swapBlocks,
    const std::array<bool, kMultiBlockBlockCount>& swapFaces)
{
    std::array<std::array<int, kMultiBlockFacesPerBlock>, kMultiBlockBlockCount> out;
    for (int targetBlock = 0; targetBlock < kMultiBlockBlockCount; ++targetBlock)
    {
        const int sourceBlock = swapBlocks ? 1 - targetBlock : targetBlock;
        for (int targetFace = 0; targetFace < kMultiBlockFacesPerBlock; ++targetFace)
        {
            const int sourceFace = swapFaces[targetBlock] ? 1 - targetFace : targetFace;
            out[targetBlock][targetFace] = candidate.blocks[sourceBlock].planeIndices[sourceFace];
        }
    }
    return out;
}

bool BestMultiBlockCandidatePermutation(
    const std::array<std::array<Eigen::Vector3d, kMultiBlockFacesPerBlock>, kMultiBlockBlockCount>& referenceNormals,
    const std::array<std::array<Eigen::Vector3d, kMultiBlockFacesPerBlock>, kMultiBlockBlockCount>& candidateNormals,
    bool& bestSwapBlocks,
    std::array<bool, kMultiBlockBlockCount>& bestSwapFaces,
    double& bestMinDot,
    double& bestScore)
{
    bestScore = -std::numeric_limits<double>::infinity();
    bestMinDot = 0.0;
    bestSwapBlocks = false;
    bestSwapFaces = {false, false};
    bool found = false;

    for (int swapBlocksValue = 0; swapBlocksValue <= 1; ++swapBlocksValue)
    {
        for (int swapFace0 = 0; swapFace0 <= 1; ++swapFace0)
        {
            for (int swapFace1 = 0; swapFace1 <= 1; ++swapFace1)
            {
                const bool swapBlocks = swapBlocksValue != 0;
                const std::array<bool, kMultiBlockBlockCount> swapFaces = {swapFace0 != 0, swapFace1 != 0};
                const auto permuted = ApplyMultiBlockPermutation(candidateNormals, swapBlocks, swapFaces);

                double score = 0.0;
                double minDot = std::numeric_limits<double>::max();
                for (int blockIndex = 0; blockIndex < kMultiBlockBlockCount; ++blockIndex)
                {
                    for (int faceIndex = 0; faceIndex < kMultiBlockFacesPerBlock; ++faceIndex)
                    {
                        const double dot = std::abs(referenceNormals[blockIndex][faceIndex].dot(permuted[blockIndex][faceIndex]));
                        score += dot;
                        minDot = std::min(minDot, dot);
                    }
                }

                if (!found || score > bestScore)
                {
                    found = true;
                    bestScore = score;
                    bestMinDot = minDot;
                    bestSwapBlocks = swapBlocks;
                    bestSwapFaces = swapFaces;
                }
            }
        }
    }
    return found;
}

HandEyeMultiBlockPlaneOptimizer::PlaneObservation ToMultiBlockOptimizerPlane(
    const PlaneObservation& plane,
    int blockIndex,
    int faceIndex)
{
    HandEyeMultiBlockPlaneOptimizer::PlaneObservation out;
    out.blockIndex = blockIndex;
    out.faceIndex = faceIndex;
    out.blockNormal = faceIndex == 0 ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY();
    out.normal = plane.normal;
    out.d = plane.d;
    out.points = plane.points;
    out.weight = plane.pointCount > 0 ? static_cast<double>(plane.pointCount) : 1.0;
    return out;
}

std::optional<MultiBlockMatchedStation> MakeMatchedMultiBlockStation(
    const MultiBlockPlaneStation& station,
    const MultiBlockTwoBlockCandidate& candidate,
    const MultiBlockIndexGrid& sourcePlaneIndices,
    double score,
    double minMatchedNormalDot,
    bool matchedByGicp = false,
    double gicpFitnessMm2 = 0.0,
    const Eigen::Matrix4f& gicpTargetFromSource = Eigen::Matrix4f::Identity())
{
    MultiBlockMatchedStation matched;
    matched.optimizerStation.timestamp = station.timestamp;
    matched.optimizerStation.baseFromFlange = station.baseFromFlange;
    matched.sourcePlaneIndices = sourcePlaneIndices;
    matched.score = score;
    matched.minMatchedNormalDot = minMatchedNormalDot;
    matched.matchedByGicp = matchedByGicp;
    matched.gicpFitnessMm2 = gicpFitnessMm2;
    matched.gicpTargetFromSource = gicpTargetFromSource;

    for (int blockIndex = 0; blockIndex < kMultiBlockBlockCount; ++blockIndex)
    {
        for (int faceIndex = 0; faceIndex < kMultiBlockFacesPerBlock; ++faceIndex)
        {
            const int planeIndex = sourcePlaneIndices[blockIndex][faceIndex];
            if (planeIndex < 0 || static_cast<size_t>(planeIndex) >= station.mergedPlanes.size())
            {
                return std::nullopt;
            }
            matched.optimizerStation.planes.push_back(ToMultiBlockOptimizerPlane(
                station.mergedPlanes[static_cast<size_t>(planeIndex)],
                blockIndex,
                faceIndex));
        }
    }
    return matched;
}

std::vector<MultiBlockMatchedStation> SelectMultiBlockMatchedStationsWithInitial(
    const std::vector<MultiBlockPlaneStation>& stations,
    const Eigen::Isometry3d& initialFlangeFromCamera)
{
    std::vector<std::vector<MultiBlockTwoBlockCandidate>> allCandidates;
    allCandidates.reserve(stations.size());
    for (const MultiBlockPlaneStation& station : stations)
    {
        std::vector<MultiBlockTwoBlockCandidate> candidates = BuildMultiBlockTwoBlockCandidates(station.mergedPlanes);
        if (candidates.empty())
        {
            PCL_WARN("No two-block plane candidate in station %s\n", station.timestamp.c_str());
            return {};
        }
        allCandidates.push_back(std::move(candidates));
    }

    std::vector<MultiBlockMatchedStation> bestMatched;
    double bestTotalScore = std::numeric_limits<double>::max();
    for (const MultiBlockTwoBlockCandidate& referenceCandidate : allCandidates.front())
    {
        const auto referenceNormals = CandidateNormalsBase(stations.front(), referenceCandidate, initialFlangeFromCamera);
        std::vector<MultiBlockMatchedStation> matched;
        matched.reserve(stations.size());

        const auto referenceIndices = ApplyMultiBlockIndexPermutation(referenceCandidate, false, {false, false});
        auto referenceMatched = MakeMatchedMultiBlockStation(
            stations.front(),
            referenceCandidate,
            referenceIndices,
            referenceCandidate.score,
            1.0);
        if (!referenceMatched)
        {
            continue;
        }
        matched.push_back(*referenceMatched);

        double totalScore = referenceCandidate.score;
        bool allOk = true;
        for (size_t stationIndex = 1; stationIndex < stations.size(); ++stationIndex)
        {
            const MultiBlockPlaneStation& station = stations[stationIndex];
            double bestStationScore = std::numeric_limits<double>::max();
            double bestStationMinDot = 0.0;
            std::array<std::array<int, kMultiBlockFacesPerBlock>, kMultiBlockBlockCount> bestIndices{};
            const MultiBlockTwoBlockCandidate* bestCandidate = nullptr;

            for (const MultiBlockTwoBlockCandidate& candidate : allCandidates[stationIndex])
            {
                const auto candidateNormals = CandidateNormalsBase(station, candidate, initialFlangeFromCamera);
                bool swapBlocks = false;
                std::array<bool, kMultiBlockBlockCount> swapFaces = {false, false};
                double minDot = 0.0;
                double normalScore = 0.0;
                if (!BestMultiBlockCandidatePermutation(
                        referenceNormals,
                        candidateNormals,
                        swapBlocks,
                        swapFaces,
                        minDot,
                        normalScore))
                {
                    continue;
                }
                if (minDot < kMultiBlockMinMatchedNormalDot)
                {
                    continue;
                }

                const double stationScore = candidate.score + 20.0 * (4.0 - normalScore);
                if (stationScore < bestStationScore)
                {
                    bestStationScore = stationScore;
                    bestStationMinDot = minDot;
                    bestIndices = ApplyMultiBlockIndexPermutation(candidate, swapBlocks, swapFaces);
                    bestCandidate = &candidate;
                }
            }

            if (!bestCandidate)
            {
                allOk = false;
                break;
            }

            auto stationMatched = MakeMatchedMultiBlockStation(
                station,
                *bestCandidate,
                bestIndices,
                bestStationScore,
                bestStationMinDot);
            if (!stationMatched)
            {
                allOk = false;
                break;
            }
            totalScore += bestStationScore;
            matched.push_back(*stationMatched);
        }

        if (allOk && matched.size() == stations.size() && totalScore < bestTotalScore)
        {
            bestTotalScore = totalScore;
            bestMatched = std::move(matched);
        }
    }
    return bestMatched;
}

std::vector<MultiBlockMatchedStation> SelectMultiBlockMatchedStationsByGicp(
    const std::vector<MultiBlockPlaneStation>& stations,
    const Eigen::Isometry3d& initialFlangeFromCamera)
{
    std::vector<std::vector<MultiBlockTwoBlockCandidate>> allCandidates;
    allCandidates.reserve(stations.size());
    for (const MultiBlockPlaneStation& station : stations)
    {
        std::vector<MultiBlockTwoBlockCandidate> candidates = BuildMultiBlockTwoBlockCandidates(station.mergedPlanes);
        if (candidates.empty())
        {
            PCL_WARN("No two-block plane candidate in station %s\n", station.timestamp.c_str());
            return {};
        }
        allCandidates.push_back(std::move(candidates));
    }

    const MultiBlockTwoBlockCandidate& referenceCandidate = allCandidates.front().front();
    const MultiBlockIndexGrid referenceIndices = ApplyMultiBlockIndexPermutation(referenceCandidate, false, {false, false});
    auto referenceMatched = MakeMatchedMultiBlockStation(
        stations.front(),
        referenceCandidate,
        referenceIndices,
        referenceCandidate.score,
        1.0,
        false,
        0.0,
        Eigen::Matrix4f::Identity());
    if (!referenceMatched)
    {
        return {};
    }

    PTR previousCloud = BuildMultiBlockCandidateCloud(stations.front(), referenceCandidate);
    if (!previousCloud || previousCloud->size() < kMultiBlockGicpMinPoints)
    {
        PCL_WARN("Reference station %s has too few merged plane points for GICP matching (%zu).\n",
            stations.front().timestamp.c_str(),
            previousCloud ? previousCloud->size() : 0);
        return {};
    }

    MultiBlockNormalGrid previousNormals = CandidateNormalsCamera(stations.front(), referenceCandidate);
    std::vector<MultiBlockMatchedStation> matched;
    matched.reserve(stations.size());
    matched.push_back(*referenceMatched);

    for (size_t stationIndex = 1; stationIndex < stations.size(); ++stationIndex)
    {
        const MultiBlockPlaneStation& station = stations[stationIndex];
        double bestStationScore = std::numeric_limits<double>::max();
        double bestStationMinDot = 0.0;
        double bestNormalScore = 0.0;
        MultiBlockIndexGrid bestIndices{};
        const MultiBlockTwoBlockCandidate* bestCandidate = nullptr;
        PTR bestCloud;
        MultiBlockNormalGrid bestNormals;
        MultiBlockGicpResult bestGicp;

        const size_t testCount = std::min(allCandidates[stationIndex].size(), kMultiBlockMaxGicpCandidateTests);
        for (size_t candidateIndex = 0; candidateIndex < testCount; ++candidateIndex)
        {
            const MultiBlockTwoBlockCandidate& candidate = allCandidates[stationIndex][candidateIndex];
            PTR currentCloud = BuildMultiBlockCandidateCloud(station, candidate);
            const Eigen::Matrix4f initialGicpGuess = InitialMultiBlockGicpGuess(
                station,
                stations[stationIndex - 1],
                initialFlangeFromCamera);
            const MultiBlockGicpResult gicp = RunMultiBlockGicp(currentCloud, previousCloud, initialGicpGuess);
            if (!gicp.success)
            {
                continue;
            }

            const MultiBlockNormalGrid candidateNormals = CandidateNormalsCamera(station, candidate);
            const Eigen::Matrix3d rotationPreviousFromCurrent = gicp.targetFromSource.block<3, 3>(0, 0).cast<double>();
            const MultiBlockNormalGrid transformedNormals = TransformMultiBlockNormals(candidateNormals, rotationPreviousFromCurrent);

            bool swapBlocks = false;
            std::array<bool, kMultiBlockBlockCount> swapFaces = {false, false};
            double minDot = 0.0;
            double normalScore = 0.0;
            if (!BestMultiBlockCandidatePermutation(
                    previousNormals,
                    transformedNormals,
                    swapBlocks,
                    swapFaces,
                    minDot,
                    normalScore))
            {
                continue;
            }
            if (minDot < kMultiBlockMinMatchedNormalDot)
            {
                continue;
            }

            const double gicpRms = std::sqrt(std::max(0.0, gicp.fitnessMm2));
            const double stationScore = candidate.score + 20.0 * (4.0 - normalScore) + 0.25 * gicpRms;
            if (stationScore < bestStationScore)
            {
                bestStationScore = stationScore;
                bestStationMinDot = minDot;
                bestNormalScore = normalScore;
                bestIndices = ApplyMultiBlockIndexPermutation(candidate, swapBlocks, swapFaces);
                bestCandidate = &candidate;
                bestCloud = currentCloud;
                bestGicp = gicp;
                bestNormals = ApplyMultiBlockPermutation(candidateNormals, swapBlocks, swapFaces);
            }
        }

        if (!bestCandidate || !bestCloud)
        {
            PCL_WARN("GICP could not find a consistent adjacent-station match for station %s.\n", station.timestamp.c_str());
            return {};
        }

        auto stationMatched = MakeMatchedMultiBlockStation(
            station,
            *bestCandidate,
            bestIndices,
            bestStationScore,
            bestStationMinDot,
            true,
            bestGicp.fitnessMm2,
            bestGicp.targetFromSource);
        if (!stationMatched)
        {
            return {};
        }

        PCL_INFO("GICP multiblock match %s: fitness=%lf rms=%lf min_dot=%lf normal_score=%lf source_points=%zu target_points=%zu\n",
            station.timestamp.c_str(),
            bestGicp.fitnessMm2,
            std::sqrt(std::max(0.0, bestGicp.fitnessMm2)),
            bestStationMinDot,
            bestNormalScore,
            bestGicp.sourcePointCount,
            bestGicp.targetPointCount);

        matched.push_back(*stationMatched);
        previousCloud = bestCloud;
        previousNormals = bestNormals;
    }

    return matched;
}

std::vector<MultiBlockMatchedStation> SelectMultiBlockMatchedStations(
    const std::vector<MultiBlockPlaneStation>& stations,
    const Eigen::Isometry3d& initialFlangeFromCamera,
    bool useInitialHandEyeForMatching)
{
    if (!useInitialHandEyeForMatching)
    {
        return SelectMultiBlockMatchedStationsByGicp(stations, initialFlangeFromCamera);
    }
    return SelectMultiBlockMatchedStationsWithInitial(stations, initialFlangeFromCamera);
}
void WriteMultiBlockReport(
    std::ostream& out,
    const fs::path& dataDir,
    const Eigen::Isometry3d& initialFlangeFromCamera,
    const std::vector<MultiBlockMatchedStation>& matchedStations,
    const HandEyeMultiBlockPlaneOptimizer::Result& result)
{
    out << std::fixed << std::setprecision(6);
    out << "Multi-block plane hand-eye calibration report\n";
    out << "Data dir: " << dataDir.string() << "\n";
    out << "Valid stations: " << matchedStations.size() << "\n";
    const bool reportUsesGicp = std::any_of(matchedStations.begin(), matchedStations.end(), [](const MultiBlockMatchedStation& station) {
        return station.matchedByGicp;
    });
    out << "matching_mode: " << (reportUsesGicp ? "adjacent_gicp" : "initial_handeye") << "\n\n";

    out << "initial_camera_center_in_flange_mm: ";
    PrintVector(out, initialFlangeFromCamera.translation());
    out << "\n";
    out << "camera_center_in_flange_mm: ";
    PrintVector(out, result.flangeFromCamera.translation());
    out << "\n";
    const Eigen::Quaterniond quaternion(result.flangeFromCamera.linear());
    const auto rpy = RpyFromRotation(result.flangeFromCamera.linear());
    out << "quaternion_wxyz: " << quaternion.w() << ' ' << quaternion.x() << ' '
        << quaternion.y() << ' ' << quaternion.z() << "\n";
    out << "rpy_xyz_deg: " << rpy[0] << ' ' << rpy[1] << ' ' << rpy[2] << "\n";
    out << "rotation_matrix:\n";
    for (int row = 0; row < 3; ++row)
    {
        out << "  " << result.flangeFromCamera.linear()(row, 0) << ' '
            << result.flangeFromCamera.linear()(row, 1) << ' '
            << result.flangeFromCamera.linear()(row, 2) << "\n";
    }

    out << "ceres_success: " << (result.success ? "yes" : "no") << "\n";
    out << "ceres_brief_report: " << result.briefReport << "\n";
    out << "ceres_initial_cost: " << result.initialCost << "\n";
    out << "ceres_final_cost: " << result.finalCost << "\n";
    out << "block_count: " << result.blockCount << "\n";
    out << "face_count: " << result.faceCount << "\n";
    out << "observation_count: " << result.observationCount << "\n";
    out << "residual_count: " << result.residualCount << "\n";
    out << "translation_constraint_rank: " << result.translationConstraintRank << "\n";
    out << "plane_mean_abs_distance_mm: " << result.meanAbsDistanceMm << "\n";
    out << "plane_rms_distance_mm: " << result.rmsDistanceMm << "\n";
    out << "plane_max_abs_distance_mm: " << result.maxAbsDistanceMm << "\n";
    out << "normal_mean_angle_deg: " << result.normalMeanAngleDeg << "\n";
    out << "normal_max_angle_deg: " << result.normalMaxAngleDeg << "\n\n";

    out << "Per-face target planes in base:\n";
    out << "block face normal_x normal_y normal_z d_base residual_count rms_mm max_mm\n";
    for (const auto& face : result.faces)
    {
        out << face.blockIndex << ' ' << face.faceIndex << ' ';
        PrintVector(out, face.normalBase);
        out << ' ' << face.dBase << ' ' << face.residualCount << ' '
            << face.rmsDistanceMm << ' ' << face.maxAbsDistanceMm << "\n";
    }

    out << "\nPer-station multiblock matching:\n";
    out << "timestamp block0_faces block1_faces score min_normal_dot matching gicp_fitness_mm2\n";
    for (const MultiBlockMatchedStation& station : matchedStations)
    {
        const char* matchingMode = station.matchedByGicp ? "gicp" : (reportUsesGicp ? "reference" : "initial");
        out << station.optimizerStation.timestamp << ' '
            << station.sourcePlaneIndices[0][0] << ',' << station.sourcePlaneIndices[0][1] << ' '
            << station.sourcePlaneIndices[1][0] << ',' << station.sourcePlaneIndices[1][1] << ' '
            << station.score << ' ' << station.minMatchedNormalDot << ' '
            << matchingMode << ' ' << station.gicpFitnessMm2 << "\n";
    }
}

// Helper: write feature extraction result to a text file for reuse by calibrate mode.
bool WriteFeatureFile(const fs::path& path, const handeye::FeatureExtractionResult& features)
{
    std::ofstream out(path);
    if (!out)
    {
        return false;
    }
    out << std::fixed << std::setprecision(9);
    out << "#feature_result_v1\n";
    out << "station_count " << features.stations.size() << "\n";
    out << "target_count " << features.targetBases.size() << "\n";
    for (const Eigen::Vector3d& t : features.targetBases)
    {
        out << "target " << t.x() << ' ' << t.y() << ' ' << t.z() << "\n";
    }
    out << "tcp_path " << features.tcpFilePath.string() << "\n\n";

    for (const handeye::StationFeatureInfo& station : features.stations)
    {
        const Eigen::Isometry3d& bff = station.baseFromFlange;
        const Eigen::Quaterniond q(bff.linear());
        out << "station " << station.timestamp << ' '
            << station.cloudPath.filename().string() << ' '
            << bff.translation().x() << ' ' << bff.translation().y() << ' ' << bff.translation().z() << ' '
            << q.w() << ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' '
            << station.detectedPlaneCount << ' ' << station.mergedPlaneCount << "\n";

        for (const handeye::PlaneInfo& plane : station.rawPlanes)
        {
            out << "raw_plane "
                << plane.normal.x() << ' ' << plane.normal.y() << ' ' << plane.normal.z() << ' '
                << plane.d << ' '
                << plane.centroid.x() << ' ' << plane.centroid.y() << ' ' << plane.centroid.z() << ' '
                << plane.pointCount << "\n";
        }
        for (const handeye::PlaneInfo& plane : station.mergedPlanes)
        {
            out << "merged_plane "
                << plane.normal.x() << ' ' << plane.normal.y() << ' ' << plane.normal.z() << ' '
                << plane.d << ' '
                << plane.centroid.x() << ' ' << plane.centroid.y() << ' ' << plane.centroid.z() << ' '
                << plane.pointCount << "\n";
        }
        for (const handeye::TrihedralCornerInfo& corner : station.cornerCandidates)
        {
            out << "corner "
                << corner.planeIndices[0] << ' ' << corner.planeIndices[1] << ' ' << corner.planeIndices[2] << ' '
                << corner.point.x() << ' ' << corner.point.y() << ' ' << corner.point.z() << ' '
                << corner.score << ' '
                << corner.angleResidualDeg << ' '
                << corner.centroidPairResidualMm << ' '
                << corner.centroidCornerResidualMm << ' '
                << (corner.centroidDistanceOk ? 1 : 0) << "\n";
        }
        out << "\n";
    }
    return true;
}

// Helper: read feature extraction result from a text file.
// Returns true on success, result is populated.
bool ReadFeatureFile(const fs::path& path, handeye::FeatureExtractionResult& features)
{
    std::ifstream in(path);
    if (!in)
    {
        return false;
    }

    features = handeye::FeatureExtractionResult{};
    size_t expectedStationCount = 0;
    size_t expectedTargetCount = 0;
    handeye::StationFeatureInfo currentStation;
    bool inStation = false;

    std::string line;
    while (std::getline(in, line))
    {
        line = Trim(line);
        if (line.empty() || line[0] == '#')
        {
            continue;
        }

        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "station_count")
        {
            iss >> expectedStationCount;
        }
        else if (token == "target_count")
        {
            iss >> expectedTargetCount;
        }
        else if (token == "target")
        {
            double x = 0, y = 0, z = 0;
            iss >> x >> y >> z;
            features.targetBases.emplace_back(x, y, z);
        }
        else if (token == "tcp_path")
        {
            std::string p;
            std::getline(iss, p);
            features.tcpFilePath = Trim(p);
        }
        else if (token == "station")
        {
            if (inStation)
            {
                features.stations.push_back(std::move(currentStation));
                currentStation = handeye::StationFeatureInfo{};
            }
            inStation = true;
            std::string ts, cloudName;
            double tx = 0, ty = 0, tz = 0, qw = 1, qx = 0, qy = 0, qz = 0;
            int dc = 0, mc = 0;
            iss >> ts >> cloudName >> tx >> ty >> tz >> qw >> qx >> qy >> qz >> dc >> mc;
            currentStation.timestamp = ts;
            currentStation.cloudPath = path.parent_path() / cloudName;
            currentStation.baseFromFlange = Eigen::Isometry3d::Identity();
            currentStation.baseFromFlange.translation() = Eigen::Vector3d(tx, ty, tz);
            currentStation.baseFromFlange.linear() = Eigen::Quaterniond(qw, qx, qy, qz).toRotationMatrix();
            currentStation.detectedPlaneCount = dc;
            currentStation.mergedPlaneCount = mc;
        }
        else if (token == "raw_plane")
        {
            handeye::PlaneInfo plane;
            double nx = 0, ny = 0, nz = 1, d = 0, cx = 0, cy = 0, cz = 0;
            size_t pc = 0;
            iss >> nx >> ny >> nz >> d >> cx >> cy >> cz >> pc;
            plane.normal = Eigen::Vector3d(nx, ny, nz);
            plane.d = d;
            plane.centroid = Eigen::Vector3d(cx, cy, cz);
            plane.pointCount = pc;
            currentStation.rawPlanes.push_back(plane);
        }
        else if (token == "merged_plane")
        {
            handeye::PlaneInfo plane;
            double nx = 0, ny = 0, nz = 1, d = 0, cx = 0, cy = 0, cz = 0;
            size_t pc = 0;
            iss >> nx >> ny >> nz >> d >> cx >> cy >> cz >> pc;
            plane.normal = Eigen::Vector3d(nx, ny, nz);
            plane.d = d;
            plane.centroid = Eigen::Vector3d(cx, cy, cz);
            plane.pointCount = pc;
            currentStation.mergedPlanes.push_back(plane);
        }
        else if (token == "corner")
        {
            handeye::TrihedralCornerInfo corner;
            double px = 0, py = 0, pz = 0;
            int ok = 0;
            iss >> corner.planeIndices[0] >> corner.planeIndices[1] >> corner.planeIndices[2]
                >> px >> py >> pz
                >> corner.score
                >> corner.angleResidualDeg
                >> corner.centroidPairResidualMm
                >> corner.centroidCornerResidualMm
                >> ok;
            corner.point = Eigen::Vector3d(px, py, pz);
            corner.centroidDistanceOk = (ok != 0);
            currentStation.cornerCandidates.push_back(corner);
        }
    }

    if (inStation)
    {
        features.stations.push_back(std::move(currentStation));
    }

    features.stationCount = static_cast<int>(features.stations.size());
    if (features.stationCount < 3)
    {
        features.code = -3;
        features.message = "Feature file has fewer than 3 stations: " + path.string();
        return false;
    }

    features.code = 0;
    features.message = "Feature file loaded: " + path.string();
    return true;
}

// Helper: load plane point sets from a feature XYZI PCD file.
// Reconstructs PlaneObservation::points from the intensity-labeled points
// that ExportFeatureCloud wrote.
bool LoadPlanePointsFromFeaturePcd(
    StationCandidates& station,
    const fs::path& featurePcdPath)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    if (pcl::io::loadPCDFile(featurePcdPath.string(), *cloud) == -1)
    {
        PCL_WARN("Cannot load feature PCD for point reconstruction: %s\n",
            featurePcdPath.string().c_str());
        return false;
    }

    // Group points by intensity ID (ID -> vector of points)
    std::map<int, std::vector<Eigen::Vector3d>> idGroups;
    for (const auto& pt : cloud->points)
    {
        if (!pcl::isFinite(pt))
            continue;
        const int id = static_cast<int>(pt.intensity);
        if (id == 0)
            continue;
        idGroups[id].emplace_back(
            static_cast<double>(pt.x),
            static_cast<double>(pt.y),
            static_cast<double>(pt.z));
    }

    if (idGroups.empty())
    {
        PCL_WARN("Feature PCD has no labeled points: %s\n", featurePcdPath.string().c_str());
        return false;
    }

    // For each corner candidate, map intensity IDs to plane indices
    constexpr size_t kMaxCandidates = 5;  // matches kMaxExportTrihedrals
    const size_t candidateCount = std::min(kMaxCandidates, station.candidates.size());
    for (size_t ti = 0; ti < candidateCount; ++ti)
    {
        CornerCandidate& corner = station.candidates[ti];
        const int trihedralId = static_cast<int>(ti) + 1;
        std::vector<PlaneObservation>& planeList = corner.usesMergedPlanes
            ? station.mergedPlanes : station.planes;

        for (size_t fi = 0; fi < 3; ++fi)
        {
            const int faceId = trihedralId * 10 + static_cast<int>(fi) + 1;
            const auto it = idGroups.find(faceId);
            if (it == idGroups.end())
            {
                continue;
            }

            const int planeIdx = corner.planeIndices[fi];
            if (planeIdx >= 0 && static_cast<size_t>(planeIdx) < planeList.size())
            {
                PlaneObservation& plane = planeList[static_cast<size_t>(planeIdx)];
                plane.points = it->second;
                plane.pointCount = plane.points.size();
            }
        }
    }

    return true;
}

} // namespace

// ============================================================================
// Module 1 – Scan collection & save (public API)
// ============================================================================

namespace handeye
{

ScanSaveResult SaveScanStation(const ScanSaveParams& params)
{
    ScanSaveResult result;
    try
    {
        if (!params.cloudData || params.cloudPointCount == 0)
        {
            result.code = -1;
            result.message = "Cloud data is null or empty";
            return result;
        }

        const std::string stem = params.fileStem.empty()
            ? ("scan_" + params.timestamp)
            : params.fileStem;

        fs::create_directories(params.outputDir);
        const fs::path cloudPath = params.outputDir / (stem + ".pcd");
        const fs::path posePath = params.outputDir / ("robot_" + params.timestamp + ".pose");
        const fs::path indexPath = params.outputDir / "robot_records.pose";

        // Write point cloud
        PointCloudIo::CameraCloud cameraCloud;
        //cameraCloud.type = static_cast<EVzResultDataType>(params.cloudDataType);
        cameraCloud.count = params.cloudPointCount;
        cameraCloud.data = params.cloudData;
        PointCloudIo::WriteCameraCloud(cloudPath, cameraCloud);

        // Write pose
        PoseIo::Pose flangePose;
        flangePose.x = params.flangeTranslation.x();
        flangePose.y = params.flangeTranslation.y();
        flangePose.z = params.flangeTranslation.z();
        flangePose.rx = params.flangeRotation.x();
        flangePose.ry = params.flangeRotation.y();
        flangePose.rz = params.flangeRotation.z();

        PoseIo::Quaternion flangeQuat;
        flangeQuat.w = params.flangeQuaternion.w();
        flangeQuat.x = params.flangeQuaternion.x();
        flangeQuat.y = params.flangeQuaternion.y();
        flangeQuat.z = params.flangeQuaternion.z();
        PoseIo::Pose tcpPose;
        tcpPose.x = params.tcpTranslation.x();
        tcpPose.y = params.tcpTranslation.y();
        tcpPose.z = params.tcpTranslation.z();

        PoseIo::WriteSinglePoseFile(posePath, params.timestamp, cloudPath,
            flangePose, flangeQuat, tcpPose);
        PoseIo::AppendPoseIndex(indexPath, params.timestamp, cloudPath,
            flangePose, flangeQuat, tcpPose);

        result.cloudPath = cloudPath;
        result.posePath = posePath;
        result.savedPointCount = params.cloudPointCount;

        // Optional base-cloud export
        if (!params.baseCloudOutputDir.empty())
        {
            Eigen::Isometry3d flangeFromCamera = params.flangeFromCamera;
            if (!IsUsableInitialHandEye(flangeFromCamera))
            {
                const fs::path matrixPath = params.baseCloudOutputDir.parent_path() / "handeye_matrix.txt";
                ReadHandEyeInitialMatrix(matrixPath, flangeFromCamera);
            }
            if (IsUsableInitialHandEye(flangeFromCamera))
            {
                fs::create_directories(params.baseCloudOutputDir);
                const fs::path baseCloudPath = params.baseCloudOutputDir / cloudPath.filename();
                Eigen::Isometry3d baseFromFlange = Eigen::Isometry3d::Identity();
                baseFromFlange.translation() = params.flangeTranslation;
                baseFromFlange.linear() = params.flangeQuaternion.toRotationMatrix();
                const Eigen::Isometry3d baseFromCamera = baseFromFlange * flangeFromCamera;
                PointCloudIo::WriteCameraCloudToBase(baseCloudPath, cameraCloud,
                    PointCloudIo::Transform{});
                // Use TransformPcdFileToBase for proper handling
                PointCloudIo::TransformPcdFileToBase(cloudPath, baseCloudPath, baseFromCamera, &std::cerr);
                result.baseCloudPath = baseCloudPath;
            }
        }

        result.code = 0;
        result.message = "Scan station saved successfully";
    }
    catch (const std::exception& ex)
    {
        result.code = -99;
        result.message = std::string("SaveScanStation exception: ") + ex.what();
    }
    return result;
}

ScanCollectionResult BeginScanCollection(const ScanCollectionParams& params)
{
    ScanCollectionResult result;
    try
    {
        fs::create_directories(params.outputDir);
        result.indexFilePath = params.outputDir / "robot_records.pose";
        result.code = 0;
        result.message = "Scan collection session started";
    }
    catch (const std::exception& ex)
    {
        result.code = -99;
        result.message = std::string("BeginScanCollection exception: ") + ex.what();
    }
    return result;
}

// ============================================================================
// Module 2 – Feature extraction (public API)
// ============================================================================

FeatureExtractionResult RunFeatureExtractionInternal(
    const FeatureExtractionParams& params,
    CornerExtractionMode mode,
    const char* featureFileName,
    const char* successPrefix,
    const char* exceptionPrefix)
{
    FeatureExtractionResult result;
    try
    {
        const fs::path absoluteDataDir = fs::absolute(params.dataDir);
        if (!fs::exists(absoluteDataDir) || !fs::is_directory(absoluteDataDir))
        {
            result.code = -1;
            result.message = "Data directory does not exist: " + absoluteDataDir.string();
            return result;
        }

        // Find TCP file
        fs::path tcpPath = absoluteDataDir / "corner.xyz";
        if (!fs::exists(tcpPath))
        {
            for (const fs::directory_entry& entry : fs::directory_iterator(absoluteDataDir))
            {
                if (entry.is_regular_file() && entry.path().extension() == ".xyz")
                {
                    tcpPath = entry.path();
                    break;
                }
            }
        }
        if (fs::exists(tcpPath))
        {
            result.targetBases = ReadTargetTcps(tcpPath);
            result.tcpFilePath = tcpPath;
        }

        // Read pose records
        PoseIo::ReadOptions poseOptions;
        poseOptions.errorStream = &std::cerr;
        const std::vector<PoseRecord> poseRecords = PoseIo::ReadPoseRecords(absoluteDataDir, poseOptions);
        if (poseRecords.empty())
        {
            result.code = -2;
            result.message = "No pose records with matching PCD files in: " + absoluteDataDir.string();
            return result;
        }

        const fs::path featureDir = params.featureOutputDir.empty()
            ? absoluteDataDir.parent_path() / "feature"
            : params.featureOutputDir;

        for (const PoseRecord& record : poseRecords)
        {
            StationCandidates station = ExtractStationCandidates(record, params, mode);
            if (station.candidates.empty())
            {
                ++result.skippedStationCount;
                continue;
            }

            // Convert to public StationFeatureInfo
            StationFeatureInfo info;
            info.timestamp = station.timestamp;
            info.cloudPath = station.cloudPath;
            info.baseFromFlange = station.baseFromFlange;
            info.detectedPlaneCount = station.detectedPlaneCount;
            info.mergedPlaneCount = station.mergedPlaneCount;

            for (const PlaneObservation& planeObs : station.planes)
            {
                PlaneInfo plane;
                plane.normal = planeObs.normal;
                plane.d = planeObs.d;
                plane.centroid = planeObs.centroid;
                plane.pointCount = planeObs.pointCount;
                plane.extentX = 0.0;
                plane.extentY = 0.0;
                info.rawPlanes.push_back(plane);
            }
            for (const PlaneObservation& planeObs : station.mergedPlanes)
            {
                PlaneInfo plane;
                plane.normal = planeObs.normal;
                plane.d = planeObs.d;
                plane.centroid = planeObs.centroid;
                plane.pointCount = planeObs.pointCount;
                plane.extentX = 0.0;
                plane.extentY = 0.0;
                info.mergedPlanes.push_back(plane);
            }
            for (const CornerCandidate& corner : station.candidates)
            {
                TrihedralCornerInfo ci;
                ci.planeIndices = corner.planeIndices;
                ci.point = corner.point;
                ci.score = corner.score;
                ci.angleResidualDeg = corner.angleResidualDeg;
                ci.centroidPairResidualMm = corner.centroidPairResidualMm;
                ci.centroidCornerResidualMm = corner.centroidCornerResidualMm;
                ci.centroidDistanceOk = corner.centroidDistanceOk;
                info.cornerCandidates.push_back(ci);
            }

            result.stations.push_back(std::move(info));
            ++result.stationCount;

            if (params.exportFeatureCloud && !station.candidates.empty())
            {
                ExportFeatureCloud(station, featureDir);
            }
        }

        if (result.stationCount < 3)
        {
            result.code = -3;
            result.message = "At least 3 valid stations are required, got "
                + std::to_string(result.stationCount);
            return result;
        }

        // Save feature file for reuse by calibrate mode
        const fs::path featureFilePath = absoluteDataDir / featureFileName;
        WriteFeatureFile(featureFilePath, result);

        result.code = 0;
        result.message = std::string(successPrefix) + ": " + std::to_string(result.stationCount)
            + " stations, " + std::to_string(result.skippedStationCount) + " skipped";
    }
    catch (const std::exception& ex)
    {
        result.code = -99;
        result.message = std::string(exceptionPrefix) + " exception: " + ex.what();
    }
    return result;
}

FeatureExtractionResult RunFeatureExtraction(const FeatureExtractionParams& params)
{
    return RunFeatureExtractionInternal(
        params,
        CornerExtractionMode::GeometryScore,
        "feature_result.txt",
        "Feature extraction completed",
        "RunFeatureExtraction");
}

FeatureExtractionResult RunWorkpieceCornerFeatureExtraction(const FeatureExtractionParams& params)
{
    return RunFeatureExtractionInternal(
        params,
        CornerExtractionMode::WorkpieceSupportPointCount,
        "workpiece_feature_result.txt",
        "Workpiece corner feature extraction completed",
        "RunWorkpieceCornerFeatureExtraction");
}

// ============================================================================
// Module 2b – Single PCD corner extraction (public API)
// ============================================================================

ExtractCornerResult ExtractCornerFromPcd(
    const std::filesystem::path& pcdPath,
    const FeatureExtractionParams& params)
{
    ExtractCornerResult result;
    try
    {
        if (!fs::exists(pcdPath))
        {
            result.code = -1;
            result.message = "PCD file not found: " + pcdPath.string();
            return result;
        }

        // Load point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        if (!PointCloudIo::LoadPointXYZ(pcdPath, *cloud, &std::cerr))
        {
            result.code = -2;
            result.message = "Failed to load PCD: " + pcdPath.string();
            return result;
        }

        std::vector<int> validIndices;
        pcl::removeNaNFromPointCloud(*cloud, *cloud, validIndices);
        if (cloud->empty())
        {
            result.code = -3;
            result.message = "PCD has no valid XYZ points: " + pcdPath.string();
            return result;
        }

        // Downsample
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudSample(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> voxGrid;
        voxGrid.setInputCloud(cloud);
        voxGrid.setLeafSize(1.0, 1.0, 1.0);
        voxGrid.filter(*cloudSample);
        if (cloudSample->empty())
        {
            result.code = -4;
            result.message = "Downsampled PCD is empty: " + pcdPath.string();
            return result;
        }

        // RANSAC plane detection
        std::vector<PlaneFitResult> detectedPlanes;
        const int detectionOk = doDetection(cloudSample, detectedPlanes, pcl::PointXYZ(),
            params.ransacMinPoints, params.ransacEpsilon, params.ransacBitmapEpsilon,
            params.planeAngleTolerance, 0.01);
        if (detectionOk == 0 || detectedPlanes.size() < 3)
        {
            result.detectedPlaneCount = static_cast<int>(detectedPlanes.size());
            result.code = -5;
            result.message = "Detected fewer than 3 planes (got "
                + std::to_string(detectedPlanes.size()) + "): " + pcdPath.string();
            return result;
        }

        // Filter planes by extent and build PlaneObservations
        std::vector<PlaneObservation> planes;
        planes.reserve(detectedPlanes.size());
        for (const PlaneFitResult& shape : detectedPlanes)
        {
            if (shape.pointCount == 0) continue;
            if (std::fabs(shape.dX) > params.planeMaxExtentMm || std::fabs(shape.dY) > params.planeMaxExtentMm ||
                std::fabs(shape.dX) < params.planeMinExtentMm || std::fabs(shape.dY) < params.planeMinExtentMm)
                continue;

            Eigen::Vector3d normal(
                static_cast<double>(shape.a),
                static_cast<double>(shape.b),
                static_cast<double>(shape.c));
            const double normalNorm = normal.norm();
            if (normalNorm <= std::numeric_limits<double>::epsilon()) continue;
            normal /= normalNorm;
            double d = static_cast<double>(shape.d) / normalNorm;
            if (d < 0.0) { normal = -normal; d = -d; }

            PlaneObservation obs;
            obs.centroid = shape.centroid.getVector3fMap().cast<double>();
            obs.normal = normal;
            obs.d = d;
            obs.pointCount = shape.pointCount;
            obs.points = shape.points;
            planes.push_back(obs);
        }

        result.detectedPlaneCount = static_cast<int>(planes.size());
        if (planes.size() < 3)
        {
            result.code = -6;
            result.message = "Fewer than 3 usable planes after size filter (got "
                + std::to_string(planes.size()) + "): " + pcdPath.string();
            return result;
        }

        // Merge similar planes
        std::vector<PlaneObservation> mergedPlanes = MergeSimilarPlanes(planes,
            params.mergeAngleDeg, params.mergeDistanceMm);
        result.mergedPlaneCount = static_cast<int>(mergedPlanes.size());

        // Build corner candidates from merged and raw planes
        std::vector<CornerCandidate> candidates = BuildCornerCandidates(mergedPlanes, true,
            params.orthogonalToleranceDeg,
            params.expectedCentroidDistanceMm, params.centroidDistanceToleranceMm,
            params.maxCandidatesPerStation);
        std::vector<CornerCandidate> rawCandidates = BuildCornerCandidates(planes, false,
            params.orthogonalToleranceDeg,
            params.expectedCentroidDistanceMm, params.centroidDistanceToleranceMm,
            params.maxCandidatesPerStation);
        candidates.insert(candidates.end(), rawCandidates.begin(), rawCandidates.end());
        std::sort(candidates.begin(), candidates.end(), [](const CornerCandidate& lhs, const CornerCandidate& rhs) {
            if (lhs.centroidDistanceOk != rhs.centroidDistanceOk)
                return lhs.centroidDistanceOk > rhs.centroidDistanceOk;
            return lhs.score < rhs.score;
        });

        if (candidates.empty())
        {
            result.code = -7;
            result.message = "No valid trihedral corner found in: " + pcdPath.string();
            return result;
        }

        // Populate ranked multi-corner result. Multiple plane combinations can
        // describe the same physical corner, so keep only spatially distinct
        // intersections while preserving score order.
        const double dedupeDistanceMm = std::max(1.0, params.ransacBitmapEpsilon * 2.0);
        result.corners.reserve(candidates.size());
        for (const CornerCandidate& candidate : candidates)
        {
            bool duplicate = false;
            for (const ExtractedCornerInfo& existing : result.corners)
            {
                if ((existing.point - candidate.point).norm() <= dedupeDistanceMm)
                {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate)
            {
                continue;
            }

            ExtractedCornerInfo corner;
            corner.point = candidate.point;
            corner.score = candidate.score;
            corner.centroidDistanceOk = candidate.centroidDistanceOk;
            corner.planeIndices = candidate.planeIndices;
            corner.usesMergedPlanes = candidate.usesMergedPlanes;

            const std::vector<PlaneObservation>& srcPlanes = candidate.usesMergedPlanes ? mergedPlanes : planes;
            for (size_t i = 0; i < 3 && i < candidate.planeIndices.size(); ++i)
            {
                const int idx = candidate.planeIndices[i];
                if (idx >= 0 && static_cast<size_t>(idx) < srcPlanes.size())
                {
                    corner.planeNormals[i] = srcPlanes[static_cast<size_t>(idx)].normal;
                    corner.planeDs[i] = srcPlanes[static_cast<size_t>(idx)].d;
                }
            }
            result.corners.push_back(corner);
        }

        if (result.corners.empty())
        {
            result.code = -7;
            result.message = "No spatially distinct trihedral corner found in: " + pcdPath.string();
            return result;
        }

        const ExtractedCornerInfo& best = result.corners.front();
        result.cornerPoint = best.point;
        result.score = best.score;
        result.centroidDistanceOk = best.centroidDistanceOk;
        result.planeNormals = best.planeNormals;
        result.planeDs = best.planeDs;

        result.code = 0;
        result.message = "Corners extracted: " + std::to_string(result.corners.size()) +
            ", best=(" +
            std::to_string(result.cornerPoint.x()) + ", " +
            std::to_string(result.cornerPoint.y()) + ", " +
            std::to_string(result.cornerPoint.z()) + ") score=" +
            std::to_string(result.score);
    }
    catch (const std::exception& ex)
    {
        result.code = -99;
        result.message = std::string("ExtractCornerFromPcd exception: ") + ex.what();
    }
    return result;
}


// ============================================================================
// Module 3 – Calibrate mode (public API)
// ============================================================================

CalibrateResult RunCalibrate(const CalibrateParams& params)
{
    CalibrateResult result;
    try
    {
        const fs::path absoluteDataDir = fs::absolute(params.dataDir);
        if (!fs::exists(absoluteDataDir) || !fs::is_directory(absoluteDataDir))
        {
            result.code = -1;
            result.message = "Data directory does not exist: " + absoluteDataDir.string();
            return result;
        }

        // Step 1: Feature extraction (or use pre-extracted stations)
        std::vector<StationCandidates> stationCandidates;
        std::vector<Eigen::Vector3d> targetBases = params.targetBases;

        if (!params.stations.empty())
        {
            // Use pre-extracted stations passed via params
            for (const StationFeatureInfo& info : params.stations)
            {
                StationCandidates station;
                station.timestamp = info.timestamp;
                station.cloudPath = info.cloudPath;
                station.baseFromFlange = info.baseFromFlange;
                station.detectedPlaneCount = info.detectedPlaneCount;
                station.mergedPlaneCount = info.mergedPlaneCount;

                for (const PlaneInfo& p : info.rawPlanes)
                {
                    PlaneObservation planeObs;
                    planeObs.normal = p.normal;
                    planeObs.d = p.d;
                    planeObs.centroid = p.centroid;
                    planeObs.pointCount = p.pointCount;
                    station.planes.push_back(planeObs);
                }
                for (const PlaneInfo& p : info.mergedPlanes)
                {
                    PlaneObservation planeObs;
                    planeObs.normal = p.normal;
                    planeObs.d = p.d;
                    planeObs.centroid = p.centroid;
                    planeObs.pointCount = p.pointCount;
                    station.mergedPlanes.push_back(planeObs);
                }
                for (const TrihedralCornerInfo& ci : info.cornerCandidates)
                {
                    CornerCandidate corner;
                    corner.planeIndices = ci.planeIndices;
                    corner.point = ci.point;
                    corner.score = ci.score;
                    corner.angleResidualDeg = ci.angleResidualDeg;
                    corner.centroidPairResidualMm = ci.centroidPairResidualMm;
                    corner.centroidCornerResidualMm = ci.centroidCornerResidualMm;
                    corner.centroidDistanceOk = ci.centroidDistanceOk;
                    station.candidates.push_back(corner);
                }
                if (!station.candidates.empty())
                {
                    stationCandidates.push_back(std::move(station));
                }
            }
        }
        else
        {
            // Try loading saved feature file first
            const fs::path featureFilePath = absoluteDataDir / "feature_result.txt";
            FeatureExtractionResult savedFeatures;
            bool loadedFeatures = false;
            if (fs::exists(featureFilePath))
            {
                loadedFeatures = ReadFeatureFile(featureFilePath, savedFeatures);
                if (loadedFeatures)
                {
                    PCL_INFO("Loaded pre-extracted features from %s: %d stations\n",
                        featureFilePath.string().c_str(), savedFeatures.stationCount);
                }
            }

            if (loadedFeatures)
            {
                // Use saved features
                if (targetBases.empty() && !savedFeatures.targetBases.empty())
                {
                    targetBases = savedFeatures.targetBases;
                }

                const fs::path featureBaseDir = absoluteDataDir.parent_path() / "feature";

                for (const StationFeatureInfo& info : savedFeatures.stations)
                {
                    StationCandidates station;
                    station.timestamp = info.timestamp;
                    station.cloudPath = info.cloudPath;
                    station.baseFromFlange = info.baseFromFlange;
                    station.detectedPlaneCount = info.detectedPlaneCount;
                    station.mergedPlaneCount = info.mergedPlaneCount;

                    for (const PlaneInfo& p : info.rawPlanes)
                    {
                        PlaneObservation planeObs;
                        planeObs.normal = p.normal;
                        planeObs.d = p.d;
                        planeObs.centroid = p.centroid;
                        planeObs.pointCount = p.pointCount;
                        station.planes.push_back(planeObs);
                    }
                    for (const PlaneInfo& p : info.mergedPlanes)
                    {
                        PlaneObservation planeObs;
                        planeObs.normal = p.normal;
                        planeObs.d = p.d;
                        planeObs.centroid = p.centroid;
                        planeObs.pointCount = p.pointCount;
                        station.mergedPlanes.push_back(planeObs);
                    }
                    for (const TrihedralCornerInfo& ci : info.cornerCandidates)
                    {
                        CornerCandidate corner;
                        corner.planeIndices = ci.planeIndices;
                        corner.point = ci.point;
                        corner.score = ci.score;
                        corner.angleResidualDeg = ci.angleResidualDeg;
                        corner.centroidPairResidualMm = ci.centroidPairResidualMm;
                        corner.centroidCornerResidualMm = ci.centroidCornerResidualMm;
                        corner.centroidDistanceOk = ci.centroidDistanceOk;
                        station.candidates.push_back(corner);
                    }

                    // Load plane point sets from the feature PCD file
                    const fs::path featurePcdPath = featureBaseDir / info.cloudPath.filename();
                    if (fs::exists(featurePcdPath))
                    {
                        LoadPlanePointsFromFeaturePcd(station, featurePcdPath);
                    }

                    if (!station.candidates.empty())
                    {
                        stationCandidates.push_back(std::move(station));
                    }
                }
            }
            else
            {
                // Read pose records and extract features from scratch
                PoseIo::ReadOptions poseOptions;
                poseOptions.errorStream = &std::cerr;
                const std::vector<PoseRecord> poseRecords = PoseIo::ReadPoseRecords(absoluteDataDir, poseOptions);
                if (poseRecords.empty())
                {
                    result.code = -2;
                    result.message = "No pose records found in: " + absoluteDataDir.string();
                    return result;
                }

                for (const PoseRecord& record : poseRecords)
                {
                    StationCandidates station = ExtractStationCandidates(record, params.featureParams);
                    if (!station.candidates.empty())
                    {
                        stationCandidates.push_back(std::move(station));
                    }
                }
            }
        }

        if (stationCandidates.size() < 3)
        {
            result.code = -3;
            result.message = "At least 3 valid stations are required, got "
                + std::to_string(stationCandidates.size());
            return result;
        }

        // Read target bases if not provided
        if (targetBases.empty())
        {
            fs::path tcpPath = absoluteDataDir / "corner.xyz";
            if (fs::exists(tcpPath))
            {
                targetBases = ReadTargetTcps(tcpPath);
            }
            
            if (targetBases.empty())
            {
                result.code = -4;
                result.message = "No target TCP points available";
                return result;
            }
        }

        const Eigen::Vector3d targetBase = targetBases.front();

        // Step 2: Global candidate selection
        std::vector<FeatureObservation> observations;
        std::optional<CalibrationResult> calibResult;
        if (targetBases.size() >= 2)
        {
            observations = SelectGloballyConsistentMultiTargetObservations(stationCandidates, targetBases);
            if (observations.size() < targetBases.size() * 3u)
            {
                result.code = -5;
                result.message = "No valid multi-target observation set for calibration";
                return result;
            }
            calibResult = Calibrate(observations, targetBases, params.enableTrihedralOptimization);
        }
        else
        {
            observations = SelectGloballyConsistentObservations(stationCandidates, targetBase);
            if (observations.size() < 3)
            {
                result.code = -5;
                result.message = "No valid observation set for calibration";
                return result;
            }
            calibResult = Calibrate(observations, targetBase, params.enableTrihedralOptimization);
        }

        if (!calibResult)
        {
            result.code = -6;
            result.message = "Calibration optimization failed";
            return result;
        }

        // Step 3: Iterative outlier rejection and re-calibration
        const double maxError = params.maxStationErrorMm;
        int removedCount = 0;
        constexpr int kMinStations = 3;

        if (maxError > 0.0001)//maxError > 0.0001
        {
            while (static_cast<int>(observations.size()) >= kMinStations)
            {
                // Find the station with the largest error
                size_t worstIdx = 0;
                double worstError = 0.0;
                for (size_t i = 0; i < calibResult->errors.size() && i < observations.size(); ++i)
                {
                    const double errNorm = calibResult->errors[i].norm();
                    if (errNorm > worstError)
                    {
                        worstError = errNorm;
                        worstIdx = i;
                    }
                }

                if (worstError <= maxError)
                {
                    break;  // All stations within tolerance
                }

                // Remove the worst station
                PCL_WARN("Removing outlier station %s: error=%.3f mm > max=%.3f mm\n",
                    observations[worstIdx].timestamp.c_str(), worstError, maxError);
                observations.erase(observations.begin() + static_cast<long>(worstIdx));
                ++removedCount;

                if (static_cast<int>(observations.size()) < kMinStations)
                {
                    break;
                }

                // Re-calibrate with the filtered set
                std::optional<CalibrationResult> recalibResult;
                if (targetBases.size() >= 2)
                {
                    recalibResult = Calibrate(observations, targetBases, params.enableTrihedralOptimization);
                }
                else
                {
                    recalibResult = Calibrate(observations, targetBase, params.enableTrihedralOptimization);
                }

                if (!recalibResult)
                {
                    PCL_WARN("Re-calibration failed after removing %d outliers\n", removedCount);
                    break;
                }
                calibResult = std::move(recalibResult);
            }
        }

        if (static_cast<int>(observations.size()) < kMinStations)
        {
            result.code = -7;
            result.message = "Too few stations after outlier removal: "
                + std::to_string(observations.size())
                + " (removed " + std::to_string(removedCount) + ")";
            return result;
        }

        result.removedStationCount = removedCount;

        // Step 4: Populate result
        const Eigen::Isometry3d& flangeFromCamera = calibResult->flangeFromCamera;
        result.flangeFromCamera = flangeFromCamera;
        result.cameraCenterInFlange = flangeFromCamera.translation();
        result.quaternion = Eigen::Quaterniond(flangeFromCamera.linear());
        const auto rpy = RpyFromRotation(flangeFromCamera.linear());
        result.rpyDeg = Eigen::Vector3d(rpy[0], rpy[1], rpy[2]);
        const Eigen::Matrix4d m = flangeFromCamera.matrix();
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 4; ++col)
                result.matrix4x4[static_cast<size_t>(row * 4 + col)] = m(row, col);

        result.stationCount = static_cast<int>(observations.size());
        result.targetCount = static_cast<int>(targetBases.size());
        result.targetBases = targetBases;

        // Nonlinear optimizer info
        result.nonlinearOptimizer.success = true;
        result.nonlinearOptimizer.initialCost = calibResult->optimizerInitialCost;
        result.nonlinearOptimizer.finalCost = calibResult->optimizerFinalCost;
        result.nonlinearOptimizer.meanErrorMm = calibResult->meanError;
        result.nonlinearOptimizer.rmsErrorMm = calibResult->rmsError;
        result.nonlinearOptimizer.maxErrorMm = calibResult->maxError;
        result.nonlinearOptimizer.briefReport = calibResult->optimizerBriefReport;

        // Trihedral bundle info
        result.trihedralBundle.attempted = calibResult->trihedralOptimizationAttempted;
        result.trihedralBundle.success = calibResult->trihedralOptimizationSuccess;
        result.trihedralBundle.stationCount = calibResult->trihedralStationCount;
        result.trihedralBundle.planeResidualCount = calibResult->trihedralPlaneResidualCount;
        result.trihedralBundle.cornerResidualCount = calibResult->trihedralCornerResidualCount;
        result.trihedralBundle.initialCost = calibResult->trihedralOptimizerInitialCost;
        result.trihedralBundle.finalCost = calibResult->trihedralOptimizerFinalCost;
        result.trihedralBundle.planeMeanAbsDistanceMm = calibResult->trihedralPlaneMeanAbsDistanceMm;
        result.trihedralBundle.planeRmsDistanceMm = calibResult->trihedralPlaneRmsDistanceMm;
        result.trihedralBundle.planeMaxAbsDistanceMm = calibResult->trihedralPlaneMaxAbsDistanceMm;
        result.trihedralBundle.cornerMeanErrorMm = calibResult->trihedralCornerMeanErrorMm;
        result.trihedralBundle.cornerRmsErrorMm = calibResult->trihedralCornerRmsErrorMm;
        result.trihedralBundle.cornerMaxErrorMm = calibResult->trihedralCornerMaxErrorMm;
        result.trihedralBundle.cornerBase = calibResult->trihedralCornerBase;
        result.trihedralBundle.cornerBaseAvailable = calibResult->trihedralCornerBaseAvailable;
        result.trihedralBundle.targetAlignmentDeltaMm = calibResult->trihedralTargetAlignmentTranslationDeltaMm;
        result.trihedralBundle.targetAlignmentRotationDeltaDeg = calibResult->trihedralTargetAlignmentRotationDeltaDeg;
        result.trihedralBundle.targetPoseOptimized = calibResult->trihedralTargetPoseOptimized;
        result.trihedralBundle.briefReport = calibResult->trihedralOptimizerBriefReport;

        result.meanErrorMm = calibResult->meanError;
        result.rmsErrorMm = calibResult->rmsError;
        result.maxErrorMm = calibResult->maxError;

        // Per-station errors
        for (size_t i = 0; i < observations.size() && i < calibResult->errors.size(); ++i)
        {
            StationCornerError se;
            se.timestamp = observations[i].timestamp;
            se.cameraPoint = observations[i].cameraPoint;
            se.predictedBase = calibResult->basePredictions[i];
            se.error = calibResult->errors[i];
            se.errorNorm = calibResult->errors[i].norm();
            se.targetIndex = observations[i].targetIndex;
            se.planeIndices = observations[i].planeIndices;
            se.candidateScore = observations[i].candidateScore;
            se.centroidDistanceOk = observations[i].centroidDistanceOk;
            result.stationErrors.push_back(se);
        }

        // Step 4: Write outputs
        const fs::path outputDir = params.outputDir.empty() ? absoluteDataDir : params.outputDir;
        fs::create_directories(outputDir);

        const fs::path reportPath = outputDir / "handeye_calibration_result.txt";
        {
            std::ofstream report(reportPath);
            if (report)
            {
                WriteReport(report, absoluteDataDir, targetBase, observations, *calibResult);
            }
            std::ostringstream reportStr;
            WriteReport(reportStr, absoluteDataDir, targetBase, observations, *calibResult);
            result.reportText = reportStr.str();
        }
        WriteReport(std::cout, absoluteDataDir, targetBase, observations, *calibResult);

        result.reportPath = reportPath;

        const fs::path matrixPath = outputDir / "handeye_matrix.txt";
        WriteHandEyeMatrix(flangeFromCamera, matrixPath);
        result.matrixPath = matrixPath;

        if (params.exportBaseClouds)
        {
            const fs::path baseDir = params.baseCloudOutputDir.empty()
                ? absoluteDataDir / "base"
                : params.baseCloudOutputDir;
            fs::create_directories(baseDir);
            // Export ALL scanned PCDs to base coordinates (not just the
            // calibration observations).  This reuses the same transform
            // logic as RunCameraCloudToBaseMode:
            //   baseFromCamera = baseFromFlange * flangeFromCamera
            PoseIo::ReadOptions allPoseOptions;
            allPoseOptions.errorStream = &std::cerr;
            const std::vector<PoseRecord> allPoseRecords = PoseIo::ReadPoseRecords(absoluteDataDir, allPoseOptions);

            size_t exportedCount = 0;
            size_t failedCount = 0;
            for (const PoseRecord& record : allPoseRecords)
            {
                const fs::path outputPath = baseDir / record.cloudPath.filename();
                const Eigen::Isometry3d baseFromCamera = record.baseFromFlange * flangeFromCamera;
                if (PointCloudIo::TransformPcdFileToBase(record.cloudPath, outputPath, baseFromCamera, &std::cerr))
                {
                    ++exportedCount;
                }
                else
                {
                    ++failedCount;
                }
            }

            PCL_INFO("Exported %zu/%zu PCD files to base coordinates. failed=%zu\n",
                exportedCount, allPoseRecords.size(), failedCount);
        }

        if (params.exportFeatureBaseClouds)
        {
            const fs::path featureBaseDir = params.featureBaseOutputDir.empty()
                ? absoluteDataDir / "featureBase"
                : params.featureBaseOutputDir;
            ExportFeatureCloudsToBase(stationCandidates, *calibResult, featureBaseDir);
        }

        result.code = 0;
        result.message = "Calibration completed: rms=" + std::to_string(calibResult->rmsError) + " mm"
            + (removedCount > 0 ? " (removed " + std::to_string(removedCount) + " outliers)" : "");
    }
    catch (const std::exception& ex)
    {
        result.code = -99;
        result.message = std::string("RunCalibrate exception: ") + ex.what();
    }
    return result;
}


// ============================================================================
// Module 4 – Multiblock mode (public API)
// ============================================================================


MultiBlockCalibrateResult RunMultiBlockCalibrate(const MultiBlockCalibrateParams& params)
{
    MultiBlockCalibrateResult result;
    try
    {
        const fs::path absoluteDataDir = fs::absolute(params.dataDir);
        if (!fs::exists(absoluteDataDir) || !fs::is_directory(absoluteDataDir))
        {
            result.code = -1;
            result.message = "Data directory does not exist: " + absoluteDataDir.string();
            return result;
        }

        // Load initial hand-eye matrix
        Eigen::Isometry3d initialFlangeFromCamera = Eigen::Isometry3d::Identity();
        fs::path resolvedMatrixPath = params.initialMatrixPath;
        if (resolvedMatrixPath.empty())
        {
            resolvedMatrixPath = params.outputDir.empty()
                ? absoluteDataDir.parent_path() / "handeye_matrix.txt"
                : params.outputDir / "handeye_matrix.txt";
        }
        if (fs::exists(resolvedMatrixPath) && ReadHandEyeInitialMatrix(resolvedMatrixPath, initialFlangeFromCamera))
        {
            result.initialMatrixWasLoaded = true;
        }
        result.initialFlangeFromCamera = initialFlangeFromCamera;

        const bool useInitialForMatching = result.initialMatrixWasLoaded
            && IsUsableInitialHandEye(initialFlangeFromCamera);
        result.matchingStrategy = useInitialForMatching ? "initial_handeye" : "adjacent_gicp";

        // Read pose records
        PoseIo::ReadOptions poseOptions;
        poseOptions.errorStream = &std::cerr;
        const std::vector<PoseRecord> poseRecords = PoseIo::ReadPoseRecords(absoluteDataDir, poseOptions);
        if (poseRecords.empty())
        {
            result.code = -2;
            result.message = "No pose records found in: " + absoluteDataDir.string();
            return result;
        }

        // Extract multiblock plane stations
        std::vector<MultiBlockPlaneStation> planeStations;
        for (const PoseRecord& record : poseRecords)
        {
            auto station = ExtractMultiBlockPlaneStation(record);
            if (station)
            {
                planeStations.push_back(std::move(*station));
            }
        }

        if (planeStations.size() < 3)
        {
            result.code = -3;
            result.message = "At least 3 valid multiblock stations are required, got "
                + std::to_string(planeStations.size());
            return result;
        }

        result.stationCount = static_cast<int>(planeStations.size());

        // Cross-station matching
        std::vector<MultiBlockMatchedStation> matchedStations = SelectMultiBlockMatchedStations(
            planeStations, initialFlangeFromCamera, useInitialForMatching);
        if (matchedStations.size() < 3)
        {
            result.code = -4;
            result.message = "No globally consistent multiblock plane matching was found";
            return result;
        }

        // Per-station match info
        for (const MultiBlockMatchedStation& matched : matchedStations)
        {
            MultiBlockStationMatchInfo matchInfo;
            matchInfo.timestamp = matched.optimizerStation.timestamp;
            matchInfo.sourcePlaneIndices = matched.sourcePlaneIndices;
            matchInfo.score = matched.score;
            matchInfo.minMatchedNormalDot = matched.minMatchedNormalDot;
            matchInfo.matchingMode = matched.matchedByGicp ? "gicp"
                : (useInitialForMatching ? "initial" : "reference");
            matchInfo.gicpFitnessMm2 = matched.gicpFitnessMm2;
            result.stationMatches.push_back(matchInfo);
        }

        // Optimize
        std::vector<HandEyeMultiBlockPlaneOptimizer::StationObservation> optimizerStations;
        for (const MultiBlockMatchedStation& matched : matchedStations)
        {
            optimizerStations.push_back(matched.optimizerStation);
        }

        HandEyeMultiBlockPlaneOptimizer::Options options;
        options.minStationCount = 3;
        options.minBlockCount = 2;
        options.minFacesPerBlock = 2;
        options.huberLossMm = params.planeHuberLossMm;
        options.requireFullTranslationRank = true;
        options.maxNumIterations = params.maxIterations;
        options.minimizerProgressToStdout = params.minimizerProgressToStdout;
        const HandEyeMultiBlockPlaneOptimizer optimizer(options);
        const HandEyeMultiBlockPlaneOptimizer::Result optResult = optimizer.Optimize(
            optimizerStations, initialFlangeFromCamera);

        // Populate result
        result.flangeFromCamera = optResult.flangeFromCamera;
        result.cameraCenterInFlange = optResult.flangeFromCamera.translation();
        result.quaternion = Eigen::Quaterniond(optResult.flangeFromCamera.linear());
        const auto rpy = RpyFromRotation(optResult.flangeFromCamera.linear());
        result.rpyDeg = Eigen::Vector3d(rpy[0], rpy[1], rpy[2]);
        const Eigen::Matrix4d m = optResult.flangeFromCamera.matrix();
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 4; ++col)
                result.matrix4x4[static_cast<size_t>(row * 4 + col)] = m(row, col);

        result.optimizationSuccess = optResult.success;
        result.initialCost = optResult.initialCost;
        result.finalCost = optResult.finalCost;
        result.residualCount = optResult.residualCount;
        result.translationConstraintRank = optResult.translationConstraintRank;
        result.planeMeanAbsDistanceMm = optResult.meanAbsDistanceMm;
        result.planeRmsDistanceMm = optResult.rmsDistanceMm;
        result.planeMaxAbsDistanceMm = optResult.maxAbsDistanceMm;
        result.normalMeanAngleDeg = optResult.normalMeanAngleDeg;
        result.normalMaxAngleDeg = optResult.normalMaxAngleDeg;
        result.optimizerBriefReport = optResult.briefReport;

        for (const auto& face : optResult.faces)
        {
            MultiBlockFaceResultInfo fr;
            fr.blockIndex = face.blockIndex;
            fr.faceIndex = face.faceIndex;
            fr.blockNormal = face.blockNormal;
            fr.normalBase = face.normalBase;
            fr.dBase = face.dBase;
            fr.observationCount = face.observationCount;
            fr.residualCount = face.residualCount;
            fr.rmsDistanceMm = face.rmsDistanceMm;
            fr.maxAbsDistanceMm = face.maxAbsDistanceMm;
            result.faceResults.push_back(fr);
        }

        // Write outputs
        const fs::path outputDir = params.outputDir.empty() ? absoluteDataDir : params.outputDir;
        fs::create_directories(outputDir);

        const fs::path reportPath = outputDir / "handeye_multiblock_result.txt";
        {
            std::ofstream report(reportPath);
            if (report)
            {
                WriteMultiBlockReport(report, absoluteDataDir, initialFlangeFromCamera,
                    matchedStations, optResult);
            }
            std::ostringstream reportStr;
            WriteMultiBlockReport(reportStr, absoluteDataDir, initialFlangeFromCamera,
                matchedStations, optResult);
            result.reportText = reportStr.str();
        }
        result.reportPath = reportPath;

        if (optResult.success)
        {
            const fs::path matrixPath = outputDir / "handeye_matrix.txt";
            WriteHandEyeMatrix(optResult.flangeFromCamera, matrixPath);
            result.matrixPath = matrixPath;
            result.code = 0;
            result.message = "Multiblock calibration completed: rms="
                + std::to_string(optResult.rmsDistanceMm) + " mm";
        }
        else
        {
            result.code = -5;
            result.message = "Multiblock optimization did not produce a usable full-rank solution";
        }
    }
    catch (const std::exception& ex)
    {
        result.code = -99;
        result.message = std::string("RunMultiBlockCalibrate exception: ") + ex.what();
    }
    return result;
}


// ============================================================================
// Legacy wrappers (backward-compatible)
// ============================================================================

int RunHandEyeCalibration(const std::filesystem::path& dataDir, const std::filesystem::path& executableDir, const handeye::AppConfig& config)
{
    handeye::CalibrateParams params;
    params.dataDir = dataDir;
    params.outputDir = dataDir;
    params.exportBaseClouds = true;
    params.baseCloudOutputDir = fs::absolute(dataDir) / "base";
    params.exportFeatureBaseClouds = config.exportFeatureCloud;
    params.featureBaseOutputDir = fs::absolute(dataDir) / "featureBase";
    params.enableTrihedralOptimization = true;

    const handeye::CalibrateResult result = handeye::RunCalibrate(params);
    if (result.code != 0)
    {
        std::cerr << result.message << "\n";
        return 1;
    }

    std::cout << "Report written: " << result.reportPath.string() << "\n";
    std::cout << "Base-coordinate PCD files written: "
        << fs::absolute(dataDir).parent_path() / "base" << "\n";
    if (config.exportFeatureCloud)
    {
        std::cout << "Base-coordinate feature PCD files written: "
            << fs::absolute(dataDir).parent_path() / "featureBase" << "\n";
    }
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
            std::cout << result.flangeFromCamera.matrix()(row, col);
        }
        std::cout << "\n";
    }
    std::cout << "Hand-eye matrix written: " << result.matrixPath.string() << "\n";
    return 0;
}

int RunHandEyeMultiBlockCalibration(
    const std::filesystem::path& dataDir,
    const std::filesystem::path& executableDir,
    const std::filesystem::path& initialMatrixPath)
{
    handeye::MultiBlockCalibrateParams params;
    params.dataDir = dataDir;
    params.outputDir = executableDir;
    params.initialMatrixPath = initialMatrixPath;

    const handeye::MultiBlockCalibrateResult result = handeye::RunMultiBlockCalibrate(params);
    if (result.code != 0)
    {
        std::cerr << result.message << "\n";
        return 1;
    }

    std::cout << "\nMulti-block report written: " << result.reportPath.string() << "\n";
    std::cout << "Hand-eye matrix written: " << result.matrixPath.string() << "\n";
    return 0;
}
}
