#include "shape_detector.h"
#include "weld_segment.h"

#ifdef _DEBUG_PCD
#include <pcl/io/pcd_io.h>
#endif
#include <pcl/console/print.h>

#include <RansacShapeDetector.h>
#include <PlanePrimitiveShape.h>
#include <PlanePrimitiveShapeConstructor.h>
#include <CylinderPrimitiveShape.h>
#include <CylinderPrimitiveShapeConstructor.h>
#include <SpherePrimitiveShape.h>
#include <SpherePrimitiveShapeConstructor.h>

using namespace weldAlgo;
namespace my3d
{
    typedef std::pair<MiscLib::RefCountPtr<PrimitiveShape>, size_t> DetectedShape;

	struct ShapePara
	{
		double sphere_min_radius = 3.0;
		double sphere_max_radius = std::numeric_limits<float>::max();

		double cylinder_min_radius = 3.0;
		double cylinder_max_radius = std::numeric_limits<float>::max();
		double cylinder_max_length = std::numeric_limits<float>::max();
	};

    int ransac_sd(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, MiscLib::Vector<DetectedShape>& shapes, 
		PointCloud& rsd_cloud, pcl::SacModel shape_type, const ShapePara& shape_para,
        int minPoints = 200,double normalThresh = 25.0,double epsilon = 2.0, double bitmapEpsilon = 4.0,double probability = 0.01)
    {
        if (!cloud || cloud->empty())
        {
			if(weldSegment::log_level_ <= LOG_LEVEL_WARN)
            PCL_WARN("ransac_sd:input pointcloud is empty!");
            return 0;
        }
        if (shape_type != pcl::SACMODEL_PLANE && shape_type != pcl::SACMODEL_SPHERE && shape_type != pcl::SACMODEL_CYLINDER)
        {
			if(weldSegment::log_level_ <= LOG_LEVEL_WARN)
            PCL_WARN("ransac_sd:unsupport shape type!");
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
        
        RansacShapeDetector::Options option;
        option.m_minSupport = minPoints;
        option.m_normalThresh = static_cast<float>(cos(DEG2RAD(normalThresh)));
        option.m_probability = probability;
        option.m_epsilon = epsilon;
        option.m_bitmapEpsilon = bitmapEpsilon;
        option.m_allowSimplification = true;
        RansacShapeDetector detector(option);
		if (shape_type == pcl::SACMODEL_PLANE)
		{
			detector.Add(new PlanePrimitiveShapeConstructor());
		}
		else if (shape_type == pcl::SACMODEL_SPHERE)
		{
			detector.Add(new SpherePrimitiveShapeConstructor(shape_para.sphere_min_radius, shape_para.sphere_max_radius));
		}
		else if (shape_type == pcl::SACMODEL_CYLINDER)
		{
			detector.Add(new CylinderPrimitiveShapeConstructor(shape_para.cylinder_min_radius, shape_para.cylinder_max_radius, shape_para.cylinder_max_length));
		}
		else
			return 0;
        shapes.clear();
        detector.Detect(rsd_cloud, 0, rsd_cloud.size(), &shapes);

        return shapes.size() > 0 ? 1 : 0;
    }

	int ShapeDetector::doDetect(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::SacModel shape_type,
		std::vector<ShapeObject>& out_shapes, pcl::PointCloud<pcl::PointXYZ>::Ptr& left_over)
	{
		out_shapes.clear();
		if (!cloud || cloud->empty())
			return 0;

		MiscLib::Vector<DetectedShape> shapes;
		PointCloud rsd_cloud;
		ShapePara shape_para;
		shape_para.sphere_min_radius = sphere_min_radius_;
		shape_para.sphere_max_radius = sphere_max_radius_;
		shape_para.cylinder_min_radius = cylinder_min_radius_;
		shape_para.cylinder_max_radius = cylinder_max_radius_;
		if (!ransac_sd(cloud, shapes, rsd_cloud, shape_type, 
			shape_para,min_point_count_, normal_thresh_, epsilon_, bitmap_epsilon_, probability_))
			return 0;

		int count = static_cast<int>(rsd_cloud.size());
		std::vector<uint8_t> assigned(count, 0);

		for (const auto& ds : shapes)
		{
			ShapeObject obj;
			obj.cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
			obj.coeffi.reset(new pcl::ModelCoefficients());
			obj.shape_type = shape_type;
			obj.indices.clear();

			const PrimitiveShape* shape = ds.first;
			size_t shapePointsCount = ds.second;

			//too many points?!
			if (shapePointsCount > count)
			{
				if(weldSegment::log_level_ <= LOG_LEVEL_WARN)
				PCL_ERROR("[ShapeDetector::doDetect] Inconsistent result!\n");
				break;
			}

			if (shapePointsCount < min_point_count_)
			{
				if(weldSegment::log_level_ <= LOG_LEVEL_WARN)
				PCL_WARN("[ShapeDetector::doDetect] Skipping shape, did not meet minimum point requirement(%zd/%d)\n", shapePointsCount, min_point_count_);
				count -= shapePointsCount;
				continue;
			}

			// 提取模型参数
			if (shape_type == pcl::SACMODEL_PLANE)
			{
				const PlanePrimitiveShape* plane = dynamic_cast<const PlanePrimitiveShape*>(shape);
				if (plane)
				{
					obj.coeffi->values.resize(4);
					obj.coeffi->values[0] = plane->Internal().getNormal()[0];
					obj.coeffi->values[1] = plane->Internal().getNormal()[1];
					obj.coeffi->values[2] = plane->Internal().getNormal()[2];
					obj.coeffi->values[3] = -plane->Internal().SignedDistToOrigin();
					// coefficients are available in obj.coeffi
				}
			}
			else if (shape_type == pcl::SACMODEL_SPHERE)
			{
				const SpherePrimitiveShape* sphere = dynamic_cast<const SpherePrimitiveShape*>(shape);
				if (sphere)
				{
					obj.coeffi->values.resize(4);
					obj.coeffi->values[0] = sphere->Internal().Center()[0];
					obj.coeffi->values[1] = sphere->Internal().Center()[1];
					obj.coeffi->values[2] = sphere->Internal().Center()[2];
					obj.coeffi->values[3] = sphere->Internal().Radius();
					// coefficients are available in obj.coeffi
				}
			}
			else if (shape_type == pcl::SACMODEL_CYLINDER)
			{
				const CylinderPrimitiveShape* cyl = dynamic_cast<const CylinderPrimitiveShape*>(shape);
				if (cyl)
				{
					obj.coeffi->values.resize(7);
					Vec3f G = cyl->Internal().AxisPosition();
					Vec3f N = cyl->Internal().AxisDirection();
					obj.coeffi->values[0] = G[0];
					obj.coeffi->values[1] = G[1];
					obj.coeffi->values[2] = G[2];
					obj.coeffi->values[3] = N[0];
					obj.coeffi->values[4] = N[1];
					obj.coeffi->values[5] = N[2];
					obj.coeffi->values[6] = cyl->Internal().Radius();
					// coefficients are available in obj.coeffi
				}
			}

			// 提取点云和索引
			int shapeCloudIndex = count - 1;
			for (size_t j = 0; j < shapePointsCount; ++j)
			{
				const Point& pt = rsd_cloud[shapeCloudIndex - j];
				obj.cloud->push_back(pcl::PointXYZ(pt.pos[0], pt.pos[1], pt.pos[2]));
				obj.indices.push_back(pt.index);
				assigned[shapeCloudIndex - j] = 1;
			}
			count -= static_cast<int>(shapePointsCount);
			out_shapes.push_back(obj);
		}

		// 输出剩余点集
		left_over.reset(new pcl::PointCloud<pcl::PointXYZ>());
		if (count > 0)
		{
			left_over->reserve(count);
		}
		for (size_t i = 0; i < rsd_cloud.size(); ++i)
		{
			if (!assigned[i])
			{
				const Point& pt = rsd_cloud[i];
				left_over->push_back(pcl::PointXYZ(pt.pos[0], pt.pos[1], pt.pos[2]));
			}
		}
		return static_cast<int>(out_shapes.size());
	}
}
