#pragma once
#include <vector>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/ModelCoefficients.h>

namespace my3d
{
	struct ShapeObject
	{
		pcl::ModelCoefficients::Ptr coeffi;
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
		std::vector<int> indices;
		pcl::SacModel shape_type;
	};

class __declspec(dllexport) ShapeDetector
{
public:
	ShapeDetector() = default;
	~ShapeDetector() = default;
	
	
	int doDetect(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::SacModel shape_type,std::vector<ShapeObject>& out_shapes,
		pcl::PointCloud<pcl::PointXYZ>::Ptr& left_over);

	int    min_point_count_ = 200;

	double sphere_min_radius_ = 3;
	double sphere_max_radius_ = 9999999.0;

	double cylinder_min_radius_ = 3;
	double cylinder_max_radius_ = 9999999.0;

	double epsilon_ = 1.5;
	double bitmap_epsilon_ = 3.0;
	double normal_thresh_ = 25.0;
	double probability_ = 0.01;

};
}
