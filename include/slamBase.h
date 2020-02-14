// C ++ Standard Library 
#include <iostream>
#include <string>
#include <vector>
#include <map>

using namespace std;


// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>


// OpenCV library 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

// PCL library 
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
// voxel_grid is used for downsampling
#include <pcl/filters/voxel_grid.h>


// Define the point cloud type 
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud; 



// Camera internal 
struct CAMERA_INTRINSIC_PARAMETERS 
{ 
    double cx, cy, fx, fy, scale;
};

// Single Image Frame Structure
struct FRAME
{
    int frameID;
    cv::Mat rgb, depth; 
    cv::Mat desp;       
    vector<cv::KeyPoint> kp; 
};

// PnP Structure
struct RESULT_OF_PNP
{
    cv::Mat rvec, tvec;
    int inliers;
};


// Function interface
// image2PonitCloud converts rgb image to point cloud 
PointCloud::Ptr image2PointCloud( cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera );

// point2dTo3d converts a single point from image coordinates to spatial coordinates
// input: 3-dimensional point Point3f (u, v, d) 
cv::Point3f point2dTo3d( cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera );



// computeKeyPointsAndDesp extracts both key points and feature descriptors
void  computeKeyPointsAndDesp (FRAME & frame);

// estimateMotion calculates the motion between two frames
// input: frame 1 and frame 2, camera internal parameters
RESULT_OF_PNP estimateMotion (FRAME & frame1, FRAME & frame2, CAMERA_INTRINSIC_PARAMETERS & camera);

// cvMat2Eigen
Eigen::Isometry3d cvMat2Eigen( cv::Mat& rvec, cv::Mat& tvec );

// joinPointCloud 
PointCloud::Ptr joinPointCloud( PointCloud::Ptr original, FRAME& newFrame, Eigen::Isometry3d T, CAMERA_INTRINSIC_PARAMETERS& camera ) ;


// Parameter reading class
class ParameterReader
{
public:
    ParameterReader( string filename="../parameters.txt" )
    {
        ifstream fin( filename.c_str() );
        if (!fin)
        {
            cerr<<"parameter file does not exist."<<endl;
            return;
        }
        while(!fin.eof())
        {
            string str;
            getline( fin, str );
            if (str[0] == '#')
            {
                // Comments starting with '#'
                continue;
            }

            int pos = str.find("=");
            if (pos == -1)
                continue;
            string key = str.substr( 0, pos );
            string value = str.substr( pos+1, str.length() );
            data[key] = value;

            if ( !fin.good() ){
                break;

            }



        }
    }

    string getData(string key)
    {
		// map<string, string>::iterator iter1 = data.find("good_match_threshold");
  //       if (iter1 == data.end())
  //       {
  //           cerr<<"Parameter name "<<"good_match_threshold"<<" not found!"<<endl;
  //           return string("NOT_FOUND");
  //       }else{
  //       	cout<<iter1->first<<endl;
  //       }

		for (const auto &p : data) {
			// cout<<p.first<<key<<endl;
			// cout<<"strcmp :"<<strcmp(p.first.c_str(), key.c_str())<<endl;
			if(strcmp(p.first.c_str(), key.c_str()) == 32){
				 // cout<<p.second<<endl;
				 return p.second;
			}
		}

		cerr<<"Parameter name "<<key<<" not found!"<<endl;
		return string("NOT_FOUND");

        // map<string, string>::iterator iter = data.find(key);
        // if (iter == data.end())
        // {
        //     cerr<<"Parameter name "<<key<<" not found!"<<endl;
        //     return string("NOT_FOUND");
        // }
    }

public:
    map<string, string> data;
};

inline static CAMERA_INTRINSIC_PARAMETERS getDefaultCamera()
{
    ParameterReader pd;
    CAMERA_INTRINSIC_PARAMETERS camera;
    camera.fx = atof( pd.getData( "camera.fx" ).c_str());
    camera.fy = atof( pd.getData( "camera.fy" ).c_str());
    camera.cx = atof( pd.getData( "camera.cx" ).c_str());
    camera.cy = atof( pd.getData( "camera.cy" ).c_str());
    camera.scale = atof( pd.getData( "camera.scale" ).c_str() );
    return camera;
}