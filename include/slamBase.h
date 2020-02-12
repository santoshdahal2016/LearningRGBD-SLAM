// C ++ Standard Library 
#include <iostream>
#include <string>
#include <vector>
#include <map>

using namespace std;

// OpenCV library 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

// PCL library 
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

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



// Parameter reading class
class ParameterReader
{
public:
    ParameterReader( string filename="./parameters.txt" )
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

            if ( !fin.good() )
                break;
        }
    }
    string getData( string key )
    {
        map<string, string>::iterator iter = data.find(key);
        if (iter == data.end())
        {
            cerr<<"Parameter name "<<key<<" not found!"<<endl;
            return string("NOT_FOUND");
        }
        return iter->second;
    }
public:
    map<string, string> data;
};