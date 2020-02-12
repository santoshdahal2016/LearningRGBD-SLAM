#include<iostream>
using namespace std;

#include "slamBase.h"

#include <opencv2/core/eigen.hpp>

#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>

// Eigen !
#include <Eigen/Core>
#include <Eigen/Geometry>

int main( int argc, char** argv )
{
    ParameterReader pd;

    FRAME frame1, frame2;
    
    frame1.rgb = cv::imread( "../data/rgb1.png" );
    frame1.depth = cv::imread( "../data/depth1.png", -1);
    frame2.rgb = cv::imread( "../data/rgb2.png" );
    frame2.depth = cv::imread( "../data/depth2.png", -1 );


    computeKeyPointsAndDesp( frame1);
    computeKeyPointsAndDesp( frame2);

    CAMERA_INTRINSIC_PARAMETERS camera;
    camera.fx = atof( pd.getData( "camera.fx" ).c_str());
    camera.fy = atof( pd.getData( "camera.fy" ).c_str());
    camera.cx = atof( pd.getData( "camera.cx" ).c_str());
    camera.cy = atof( pd.getData( "camera.cy" ).c_str());
    camera.scale = atof( pd.getData( "camera.scale" ).c_str() );

    cout<<"solving pnp"<<endl;

    RESULT_OF_PNP result = estimateMotion( frame1, frame2, camera );

    cout<<result.rvec<<endl<<result.tvec<<endl;

    // Process result
    // Convert the rotation vector into a rotation matrix
    cv::Mat R;
    cv::Rodrigues( result.rvec, R );
    Eigen::Matrix3d r;
    cv::cv2eigen(R, r);
  
    // Convert the translation vector and rotation matrix into a transformation matrix
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

    Eigen::AngleAxisd angle(r);
   
    T = angle;
    T(0,3) = result.tvec.at<double>(0,0); 
    T(1,3) = result.tvec.at<double>(0,1); 
    T(2,3) = result.tvec.at<double>(0,2);

    // Convert the point cloud
    cout<<"converting image to clouds"<<endl;
    PointCloud::Ptr cloud1 = image2PointCloud( frame1.rgb, frame1.depth, camera );
    PointCloud::Ptr cloud2 = image2PointCloud( frame2.rgb, frame2.depth, camera );

    // Merge point clouds
    cout<<"combining clouds"<<endl;
    PointCloud::Ptr output (new PointCloud());
    pcl::transformPointCloud( *cloud1, *output, T.matrix() );
    *output += *cloud2;
    pcl::io::savePCDFile("./result.pcd", *output);
    cout<<"Final result saved."<<endl;

    pcl::visualization::CloudViewer viewer( "viewer" );
    viewer.showCloud( output );
    while( !viewer.wasStopped() )
    {
        
    }
    return 0;
}