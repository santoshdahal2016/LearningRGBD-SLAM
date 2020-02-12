#include "slamBase.h"

PointCloud::Ptr image2PointCloud( cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    // Point cloud variable
    // Use smart pointer to create an empty point cloud. This pointer will be released automatically when used up. 
    PointCloud::Ptr cloud ( new PointCloud );
  
  	cout<<"-> Before loop of reading depth."<<endl;

    // traverse the depth map 
    for (int m = 0; m < depth.rows; m++)
        for (int n=0; n < depth.cols; n++)
        {
            // Get the value at (m, n) in the depth map 
            ushort d = depth.ptr<ushort>(m)[n];
            // d may have no value, if so, skip this point
            if (d == 0)
                continue;
            // If d exists, add a point to the point cloud 
            PointT p;

            // Calculate the space coordinates of this point 
            p.z = double(d) / camera.scale;
            p.x = (n - camera.cx) * p.z / camera.fx;
            p.y = (m - camera.cy) * p.z / camera.fy;
            
            // Get its color from the rgb image
            // rgb is a three-channel BGR format map, so Get colors in the following order             
            p.b = rgb.ptr<uchar>(m)[n*3];
            p.g = rgb.ptr<uchar>(m)[n*3+1];
            p.r = rgb.ptr<uchar>(m)[n*3+2];

            // Add p to the point cloud in 
            cloud->points.push_back(p);
        }
    // set and saved point cloud 
    cout<<"-> After loop"<<endl;
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cout<<"point cloud size = "<<cloud->points.size()<<endl;
    cloud->is_dense = false;
    return cloud;
}

cv::Point3f point2dTo3d( cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    cv::Point3f p; // 3D Point
    p.z = double( point.z ) / camera.scale;
    p.x = ( point.x - camera.cx) * p.z / camera.fx;
    p.y = ( point.y - camera.cy) * p.z / camera.fy;
    return p;
}


// computeKeyPointsAndDesp extracts both key points and feature descriptors
void computeKeyPointsAndDesp( FRAME& frame)
{
    cv::Ptr<cv::FeatureDetector> _detector;
    cv::Ptr<cv::DescriptorExtractor> _descriptor;

    _detector = cv::ORB::create();
    _descriptor = cv::ORB::create();


    _detector->detect( frame.rgb, frame.kp );
    _descriptor->compute( frame.rgb, frame.kp, frame.desp );

    return;
}

// estimateMotion calculates the motion between two frames
// Input: Frame 1 and Frame 2
// Output: rvec and tvec
RESULT_OF_PNP estimateMotion( FRAME& frame1, FRAME& frame2, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    static ParameterReader pd;
    vector< cv::DMatch > matches;
    cv::BFMatcher matcher;
    matcher.match( frame1.desp, frame2.desp, matches );
   
    cout<<"find total "<<matches.size()<<" matches."<<endl;
    vector< cv::DMatch > goodMatches;
    double minDis = 9999;
    double good_match_threshold = atof( pd.getData( "good_match_threshold" ).c_str() );
    for ( size_t i=0; i<matches.size(); i++ )
    {
        if ( matches[i].distance < minDis )
            minDis = matches[i].distance;
    }

    for ( size_t i=0; i<matches.size(); i++ )
    {
        if (matches[i].distance < good_match_threshold*minDis)
            goodMatches.push_back( matches[i] );
    }

    cout<<"good matches: "<<goodMatches.size()<<endl;
    // 3D point of the first frame
    vector<cv::Point3f> pts_obj;
    // Image point of the second frame
    vector< cv::Point2f > pts_img;

    // Camera internal parameters
    for (size_t i=0; i<goodMatches.size(); i++)
    {
        // query is the first and train is the second
        cv::Point2f p = frame1.kp[goodMatches[i].queryIdx].pt;
        // Be careful about getting d! x is right, y is down, so y is the row, x is the column!
        ushort d = frame1.depth.ptr<ushort>( int(p.y) )[ int(p.x) ];
        if (d == 0)
            continue;
        pts_img.push_back( cv::Point2f( frame2.kp[goodMatches[i].trainIdx].pt ) );

        // Convert (u, v, d) to (x, y, z)
        cv::Point3f pt ( p.x, p.y, d );
        cv::Point3f pd = point2dTo3d( pt, camera );
        pts_obj.push_back( pd );
    }

    double camera_matrix_data[3][3] = {
        {camera.fx, 0, camera.cx},
        {0, camera.fy, camera.cy},
        {0, 0, 1}
    };

    cout<<"solving pnp"<<endl;
    // Build Camera Matrix
    cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
    cv::Mat rvec, tvec, inliers;
    // Slove pnp
    cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, 0.99, inliers );

    RESULT_OF_PNP result;
    result.rvec = rvec;
    result.tvec = tvec;
    result.inliers = inliers.rows;

    return result;
}