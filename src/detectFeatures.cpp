#include<iostream>
#include "slamBase.h"
using namespace std;

// OpenCV Feature Detection module
#include <opencv2/features2d/features2d.hpp>
// #include <opencv2/nonfree/nonfree.hpp> // use this if you want to use SIFT or SURF
#include <opencv2/calib3d/calib3d.hpp>

int main( int argc, char** argv )
{
	// Declare and read two rgb and depth maps from the data folder    
	cv::Mat rgb1 = cv::imread( "../data/rgb1.png");
    cv::Mat rgb2 = cv::imread( "../data/rgb2.png");
    cv::Mat depth1 = cv::imread( "../data/depth1.png", -1);
    cv::Mat depth2 = cv::imread( "../data/depth2.png", -1);

    // declare feature exctrator and the excrator decriptor
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> descriptor;

    // SIFT, SURF , ORB
    
    // enable sift, surf  for non free module
    // cv::initModule_nonfree();
    // _detector = cv::FeatureDetector::create( "SIFT" );
    // _descriptor = cv::DescriptorExtractor::create( "SIFT" );
    

// Detecting Features and descriptor calculation

// To detect feature in image we need to 
// 1. Calcualte Keypoints
    // Keypoints Structure : Point2f _pt, float _size, float _angle=-1, float _response=0, int _octave=0, int _class_id=-1
// 2. Calculate Discriptor for the pixel around the keypoints
    // Discriptor is a matrix structure of cv::mat each row of which represents a feature vector corresponding to the Keypoint , The more similar the descriptors of the two keypoints, the more similar the two keypoints are.


    detector = cv::ORB::create();
    descriptor = cv::ORB::create();

    vector< cv::KeyPoint > kp1, kp2; //key  point
    detector->detect( rgb1, kp1 );  // key descriptor
    detector->detect( rgb2, kp2 );

    cout<<"Key points of two images: "<<kp1.size()<<", "<<kp2.size()<<endl;
    
    // Visulizing Keypoint
    cv::Mat imgShow;
    cv::drawKeypoints( rgb1, kp1, imgShow, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    cv::imshow( "keypoints", imgShow );
    cv::imwrite( "../data/keypoints.png", imgShow );
    cv::waitKey(0); //Pause for key
   
    // Calcule Keypoints
    cv::Mat desp1, desp2;
    descriptor->compute( rgb1, kp1, desp1 );
    descriptor->compute( rgb2, kp2, desp2 );

// Feature matching : 
    // Match descriptor : In OpenCV, you need to choose a matching algorithm, such as bruteforce, Fast Library for Approximate Nearest Neighbour (FLANN), and so on. Here we build a FLANN matching algorithm:
    // 　After the match is complete, the algorithm returns some DMatch structures. The structure contains the following members:
		// queryIdx The index of the source feature descriptor (that is, the first image).
		// trainIdx index of the target feature descriptor (second image)
		// distance Matching distance, the larger the worse the match.

    vector< cv::DMatch > matches; 
    cv::BFMatcher matcher;
    matcher.match( desp1, desp2, matches );
    cout<<"Find total "<<matches.size()<<" matches."<<endl;

    // Visualization : show matching features
    cv::Mat imgMatches;
    cv::drawMatches( rgb1, kp1, rgb2, kp2, matches, imgMatches );
    cv::imshow( "matches", imgMatches );
    cv::imwrite( "../data/matches.png", imgMatches );
    cv::waitKey( 0 );

    // Filter match and remove too much distance
    // Rule here is to remove match which is greater than four times minimum distance
    vector< cv::DMatch > goodMatches;
    double minDis = 9999;
    for ( size_t i=0; i<matches.size(); i++ )
    {
        if ( matches[i].distance < minDis )
            minDis = matches[i].distance;
    }
    cout<<"min dis = "<<minDis<<endl;

    for ( size_t i=0; i<matches.size(); i++ )
    {
        if (matches[i].distance < 10*minDis)
            goodMatches.push_back( matches[i] );
    }

    // Show good matches
    cout<<"good matches="<<goodMatches.size()<<endl;
    cv::drawMatches( rgb1, kp1, rgb2, kp2, goodMatches, imgMatches );
    cv::imshow( "good matches", imgMatches );
    cv::imwrite( "./data/good_matches.png", imgMatches );
    cv::waitKey(0);

// Solving Pnp

    // Calculate the motion between images
    // Key function：cv::solvePnPRansac()
    // Prepare necessary parameter for  calling function
    
    // 3D point of first frame
    vector<cv::Point3f> pts_obj;
    // Image point of second frame
    vector< cv::Point2f > pts_img;

    // Camera internal reference
    CAMERA_INTRINSIC_PARAMETERS C;
    C.cx = 325.5;
    C.cy = 253.5;
    C.fx = 518.0;
    C.fy = 519.0;
    C.scale = 1000.0;
    cout<<"checkpoint 1"<<endl;

    for (size_t i=0; i<goodMatches.size(); i++)
    {
        // query is first , train is second
        cv::Point2f p = kp1[goodMatches[i].queryIdx].pt;
        // Be careful when getting d! x is right, y is down, so y is the row, x is the column! 
        ushort d = depth1.ptr<ushort>( int(p.y) )[ int(p.x) ];
        if (d == 0)
            continue;
        pts_img.push_back( cv::Point2f( kp2[goodMatches[i].trainIdx].pt ) );

        // Convert (u, v, d) to (x, y, z) 
        cv::Point3f pt ( p.x, p.y, d );
        cv::Point3f pd = point2dTo3d( pt, C );
        pts_obj.push_back( pd );
    }
    cout<<"checkpoint 2"<<endl;

    double camera_matrix_data[3][3] = {
        {C.fx, 0, C.cx},
        {0, C.fy, C.cy},
        {0, 0, 1}
    };

    // Build Camera Matrix
    cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
    cv::Mat rvec, tvec, inliers;
    // Slove solvePnPRansac
// RANSAC : The idea is to randomly take a part of the existing match and estimate its motion. Because the correct matching results must be similar, and the false matching results must be flying wild. Just take out the convergence result.
    cout<<"checkpoint 3"<<endl;

    // // solvePnPRansac(InputArray _opoints, InputArray _ipoints,
    //                     InputArray _cameraMatrix, InputArray _distCoeffs,
    //                     OutputArray _rvec, OutputArray _tvec, bool useExtrinsicGuess,
    //                     int iterationsCount, float reprojectionError, double confidence,
    //                     OutputArray _inliers, int flags)
    cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, 0.99, inliers );
    cout<<"checkpoint 4"<<endl;

    cout<<"inliers: "<<inliers.rows<<endl;
    cout<<"R="<<rvec<<endl;
    cout<<"t="<<tvec<<endl;

    // Draw inlier mactches
    vector< cv::DMatch > matchesShow;
    for (size_t i=0; i<inliers.rows; i++)
    {
        matchesShow.push_back( goodMatches[inliers.ptr<int>(i)[0]] );    
    }
    cv::drawMatches( rgb1, kp1, rgb2, kp2, matchesShow, imgMatches );
    cv::imshow( "inlier matches", imgMatches );
    cv::imwrite( "./data/inliers.png", imgMatches );
    cv::waitKey( 0 );

    return 0;
}