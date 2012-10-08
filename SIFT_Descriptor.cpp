#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#define ASSERT(condition, message) \
        if (! (condition)) { \
            std::cerr << "\nAssertion `" #condition "` failed "	\
			<< "\nLine : " << __LINE__ << "\n" << message << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \


using namespace cv;

void readme();												//README function for showing usage


int main (int argc, char** argv)

{
	if(argc != 5)
	{
		readme();										//Arguments less than five triggers README function
		return -1;
	}

	Mat img_object;
	Mat img_object_temp = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE);				//Loading Object, scene images
	Mat img_scene  = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE);

	if(!img_object.data && !img_scene.data)
	{
		std::cout<< " ---(!) ERROR READING IMAGES " <<std::endl;				//Faulty images or no files of the names detected
		return -1;
	}

	for (int i =0; i< atoi(argv[3]); i++)
	{
		pyrDown(img_object_temp, img_object, Size( img_object_temp.cols/2, img_object_temp.rows/2));  // Scaling down the object image by a factor of 2^[argv[3]]

		img_object_temp = img_object;
	}

	std::vector<KeyPoint> keypoints_object, keypoints_scene;					//Initializing data structures for keypoints & descriptors
	Mat descriptors_object, descriptors_scene;

	SiftFeatureDetector detector; 						
	SiftDescriptorExtractor extractor;								//Initializing SIFT functions 


	detector.detect(img_object, keypoints_object);							//Using the SIFT detector for keypoints								
	detector.detect(img_scene, keypoints_scene);

	extractor.compute(img_object, keypoints_object, descriptors_object);				//Using SIFT descriptor extractor
	extractor.compute(img_scene, keypoints_scene, descriptors_scene);	

	FlannBasedMatcher matcher;				
	std::vector <DMatch> matches;									//Initializing matcher and vector for matches extracted
	matcher.match( descriptors_object, descriptors_scene, matches);

	double max_dist = 0;
	double min_dist = 100;										//Initializing distance measuremenets for matches

	for (int i = 0; i <descriptors_object.rows; i++)
	{
		double dist = matches[i].distance;	
		if( dist < min_dist) min_dist = dist;							// finding max and min distance values
		if( dist > max_dist) max_dist = dist;
	}

	printf ("-- Max Distance : %f \n", max_dist);
	printf ("-- Min Distance : %f \n", min_dist);

	std::vector<DMatch> good_matches;

	for(int i = 0; i < descriptors_object.rows; i++)
	{
		if (matches[i].distance < atoi(argv[4])*min_dist)					//Classifying good matches from generic matches by defining a threshold
		{
			good_matches.push_back(matches[i]);
		}
	}

	ASSERT(good_matches.size() >= 4 , "Not enough good matches were detected!");	// Checking whether good matches were found

	std::cout<<good_matches.size()<<std::endl;
	Mat img_matches;


	drawMatches (img_object, keypoints_object, img_scene, keypoints_scene, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (int i = 0; i <good_matches.size(); i++)
	{
		obj.push_back(keypoints_object[ good_matches[i].queryIdx ].pt);
		scene.push_back(keypoints_scene[ good_matches[i].trainIdx ].pt);
	}


	   Mat H = findHomography( obj, scene, CV_RANSAC );

  //-- Get the corners from the image_1 ( the object to be "detected" )
  std::vector<Point2f> obj_corners(4);
  obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
  obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
  std::vector<Point2f> scene_corners(4);

  perspectiveTransform( obj_corners, scene_corners, H);

  //-- Draw lines between the corners (the mapped object in the scene - image_2 )
  line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
  line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );

  //-- Show detected matches
  imshow( "Good Matches & Object detection", img_matches );

  waitKey(0);
  return 0;
	}

	/**
	 * @function readme
	 */
void readme()
	{ std::cout << " Usage: ./SIFT_descriptor <img1> <img2> <resize parameter in power of two> <threshold multiplier for min distance> \n FOR FURTHER INFO CHECK README \n" << std::endl; }
