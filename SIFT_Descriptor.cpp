#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"


using namespace cv;

//Sorting Comparison Function - Added by Ritesh
bool myobject (DMatch i,DMatch j) 
{ 
	return (i.distance<j.distance); 
}


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

	float size = atof(argv[3]);
	
	resize(img_object_temp, img_object, Size(), size, size, INTER_AREA);
	

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
	
	std::sort(matches.begin(),matches.end(),myobject);
	
	int number_of_matches = atoi(argv[4]);
	
	if(number_of_matches > matches.size())
		{
			number_of_matches = matches.size();
			}
	
	for(int i = 0; i < atoi(argv[4]) ; i++)
	{
		good_matches.push_back(matches[i]);
	}

	Mat img_matches;

	drawMatches (img_object, keypoints_object, img_scene, keypoints_scene, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (std::vector<DMatch>::iterator it = good_matches.begin(); it != good_matches.end(); it++ )
	{
		obj.push_back(keypoints_object[ (*it).queryIdx ].pt);
		scene.push_back(keypoints_scene[ (*it).trainIdx ].pt);
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
	{ std::cout << " Usage: ./SIFT_descriptor <object_image> <scene_image> <resize parameter> <number of matches to be used (integer)> \n FOR FURTHER INFO CHECK README \n" << std::endl; }

