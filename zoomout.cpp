#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

/**
 * @function main
 */
Mat zoomout( Mat src, int iterations )
{



  
  Mat dst = src;

  
  for( int i = 0; i<iterations; i++)
  {	
	  pyrDown( src, dst, Size( src.cols/2, src.rows/2 ) );
      src = dst;
  }


   

  return dst;
}

