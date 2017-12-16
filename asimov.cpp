#include <iostream>
#include <string>

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/cudabgsegm.hpp"
#include "opencv2/cudalegacy.hpp"
#include "opencv2/video.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudacodec.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

//forward declarations
cv::cuda::GpuMat processImage(Mat &img);
void blur(Mat &img);
Rect findContours(Mat &img);

int main(int argc, const char** argv)
{
    VideoCapture cap;

    cout << "opening camera...\n";
    cap.open(0);

    if (!cap.isOpened())
    {
        cerr << "can not open camera or video file" << endl;
        return -1;
    }
    cout << "done.\n";

    cout << "creating/reading to frame...\n";
    Mat frame;
    cap >> frame;
    cout << "done.\n";

    cout << "creating GPU Mat frame...\n";
    GpuMat d_frame(frame);
    cout << "done.\n";

    cout << "creating MOG pointers...\n";
    Ptr<BackgroundSubtractor> mog2 = cuda::createBackgroundSubtractorMOG2();
    cout << "done.\n" << std::endl;

    cout << "creating GPU Mat objects...\n";
    GpuMat d_MOG2_fgmask;
    GpuMat d_MOG2_fgimg;
    GpuMat d_MOG2_bgimg;
    cout << "Done.\n";

    cout << "creating MOG2 (non GPU) Mat objects...\n";
    Mat MOG2_fgmask;
    Mat MOG2_fgimg;
    Mat MOG2_bgimg;
    cout << "done.\n";

    cout << "applying MOGs...\n";
    mog2->apply(d_frame, d_MOG2_fgmask);
    cout << "done.\n" << endl;

    cout << "creating window objects...\n";

    namedWindow( "Contours", WINDOW_AUTOSIZE );
    cout << "done.\n";
    for(;;)
    {
        cap >> frame;
        Mat backup_frame = frame;

        if (frame.empty())
            break;

        //process image to make further computations easier
        d_frame = processImage(frame);

        int64 start = cv::getTickCount();

        //update the model
        mog2->apply(d_frame, d_MOG2_fgmask);
        mog2->getBackgroundImage(d_MOG2_bgimg);
           
        double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
        std::cout << "FPS : " << fps << std::endl;

        d_MOG2_fgimg.create(d_frame.size(), d_frame.type());
        d_MOG2_fgimg.setTo(Scalar::all(0));
        d_frame.copyTo(d_MOG2_fgimg, d_MOG2_fgmask);

        d_MOG2_fgmask.download(MOG2_fgmask);
        d_MOG2_fgimg.download(MOG2_fgimg);

        d_frame.download(frame);

        Rect boundRect = findContours(MOG2_fgimg);
	    Scalar color = Scalar(0, 255, 0);
	    rectangle( backup_frame, boundRect.tl(), boundRect.br(), color, 2, 8, 0 );

	    imshow("Contours", backup_frame);

        int key = waitKey(30);
        if (key == 27)
            break;
    }

    return 0;
}

//convert image to grayscale and apply Gaussian blur
cv::cuda::GpuMat processImage(Mat &img)
{
	cv::cuda::GpuMat gpu_img(img);
	cv::cuda::cvtColor(gpu_img, gpu_img, CV_BGR2GRAY);

	gpu_img.download(img);
	blur(img);
	
	gpu_img.upload(img);
    return gpu_img;
}

void blur(cv::Mat &img)
{
	Size ksize(21, 21);
	int sigma1 = 0;
	//int sigma2 = 0;
	Ptr<cuda::Filter> gauss = cuda::createGaussianFilter(img.type(), -1, ksize, sigma1);

	cv::cuda::GpuMat dst;
	cv::cuda::GpuMat gpu_img(img);
    gauss->apply(gpu_img, dst);

    cv::Mat dst_gold;
    cv::GaussianBlur(img, img, ksize, sigma1);
}

Rect findContours(Mat &img)
{
	RNG rng(12345);
	int thresh = 50;
  	Mat threshold_output;
  	vector<vector<Point> > contours;
  	vector<Vec4i> hierarchy;

  	/// Detect edges using Threshold
  	threshold( img, threshold_output, thresh, 255, THRESH_BINARY );

  	// flood fill contours
  	//floodFill(threshold_output, Point(0, 0), Scalar(255));

  	namedWindow( "Edges", CV_WINDOW_AUTOSIZE );
  	imshow( "Edges", threshold_output );
  	/// Find contours
  	findContours( threshold_output, contours, hierarchy, RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

  	/// Approximate contours to polygons + get bounding rects and circles
 	vector<Point> contours_poly;
  	Rect boundRect;
  	Point2f center;

 	// find the largest contour
 	double max_area = 0;
 	int max_index = 0;
 	for ( int i = 0; i < contours.size(); i++)
 	{
 		double size = contourArea(contours[i]);
 		if (size > max_area)
 		{
 			max_area = size;
 			max_index = i;
 		}
 	}
 
    Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
 	if (max_index > 0)
 	{
	    boundRect = boundingRect( contours[max_index] );
	}
  	return boundRect;
}

