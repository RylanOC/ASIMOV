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


//forward declarations
cv::cuda::GpuMat processImage(cv::Mat &img);
void blur(cv::Mat &img);
cv::Rect findContours(cv::Mat &img);

int main(int argc, const char** argv)
{
    cv::VideoCapture cap;

    cap.open(0);

    if (!cap.isOpened())
    {
        std::cerr << "can not open camera or video file" << std::endl;
        return -1;
    }
    cv::Mat frame;
    cap >> frame;

    cv::cuda::GpuMat d_frame(frame);

    cv::Ptr<cv::BackgroundSubtractor> mog2 = cv::cuda::createBackgroundSubtractorMOG2();

    cv::cuda::GpuMat d_MOG2_fgmask;
    cv::cuda::GpuMat d_MOG2_fgimg;
    cv::cuda::GpuMat d_MOG2_bgimg;

    cv::Mat MOG2_fgmask;
    cv::Mat MOG2_fgimg;
    cv::Mat MOG2_bgimg;

    mog2->apply(d_frame, d_MOG2_fgmask);

    cv::namedWindow( "Contours", cv::WINDOW_AUTOSIZE );
    cv::namedWindow( "Edges", cv::WINDOW_AUTOSIZE );
    for(;;)
    {
        cap >> frame;
        cv::Mat backup_frame = frame;

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
        d_MOG2_fgimg.setTo(cv::Scalar::all(0));
        d_frame.copyTo(d_MOG2_fgimg, d_MOG2_fgmask);

        d_MOG2_fgmask.download(MOG2_fgmask);
        d_MOG2_fgimg.download(MOG2_fgimg);

        d_frame.download(frame);

        cv::Rect boundRect = findContours(MOG2_fgimg);
	    cv::Scalar color = cv::Scalar(0, 255, 0);
	    cv::rectangle( backup_frame, boundRect.tl(), boundRect.br(), color, 2, 8, 0 );

	    cv::imshow("Contours", backup_frame);

        int key = cv::waitKey(30);
        if (key == 27)
            break;
    }

    return 0;
}

//convert image to grayscale and apply Gaussian blur
cv::cuda::GpuMat processImage(cv::Mat &img)
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
	cv::Size ksize(21, 21);
	int sigma1 = 0;
	//int sigma2 = 0;
	cv::Ptr<cv::cuda::Filter> gauss = cv::cuda::createGaussianFilter(img.type(), -1, ksize, sigma1);

	cv::cuda::GpuMat dst;
	cv::cuda::GpuMat gpu_img(img);
    gauss->apply(gpu_img, dst);

    cv::Mat dst_gold;
    cv::GaussianBlur(img, img, ksize, sigma1);
}

cv::Rect findContours(cv::Mat &img)
{
	cv::RNG rng(12345);
	int thresh = 50;
  	cv::Mat threshold_output;
  	std::vector<std::vector<cv::Point> > contours;
  	std::vector<cv::Vec4i> hierarchy;

  	/// Detect edges using Threshold
  	cv::threshold( img, threshold_output, thresh, 255, cv::THRESH_BINARY );

  	// flood fill contours
  	//floodFill(threshold_output, Point(0, 0), Scalar(255));

  	cv::imshow( "Edges", threshold_output );
  	/// Find contours
  	cv::findContours( threshold_output, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

  	/// Approximate contours to polygons + get bounding rects and circles
 	std::vector<cv::Point> contours_poly;
  	cv::Rect boundRect;
  	cv::Point2f center;

 	// find the largest contour
 	double max_area = 0;
 	int max_index = 0;
 	for ( int i = 0; i < contours.size(); i++)
 	{
 		double size = cv::contourArea(contours[i]);
 		if (size > max_area)
 		{
 			max_area = size;
 			max_index = i;
 		}
 	}
 
    cv::Mat drawing = cv::Mat::zeros( threshold_output.size(), CV_8UC3 );
 	if (max_index > 0)
 	{
	    boundRect = cv::boundingRect( contours[max_index] );
	}
  	return boundRect;
}

