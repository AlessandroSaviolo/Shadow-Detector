#ifndef UTILITY_H
#define UTILITY_H

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;


class Utility
{
public:
	String datasetSelection();

	vector<Mat> loadImages(String& path);

	void resampleImage(Mat& image);
};

#endif