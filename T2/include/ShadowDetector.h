#ifndef SHADOWDETECTOR_H
#define SHADOWDETECTOR_H

#include "Utility.h"

using namespace cv;
using namespace std;


class ShadowDetector
{
public:
	Mat applyFilters(Mat& image);

	Mat computeSamples(Mat& filteredImage);

	Mat runKmeans(Mat& filteredImage, Mat& samples, const int& numSegments, vector<int>& labels, int& attempts, Mat& centers);

	Mat intensityThresholding(Mat& segmentedImage);

	vector<Mat> computeMasks(const int& numSegments, Mat& segmentedImage, vector<int>& labels);

	vector<int> computeCandidates(Mat& filteredImage, Mat& thresholdedImage, vector<int>& labels);

	Mat computeLBP(Mat image);

	Mat computeMergeTable(vector<int>& candidates, vector<Mat>& masks, Mat& image, Mat& LBPimage);

	vector<Mat> applyMerge(vector<int>& candidates, Mat& mergeTable, vector<Mat>& masks);

	vector<Mat> compareCandidates(vector<int>& candidates, vector<Mat>& masks, Mat image, Mat& LBPimage, vector<Mat>& candidateMasks);
};

#endif