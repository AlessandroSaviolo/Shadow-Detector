#include "ShadowDetector.h"

// ----------------------------------------- PRE-PROCESSING FILTERING TO IMPROVE SEGMENTATION -----------------------------------------

Mat ShadowDetector::applyFilters(Mat& image)
{
/*
	Bilateral filter and Closure operation are applied to the image in order to remove (partially) noise and small holes.
	This step is not mandatory but it helps to obtain more precise segmentation of the image.
*/

	// Bilateral filter parameters
	int d = 9;
	double sigmaColor = 40;
	double sigmaSpace = 80;

	Mat filteredImage(image.size(), image.type());

	// Run Bilateral filter
	bilateralFilter(image, filteredImage, d, sigmaColor, sigmaSpace);

	// Build the kernel for the Closure operation
	int size = 4;
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(size, size));

	// Apply Closure (Dilate and Erode filters) numClosures times
	int numClosures = 2;
	for (int x = 0; x < numClosures; x++)
	{
		dilate(filteredImage, filteredImage, kernel);
		erode(filteredImage, filteredImage, kernel);
	}

	return filteredImage;
};


// -------------------------------------------------------------- SEGMENTATION -------------------------------------------------------------

Mat ShadowDetector::computeSamples(Mat& filteredImage)
{
/*
	Consider the image in the HSV color space and segment it by using kmeans. Samples given in input to kmeans will contain both the color information (H, S, V channels) and the position.
*/

	cvtColor(filteredImage, filteredImage, CV_BGR2HSV);

	// Compute samples (5 cols: 3 for color information, 2 for coordinates)
	Mat samples(filteredImage.rows * filteredImage.cols, 5, CV_32F);

	// All the samples are normalized
	for (int x = 0; x < filteredImage.rows; x++)
		for (int y = 0; y < filteredImage.cols; y++)
		{
			samples.at<float>(y + x * filteredImage.cols, 0) = (float)filteredImage.at<Vec3b>(x, y)[0] / 255;
			samples.at<float>(y + x * filteredImage.cols, 1) = (float)filteredImage.at<Vec3b>(x, y)[1] / 255;
			samples.at<float>(y + x * filteredImage.cols, 2) = (float)filteredImage.at<Vec3b>(x, y)[2] / 255;
			samples.at<float>(y + x * filteredImage.cols, 3) = ((float)x / (float)filteredImage.rows);
			samples.at<float>(y + x * filteredImage.cols, 4) = ((float)y / (float)filteredImage.cols);
		}

	return samples;
};

Mat ShadowDetector::runKmeans(Mat& filteredImage, Mat& samples, const int& numSegments, vector<int>& labels, int& attempts, Mat& centers)
{
/*
	Parameters for kmeans:
		numSegments: Number of segments to split the set by
		labels: Output integer array that stores the segment indices for every sample
		attempts: Flag to specify the number of times the algorithm is executed using different initial labellings. The algorithm returns the labels that yield the best compactness
		centers: Output matrix of the segment centers, one row per each segment center
*/

	// Run Kmeans
	kmeans(samples, numSegments, labels, TermCriteria(CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);
	
	// De-Normalize values of found centers
	for (int x = 0; x < centers.rows; x++)
	{
		centers.at<float>(x, 0) = centers.at<float>(x, 0) * 255;
		centers.at<float>(x, 1) = centers.at<float>(x, 1) * 255;
		centers.at<float>(x, 2) = centers.at<float>(x, 2) * 255;
	}

	Mat segmentedImage(filteredImage.size(), filteredImage.type());

	for (int x = 0; x < segmentedImage.rows; x++)
		for (int y = 0; y < segmentedImage.cols; y++)
		{
			// Find segment label of (x, y) at (y + x * filteredImage.cols)
			int segmentLabel = labels[y + x * filteredImage.cols];

			// Assign the value of the center of that segment (identified by the segment label) to (x, y) of segmentedImage
			segmentedImage.at<Vec3b>(x, y)[0] = centers.at<float>(segmentLabel, 0);
			segmentedImage.at<Vec3b>(x, y)[1] = centers.at<float>(segmentLabel, 1);
			segmentedImage.at<Vec3b>(x, y)[2] = centers.at<float>(segmentLabel, 2);
		}

	cvtColor(segmentedImage, segmentedImage, CV_HSV2BGR);

	return segmentedImage;
};



// ---------------------------------------------------------- INTENSITY THRESHOLDING ----------------------------------------------------------

Mat ShadowDetector::intensityThresholding(Mat& segmentedImage)
{
/*
	Since shadows usually have a lower intensity then the major part of the image, it's rather effective to threshold the intensity of the resulting segmentation.
	Thresholding can be done both for the lower intensities and the upper intensities.
	The lower threshold allows to erase many false positives (such as some black objects) since shadows cannot be too dark.
	The upper threshold allows to consider far less segments in the detection since shadows usually are contained in the segments with the lowest intensities.
	If the two thresholds are chosen too strict, they give birth to many false negatives. On the other hand, they cannot even be chosen too soft, since they would give birth to many false positives.

	The approach used consists of applying a Simple Threshold to the intensity channel:
		Compute the binary image by using the method inRange to apply both the lower and the upper threshold. Thresholds are fixed but work greatly in most of the cases.
		There are two cases: the first is the more general and used, the second is applied when the image is strongly illuminated
		(the upper bound on the intensity must be increased since a strong illumination on the image affects both the objects and the shadows)

	Steps:
		1. Convert the image to HSV color space
		2. Split the HSV image and consider only it's third channel, V (which represents intensity)
		3. Apply Simple Threshold to the intensity channel (both lower and upper)
*/

	// 1. Convert the image to HSV color space
	Mat HSVimage;
	cvtColor(segmentedImage, HSVimage, CV_BGR2HSV);

	// 2. Split the HSV image and consider only it's third channel, V (which represents intensity)
	vector<Mat> channels;
	split(HSVimage, channels);
	Mat intensityImage = channels[2];

	// 3. Apply Simple Threshold to the intensity channel (both lower and upper)
	double min, max;
	minMaxLoc(intensityImage, &min, &max);

	int upperThreshold = 100;
	int lowerThreshold = 5;

	if (min > 50)
		upperThreshold = 150;

	Mat thresholdedImage(intensityImage.size(), CV_8UC1);
	inRange(intensityImage, Scalar(lowerThreshold), Scalar(upperThreshold), thresholdedImage);

	return thresholdedImage;
};



// ---------------------------------------------------------------- MASKS COMPUTATION ----------------------------------------------------------------

vector<Mat> ShadowDetector::computeMasks(const int& numSegments, Mat& segmentedImage, vector<int>& labels)
{
/*
	Compute a mask for each segment.

	Steps:
		1. Create the vector which will store all the masks and initialize all the masks
		2. Color in white the pixels of the segment for its corresponding mask
*/

	// 1. Create the vector which will store all the masks
	vector<Mat> masks;

	// Initialize all the masks
	for (int i = 0; i < numSegments; i++)
	{
		// Binary mask
		Mat mask(segmentedImage.size(), CV_8UC1);
		threshold(mask, mask, 255, 255, THRESH_BINARY);

		// Push it inside the vector
		masks.push_back(mask);
	}

	// 2. Color in white the pixels of the segment for its corresponding mask
	for (int x = 0; x < segmentedImage.rows; x++)
		for (int y = 0; y < segmentedImage.cols; y++)
		{
			// Find segment label at (y + x * segmentedImage.cols)
			int segmentLabel = labels[y + x * segmentedImage.cols];

			// Consider mask with index segmentLabel
			Mat mask = masks[segmentLabel];

			// Set white the point found
			mask.at<uchar>(x, y) = 255;
		}

	return masks;
};



// -------------------------------------------------------------- CANDIDATES COMPUTING ---------------------------------------------------------------

vector<int> ShadowDetector::computeCandidates(Mat& filteredImage, Mat& thresholdedImage, vector<int>& labels)
{
/*
	Find those segments that are contained in the white components of thresholdedImage and consider them as shadow candidates.
	The method returns a vector which containes the indexes of all the candidate segments.

	Steps:
		1. Look for every pixel in the thresholded image
		2. If it's a match (white), consider its label and if its label is not in the vector of candidates, add it (otherwise it's already been selected as a candidate)
*/

	// Store the index of each candidate shadow segment (there are at most numSegments)
	vector<int> candidates;

	// 1. Look for every pixel in the thresholded image
	for (int x = 0; x < thresholdedImage.rows; x++)
		for (int y = 0; y < thresholdedImage.cols; y++)
		{
			// 2. If it's a match (white)
			if (thresholdedImage.at<uchar>(x, y) == 255)
			{
				// Consider its label
				int segmentLabel = labels[y + x * filteredImage.cols];
			
				// If its label is not in the vector of candidates, add it (otherwise it's already been selected as a candidate)
				if (find(candidates.begin(), candidates.end(), segmentLabel) == candidates.end())
					candidates.push_back(segmentLabel);
			}
		}

	return candidates;
};



// ------------------------------------------------------ COMPUTE LOCAL BINARY PATTERN IMAGE ----------------------------------------------------------

Mat ShadowDetector::computeLBP(Mat image)
{
/*
	Since Local Binary Patterns are invariant to both gray scale and rotation, they can be used to capture the texture information contained in each segment.
	Consider each pixel of the image (converted to grayscale) and center a 3x3 window on it. Compute the LBP value of the considered pixel using the other pixels in the window.
	For each LBP value computed, assign it to the corresponding pixel (same coordinates as the pixel in the center of the window) in the LBPimage returned by the algorithm.

	Steps:
		1. Convert image to Gray color space (LBP works on grayscale images)
		2. Compare the central pixel value with the neighbouring pixel values
		3. Assign the computed value to the corresponding pixel
*/

	// Store LBP image
	Mat LBPimage(image.size(), CV_8UC1);

	// 1. Convert image to Gray color space (LBP works on grayscale images)
	Mat grayImage;
	cvtColor(image, grayImage, CV_BGR2GRAY);

	int center, LBPvalue;

	// Consider anti-clockwise direction
	for (int x = 1; x < grayImage.rows - 1; x++)
		for (int y = 1; y < grayImage.cols - 1; y++)
		{
			// Store the value of the current pixel (center of the 3x3 window)
			center = grayImage.at<uchar>(x, y);

			// Decimal number representing the value of the center pixel in the LBPimage returned by the algorithm
			LBPvalue = 0;

			// 2. Compare the central pixel value with the neighbouring pixel values: if the value of the central pixel is greater or equal to the value
			// of the considered pixel in the window, add it (with the corresponding value expressed in decimal form) to the final pixel value
			if (center <= grayImage.at<uchar>(x - 1, y))
				LBPvalue += 1;

			if (center <= grayImage.at<uchar>(x - 1, y - 1))
				LBPvalue += 2;

			if (center <= grayImage.at<uchar>(x, y - 1))
				LBPvalue += 4;

			if (center <= grayImage.at<uchar>(x + 1, y - 1))
				LBPvalue += 8;

			if (center <= grayImage.at<uchar>(x + 1, y))
				LBPvalue += 16;

			if (center <= grayImage.at<uchar>(x + 1, y + 1))
				LBPvalue += 32;

			if (center <= grayImage.at<uchar>(x, y + 1))
				LBPvalue += 64;

			if (center <= grayImage.at<uchar>(x - 1, y + 1))
				LBPvalue += 128;

			// 3. Assign the computed value to the corresponding pixel
			LBPimage.at<uchar>(x, y) = LBPvalue;
		}

	return LBPimage;
};



// ---------------------------------------------------------------- NEIGHBORS COMPUTING ---------------------------------------------------------------

vector<int> computeNeighbors(vector<Mat>& masks, Mat& candidateMask)
{
/*
	Given a candidate, compute its neighbors (closest segments) and return their index.

	Steps:
		1. Apply Dilate filter to the candidate's mask
		2. For each possible neighbor of the candidate apply Dilate filter
		3. Apply AND operator to extract intersections
		4. Check if there is any intersection, if so, consider the non-candidate as neighbor of the candidate
*/

	// Store the neighbors of the candidate
	vector<int> neighbors;

	// Build the kernel for the dilation
	int size = 8;
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(size, size));

	// 1. Apply Dilate filter to the candidate's mask
	Mat candidateMaskDilated(candidateMask.size(), candidateMask.type());
	dilate(candidateMask, candidateMaskDilated, kernel);

	// Store the index of the all possible neighbors of the candidate (all masks for now)
	vector<int> segments;
	for (int i = 0; i < masks.size(); i++)
	{
		segments.push_back(i);
	}

	// 2. For each possible neighbor of the candidate
	for (int j = 0; j < segments.size(); j++)
	{
		// Find its mask
		Mat segmentMask = masks[segments[j]];
		
		// Apply Dilate filter
		Mat segmentMaskDilated(segmentMask.size(), segmentMask.type());
		dilate(segmentMask, segmentMaskDilated, kernel);

		// 3. Apply AND operator to extract intersections
		Mat intersection(candidateMask.size(), candidateMask.type());
		bitwise_and(candidateMaskDilated, segmentMaskDilated, intersection);

		// 4. Check if there is any intersection, if so, consider it as a neighbor and add its index to the vector
		if (countNonZero(intersection) > 0)
		{
			neighbors.push_back(segments[j]);
		}
	}

	return neighbors;
};



// ------------------------------------------------------------------- MASKS EROSION -------------------------------------------------------------------

vector<Mat> erodeMasks(vector<Mat> masks)
{
/*
	Apply Erode filter to each mask passed to the method and return the new set of masks.
*/

	vector<Mat> erodedMasks;

	for (int x = 0; x < masks.size(); x++)
	{
		Mat erodedMask(masks[0].size(), masks[0].type());
		erode(masks[x], erodedMask, getStructuringElement(MORPH_ELLIPSE, Size(4, 4)));
		erodedMasks.push_back(erodedMask);
	}

	return erodedMasks;
};



// ------------------------------------------------------------- MERGE 'SIMILAR' CANDIDATES ------------------------------------------------------------

Mat ShadowDetector::computeMergeTable(vector<int>& candidates, vector<Mat>& masks, Mat& image, Mat& LBPimage)
{
/*
	For every candidate, compute its neighbors and check if any of them is enough "similar" to the candidate to be merged with.
	The similarity is computed by using both color (absolute difference of the mean of each channel) and texture (LBP histogram comparison) information.

	The method returns a matrix (mergeTable) which sizes are defined by the number of segments (an additional column is added for later computation - see applyMerge method),
	so that each row/column identifies a unique segment. The matrix is initialized with all 0's (symbolizing that initially no merges have to be computed).
	Everytime a candidate finds a neighbor that need to be merged with, then the value of the corresponding point in the matrix (defined by the candidate index as row and the neighbor index as column) is changed to 100 (merging flag).

	Steps for every candidate:
		1. Compute the LBP histogram and the BGR mean of the selected candidate
		2. Find the neighbors of the selected candidate
		3. Compute the LBP histogram and the BGR mean of the selected neighbor
		4. Compare LBP histograms by using BHATTACHARYYA distance and then threshold the resulting ratio
		5. Compare BGR means by taking the sum of all the absolute differences for each channel (B, G and R) and then threshold the resulting ratio
		6. If the computed ratios are below both thresholds, then add the considered neighbor to the list of segments to be merged with the considered candidate
*/

	// Store the list of segments to be merged. Each point of the matrix will be set to 100 if the relative segment (the row/column represent the index of the segment) will have to be merged
	int numSegments = masks.size();
	Mat mergeTable = Mat::zeros(numSegments, numSegments + 1, CV_8UC1);

/*
	Since for the comparison of the segments we are interested in studying the properties of all the pixels belonging to the main connected component(s), it's rather
	effective to reject the spurious pixels which affect negatively (since don't have "anything" in common with the main connected component of the segment) the results
	of the comparison.
*/
	vector<Mat> erodedMasks = erodeMasks(masks);

	// Variables for computing the LBP histogram
	int LBPbins = 256;
	float LBPrange[] = { 0, 256 };
	const float* LBPHistRange = { LBPrange };
	Mat candidateLBPhistogram, neighborLBPhistogram;

	// Store the neighbors of each candidate
	vector<int> neighbors;

	// For every candidate
	for (int i = 0; i < candidates.size(); i++)
	{
		int candidateIndex = candidates[i];

		// 1. Compute the LBP histogram of the selected candidate
		calcHist(&LBPimage, 1, 0, erodedMasks[candidateIndex], candidateLBPhistogram, 1, &LBPbins, &LBPHistRange);

		// Compute the BGR mean of the selected candidate
		Scalar candidateBGRmean = mean(image, erodedMasks[candidateIndex]);

		// 2. Find the neighbors of the selected candidate
		neighbors = computeNeighbors(erodedMasks, erodedMasks[candidateIndex]);

		// For every neighbor of the candidate consider only those that are also candidates
		for (int j = 0; j < neighbors.size(); j++)
			if (find(candidates.begin(), candidates.end(), neighbors[j]) != candidates.end())
			{
				int neighborIndex = neighbors[j];

				// 3. Compute the LBP histogram of the selected neighbor
				calcHist(&LBPimage, 1, 0, erodedMasks[neighborIndex], neighborLBPhistogram, 1, &LBPbins, &LBPHistRange);

				// Compute the BGR mean of the selected neighbor
				Scalar neighborBGRmean = mean(image, erodedMasks[neighborIndex]);

				// 4. Compare LBP histograms by using BHATTACHARYYA distance and then threshold the resulting ratio
				// LBPratio takes value from 0 to 1, the less the result the better the match
				double LBPratio = compareHist(candidateLBPhistogram, neighborLBPhistogram, CV_COMP_BHATTACHARYYA);

				// 5. Compare BGR means by taking the sum of all the absolute differences for each channel (B, G and R) and then threshold the resulting ratio
				double BGRratio = abs(candidateBGRmean[0] - neighborBGRmean[0]) + abs(candidateBGRmean[1] - neighborBGRmean[1]) + abs(candidateBGRmean[2] - neighborBGRmean[2]);

				// Thresholds
				double LBPthreshold = 0.15;
				double BGRthreshold = 70;

				// 6. If the computed ratios are below both thresholds, then add the considered neighbor to the list of segments to be merged with the considered candidate
				// Set to 100 the point which row and column are identified resp. by the candidate and the neighbor indexes
				// The number 100 is just a flag to simbolize that at that position the segment identified by the column has to be merged with the segment identified by the row
				if (BGRratio <= BGRthreshold && LBPratio <= LBPthreshold)
					mergeTable.at<uchar>(candidateIndex, neighborIndex) = 100;
			}
	}

	//cout << mergeTable << endl;
	//waitKey(0);

	return mergeTable;
};



vector<Mat> ShadowDetector::applyMerge(vector<int>& candidates, Mat& mergeTable, vector<Mat>& masks)
{
	/*
		Given the merge table (which contains for each candidate the segments that have to be merged with), for each candidate (which index corresponds to the row number)
		create a merge list (where to store the indexes of the segments to be merged) and fill it with all the segments chosen for the merge.
		Moreover, for each segment added to the merge list (except the candidate considered) add also those segments that have been chosen for the merge with it
		(and so on until no more segments are added to the list). This is the crucial point and the source of complexity for the method.

		Each row/column of the merge table corresponds to the index of a segment. A column has been added to the matrix in order to keep track (in this method) of the seen/unseen rows.
		In other words, everytime a row (candidate) is visited, the value of the point corresponding to that row and to the last column in the matrix is changed to 200 (seen flag).
		This allows the method to don't consider a row more than once.

		By doing so, after filling up successfully the merge list and after checking the corresponding segments as seen, we are able to merge all the segments that have been found "similar" from each other,
		without making any repetition (in the next loops those segments won't be considered anymore because they have already been seen).

		The merging is done by taking the masks of all the segments in the merge list and doing an OR operation, storing the results in a binary image (initialized to black).

		Steps:
			1. Consider each unseen candidate
			2. Check if any neighbor of the candidate has been selected for the merging
			3. Consider all the segments which index is stored in the mergeList (except the considered candidate) and check if they have, in turn, to be merged with other segments
			4. Check if there are other segments to be merged, if so add them to the merge list and update the counter
			5. Create the binary image that will contain the mask and merge all the masks of the segments in the mergeList
	*/

	// Store the set of masks of the (merged) candidates
	vector<Mat> candidateMasks;

	// 1. Consider each unseen candidate
	for (int i = 0; i < candidates.size(); i++)
		if (mergeTable.at<uchar>(candidates[i], mergeTable.cols - 1) != 200)
		{
			// Create the merge list where to place all the indexes of the segments that have to be merged
			vector<int> mergeList;

			// Add the candidate considered in this loop (since even if it has no neighbors to be merged with, we still want to keep it in the set of masks)
			mergeList.push_back(candidates[i]);

			// Use this counter to keep track of the number of other rows that must be considered during this loop
			int count = 0;

			// 2. Check if any neighbor of the candidate has been selected for the merging
			for (int col = 0; col < mergeTable.cols - 1; col++)
				if (mergeTable.at<uchar>(candidates[i], col) == 100)
				{
					// If a neighbor is found, add its index to the merge list
					mergeList.push_back(col);

					// Keep track of how many segments have to be merged
					count++;
				}

			// Check the row as seen (seen flag)
			mergeTable.at<uchar>(candidates[i], mergeTable.cols - 1) = 200;

		/*
			cout << mergeTable << endl;
			waitKey(0);
		*/

			// 3. Consider all the segments which index is stored in the mergeList (except the considered candidate) and check if they have, in turn, to be merged with other segments
			for (int t = 1; t <= count; t++)
			{
				// Consider the element [t] of the merge list, which is the index of the segment to be merged with candidate[i]
				int rowIndex = mergeList[t];

				// 4. Check if there are other segments to be merged, if so add them to the merge list and update the counter
				for (int col = 0; col < mergeTable.cols - 1; col++)
					if (mergeTable.at<uchar>(rowIndex, col) == 100 && find(mergeList.begin(), mergeList.end(), col) == mergeList.end())
					{
						// Add the found segment index to the merge list
						mergeList.push_back(col);

						// Update the counter because another segment has been found for the merging
						count++;
					}

				// Check the row as seen (seen flag)
				mergeTable.at<uchar>(rowIndex, mergeTable.cols - 1) = 200;
			}

		/*
			cout << mergeTable << endl;
			waitKey(0);
		*/

			// 5. Create the image that will contain the mask
			Mat newMask = Mat::zeros(masks[0].size(), CV_8UC1);

			// Merge all the masks of the segments in the mergeList
			for (int x = 0; x < mergeList.size(); x++)
			{
				Mat tmpMask = Mat::zeros(masks[0].size(), CV_8UC1);

				bitwise_or(masks[mergeList[x]], newMask, tmpMask);

				newMask = tmpMask;
			}

			candidateMasks.push_back(newMask);
		}

	return candidateMasks;
};



// ------------------------------------------------------------- EDGE IMAGE COMPUTATION ------------------------------------------------------------------

Mat computeEdgeImage(Mat image)
{
	Mat edgeImage;

	// Remove (some) noise
	GaussianBlur(image, image, Size(7, 7), 0);

	// Run Canny to detect edges
	Canny(image, edgeImage, 15, 45);

	return edgeImage;
};



// -------------------------------------------------------------- EDGE RATIO COMPUTATION ------------------------------------------------------------------

double computeEdgeRatio(Mat& mask, Mat& edgeImage)
{
/*
	Since segments under shadow retain most of their texture, it's possible to assume that the number of edges in a shadow segment cannot be higher than
	the number of edges in the similar neighbor segment.
	Here, a Edge Ratio is used because different segments have different number of pixels.
*/

	Mat tmp = Mat::zeros(mask.size(), CV_8UC1);
	edgeImage.copyTo(tmp, mask);

	return double(countNonZero(tmp)) / double(countNonZero(mask));
};



// --------------------------------------------------------------- CANDIDATES COMPARISON ------------------------------------------------------------------

/*
	Compare each candidate with its non-candidate neighbors. The comparison is based on texture, chromaticity and edge similarity.
	If a candidate has similar properties with one of its non-candidate neighbors (at least one), then consider it as a shadow segment.
	
	Steps:
		1. For each candidate compute: neighbors, Edge Ratio, Mean of each HSV channel, LBP histogram
		2. For each non-candidate neighbor of the candidate compute: Edge Ratio, Mean of each HSV channel, LBP histogram
		3. Compute candidate and the neighbor Edge Ratio difference
		4. Compute candidate and the neighbor's Hue absolute difference
		5. Compute candidate and the neighbor's Saturation absolute difference
		6. Compare LBP histograms using BHATTACHARYYA
		7. Threshold the comparisons and if the candidate has at least one "similar" neighbor than add it to the vector of shadows
*/

vector<Mat> ShadowDetector::compareCandidates(vector<int>& candidates, vector<Mat>& masks, Mat image, Mat& LBPimage, vector<Mat>& candidateMasks)
{
	// Store the index of each shadow segment
	vector<Mat> shadows;

	// Store neighbors
	vector<int> neighbors;

	// Compute the edge image
	Mat edgeImage = computeEdgeImage(image);

/*
	Since for the comparison of the segments we are interested in studying the properties of all the pixels belonging to the main connected component(s), it's rather
	effective to reject the spurious pixels which affect negatively (since don't have "anything" in common with the main connected component of the segment) the results
	of the comparison.
*/
	vector<Mat> erodedMasks = erodeMasks(masks);
	vector<Mat> erodedCandidateMasks = erodeMasks(candidateMasks);

	// Convert image to HSV since for the comparison only chromaticity is analyzed
	Mat HSVimage;
	cvtColor(image, HSVimage, CV_BGR2HSV);

	// Feature variables
	double candidateEdgeRatio, neighborEdgeRatio;
	Scalar candidateHSVmean, neighborHSVmean;
	Mat candidateLBPhistogram, neighborLBPhistogram;

	// LBP histogram variables
	int LBPbins = 256;
	float LBPrange[] = { 0, 256 };
	const float* LBPHistRange = { LBPrange };

	// Comparison variables
	double diffEdgeRatio, diffHue, diffSat, LBPratio;

	// 1. For each candidate
	for (int i = 0; i < erodedCandidateMasks.size(); i++)
	{
		Mat candidate = erodedCandidateMasks[i];

		// Compute the neighbors of the candidate
		// Use the non-eroded mask to compute the neighbors
		neighbors = computeNeighbors(erodedMasks, candidateMasks[i]);

		// Compute the Edge Ratio fo the candidate
		candidateEdgeRatio = computeEdgeRatio(candidate, edgeImage);

		// Compute the mean of each HSV channel of the candidate
		candidateHSVmean = mean(HSVimage, candidate);

		// Compute the LBP histogram of the candidate
		calcHist(&LBPimage, 1, 0, candidate, candidateLBPhistogram, 1, &LBPbins, &LBPHistRange);

		// 2. For each non-candidate neighbor of the candidate
		for (int j = 0; j < neighbors.size(); j++)
		{
			if (find(candidates.begin(), candidates.end(), neighbors[j]) == candidates.end())
			{
				int neighbor = neighbors[j];

				// Compute the Edge Ratio fo the neighbor
				neighborEdgeRatio = computeEdgeRatio(erodedMasks[neighbor], edgeImage);

				// Compute the mean of each HSV channel of the neighbor
				neighborHSVmean = mean(HSVimage, erodedMasks[neighbor]);

				// Compute the LBP histogram of the neighbor
				calcHist(&LBPimage, 1, 0, erodedMasks[neighbor], neighborLBPhistogram, 1, &LBPbins, &LBPHistRange);

				// 3. Compute candidate and the neighbor Edge Ratio difference
				diffEdgeRatio = neighborEdgeRatio - candidateEdgeRatio;

				// 4. Compute candidate and the neighbor's Hue absolute difference
				diffHue = abs(candidateHSVmean[0] - neighborHSVmean[0]);

				// 5. Compute candidate and the neighbor's Saturation absolute difference
				diffSat = abs(candidateHSVmean[1] - neighborHSVmean[1]);

				// 6. Compare LBP histograms using BHATTACHARYYA
				LBPratio = compareHist(candidateLBPhistogram, neighborLBPhistogram, CV_COMP_BHATTACHARYYA);

			
				//Mat candidateMask, neighborMask;

				//image.copyTo(candidateMask, candidateMasks[i]);
				//image.copyTo(neighborMask, masks[neighbor]);

				//namedWindow("candidateMask", WINDOW_NORMAL);
				//namedWindow("neighborMask", WINDOW_NORMAL);
				//imshow("candidateMask", candidateMask);
				//imshow("neighborMask", neighborMask);

				//imwrite("D:/workspace/T2/data/candidateMask.jpg", candidateMask);
				//imwrite("D:/workspace/T2/data/neighborMask.jpg", neighborMask);

				//cout << "Hue: " << diffHue << endl << "Sat: " << diffSat << endl << "LBP: " << LBPratio << endl << "diffEdgeRatio: " << diffEdgeRatio << endl;
				//waitKey(0);
			

				// 7. Threshold the comparisons
				// diffEdgeRatio >= -0.05 && LBPratio <= 0.25 to threshold the texture similarity
				// if diffHue <= 25 then consider it a shadow candidate
				// if diffHue is higher but still <= 50, then use a more strict texture condition
				if((diffEdgeRatio >= -0.05 && LBPratio <= 0.25) && (diffHue <= 25 || (diffHue <= 50 && LBPratio <= 0.15)))
				{
					// If the candidate has at least one "similar" neighbor than add it to the vector of shadows
					shadows.push_back(candidateMasks[i]);

					// No need to check other neighbors
					j = neighbors.size();
				}
			}
			
		}
	}

	return shadows;
};