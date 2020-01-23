#include "ShadowDetector.h"

int main(int argc, char** argv)
{	
	// Create objects
	ShadowDetector detector;
	Utility util;

	// Choose the dataset
	String path = util.datasetSelection();

	// Load the images of the dataset chosen
	vector<Mat> images = util.loadImages(path);

	// Create the window for displaying each image
	namedWindow("Image", WINDOW_NORMAL);

	Mat image;

	for (int k = 0; k < images.size(); k++)
	{	
		image = images[k];

		// Resample image if it's too big
		if(image.rows >= 2000 || image.cols >= 2000)
			util.resampleImage(image);

		imshow("Image", image);


		// ----------------------------------------- PRE-PROCESSING FILTERING TO IMPROVE SEGMENTATION -----------------------------------------

	/*
		Bilateral filter and Closure operation are applied to the image in order to remove (partially) noise and small holes.
		This step is not mandatory but it helps to obtain more precise segmentation of the image.
	*/

		Mat filteredImage = detector.applyFilters(image);

	/*
		namedWindow("Image after Preprocessing", WINDOW_NORMAL);
		imshow("Image after Preprocessing", filteredImage);
		waitKey(0);
	*/


	// -------------------------------------------------------------- SEGMENTATION ---------------------------------------------------------------
		
	/*
		Consider the image in the HSV color space and segment it by using kmeans. Samples given in input to kmeans will contain both the color information (H, S, V channels) and the position.
	*/

		// Compute samples for kmeans
		Mat samples = detector.computeSamples(filteredImage);
		
	/*
		Parameters for kmeans:
			numSegments: Number of segments to split the set by
			labels: Output integer array that stores the segment indices for every sample
			attempts: Flag to specify the number of times the algorithm is executed using different initial labellings. The algorithm returns the labels that yield the best compactness
			centers: Output matrix of the segment centers, one row per each segment center
	*/
		
		const int numSegments = 100;
		vector<int> labels;
		int attempts = 3;
		Mat centers;
		
		// Run K-means algorithm and return segmentation result
		Mat segmentedImage = detector.runKmeans(filteredImage, samples, numSegments, labels, attempts, centers);

	/*
		namedWindow("Segmented image", WINDOW_NORMAL);
		imshow("Segmented image", segmentedImage);
		waitKey(0);
	*/


	// ------------------------------------------------------------ INTENSITY THRESHOLDING ---------------------------------------------------------------

	/*
		Since shadows usually have a lower intensity then the major part of the image, it's rather effective to threshold the intensity of the resulting segmentation.
		Thresholding can be done both for the lower intensities and the upper intensities.
		The lower threshold allows to erase many false positives (such as some black objects) since shadows cannot be too dark.
		The upper threshold allows to consider far less segments in the detection since shadows usually are contained in the segments with the lowest intensities.
		If the two thresholds are chosen too strict, they give birth to many false negatives. On the other hand, they cannot even be chosen too soft, since they would give birth to many false positives.
	*/

		Mat thresholdedImage = detector.intensityThresholding(segmentedImage);

	/*
		namedWindow("Thresholded image", WINDOW_NORMAL);
		imshow("Thresholded image", thresholdedImage);
		waitKey(0);
	*/


	// ---------------------------------------------------------------- MASKS COMPUTATION ----------------------------------------------------------------

	/*
		Compute a mask for each segment.
	*/

		vector<Mat> masks = detector.computeMasks(numSegments, segmentedImage, labels);

	/*
		for (int x = 0; x < masks.size(); x++)
		{
			Mat maskImage;
			image.copyTo(maskImage, masks[x]);
			namedWindow("Masks", WINDOW_NORMAL);
			imshow("Masks", maskImage);
			waitKey(0);
		}
	*/


	// -------------------------------------------------------------- CANDIDATES COMPUTING ---------------------------------------------------------------

	/*
		Find those segments that are contained in the white components of thresholdedImage and consider them as shadow candidates.
		The method returns a vector which containes the indexes of all the candidate segments.
	*/

		vector<int> candidates = detector.computeCandidates(filteredImage, thresholdedImage, labels);

	/*
		for (int x = 0; x < candidates.size(); x++)
		{
			Mat candidateImage;
			image.copyTo(candidateImage, masks[candidates[x]]);
			namedWindow("Candidate", WINDOW_NORMAL);
			imshow("Candidate", candidateImage);
			waitKey(0);
		}
	*/


	// ------------------------------------------------------ COMPUTE LOCAL BINARY PATTERN IMAGE ----------------------------------------------------------

	/*
		Since Local Binary Patterns are invariant to both gray scale and rotation, they can be used to capture the texture information contained in each segment.
		Consider each pixel of the image (converted to grayscale) and center a 3x3 window on it. Compute the LBP value of the considered pixel using the other pixels in the window.
		For each LBP value computed, assign it to the corresponding pixel (same coordinates as the pixel in the center of the window) in the LBPimage returned by the algorithm.
	*/

		Mat LBPimage = detector.computeLBP(image);

	/*
		namedWindow("LBPimage", WINDOW_NORMAL);
		imshow("LBPimage", LBPimage);
		waitKey(0);
	*/


	// ------------------------------------------------------------- MERGE 'SIMILAR' CANDIDATES ------------------------------------------------------------

	/*
		For every candidate, compute its neighbors and check if any of them is enough "similar" to the candidate to be merged with.
		The similarity is computed by using both color (absolute difference of the mean of each channel) and texture (LBP histogram comparison) information.
	*/

		// Find the list of clusters that have to be merged
		Mat mergeTable = detector.computeMergeTable(candidates, masks, image, LBPimage);

		// Compute the new set of masks for the merged clusters
		vector<Mat> candidateMasks = detector.applyMerge(candidates, mergeTable, masks);

	/*
		for (int x = 0; x < candidateMasks.size(); x++)
		{
			Mat candidates;
			image.copyTo(candidates, candidateMasks[x]);
			namedWindow("Candidates", WINDOW_NORMAL);
			imshow("Candidates", candidates);
			waitKey(0);
		}
	*/


	// --------------------------------------------------------------- CANDIDATES COMPARISON ----------------------------------------------------------------

	/*
		Compare each candidate with its non-candidate neighbors. The comparison is based on texture, chromaticity and gradient similarity.
		If a candidate has similar properties with one of its non-candidate neighbors (at least one), then consider it as a shadow segment.
	*/

		vector<Mat> shadows = detector.compareCandidates(candidates, masks, image, LBPimage, candidateMasks);


		// Show the final shadow mask
		Mat shadowMask = Mat::zeros(masks[0].size(), masks[0].type());
		for (int x = 0; x < shadows.size(); x++)
			bitwise_or(shadowMask, shadows[x], shadowMask);

		namedWindow("Shadow", WINDOW_NORMAL);
		imshow("Shadow", shadowMask);
		imwrite("D:/workspace/T2/data/shadowMask.jpg", shadowMask);
		waitKey(0);
	
	

	// -------------------------------------------------------------------------------------------------------------------------------------------------------

	}

	return 0;
}