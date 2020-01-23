#include "Utility.h"

// ---------------------- SELECT THE DATASET -------------------------

String Utility::datasetSelection()
{
	String path;

	// Stopping-condition
	bool stop = false;

	while (!stop)
	{
		cout << "Choose one of the following datasets (type the index):" << endl;
		cout << "1 - Random Dataset" << endl << "2 - Nature Dataset" << endl << "3 - LawnBowling Dataset" << endl << "4 - Black Dataset" << endl;

		// Number of the dataset
		int number;

		// Input dataset number
		cin >> number;

		if (number == 1)
		{
			stop = true;
			cout << "You chose dataset 1 which contains Random images" << endl;
			path = ("D:/workspace/T2/data/Random/%01d.jpg");
		}
		else if (number == 2)
		{
			stop = true;
			cout << "You chose dataset 2 which contains Nature images" << endl;
			path = ("D:/workspace/T2/data/Nature/%01d.jpg");
		}
		else if (number == 3)
		{
			stop = true;
			cout << "You chose dataset 3 which contains LawnBowling images" << endl;
			path = ("D:/workspace/T2/data/LawnBowling/%01d.jpg");
		}
		else if (number == 4)
		{
			stop = true;
			cout << "You chose dataset 4 which contains Black images" << endl;
			path = ("D:/workspace/T2/data/Black/%01d.jpg");
		}
		else
			cout << "Wrong input, it must be a number between 1 and 4" << endl;
	}

	return path;
};



// ---------------------- LOAD SET OF IMAGES --------------------------

vector<Mat> Utility::loadImages(String& path)
{
	vector<Mat> loadedImages;
	VideoCapture cap(path);
	Mat image;
	for (;;)
	{
		cap >> image;
		if (image.empty())
		{
			break;
		}
		loadedImages.push_back(image.clone());
	}
	return loadedImages;
};



// ------------------------- RESAMPLE IMAGE ----------------------------

void Utility::resampleImage(Mat& image)
{
	while (image.rows >= 2000 || image.cols >= 2000)
	{
		// Use INTER_AREA as interpolation since it works good for decimation
		resize(image, image, Size(), 0.8, 0.8, INTER_AREA);
	}
};