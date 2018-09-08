#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <vector>
#include <stdlib.h> 
#include <time.h> 

using namespace std;
using namespace cv;
const int K = 2;

double getmax(double ary[], int *index)
{
	double max = ary[0];
	*index = 0;
	for (int i = 0; i < K; i++)
	if (ary[i]>max){ max = ary[i]; *index = i; }
	return max;
}
void nonrepeatRand(int* list, int range)
{

	list[0] = rand() % range;
	for (int i = 1; i < 4; i++){
		int g = rand() % range;
		for (int j = 0; j < i; j++)
		while (g == list[j]) g = rand() % range;
		list[i] = g;
	}
}
Mat generateA(vector<KeyPoint> object, vector<KeyPoint> target)
{
	float x1, x2, x3, x4, y1, y2, y3, y4;
	float X1, X2, X3, X4, Y1, Y2, Y3, Y4;

	x1 = target[0].pt.x; y1 = target[0].pt.y;
	x2 = target[1].pt.x; y2 = target[1].pt.y;
	x3 = target[2].pt.x; y3 = target[2].pt.y;
	x4 = target[3].pt.x; y4 = target[3].pt.y;

	X1 = object[0].pt.x; Y1 = object[0].pt.y;
	X2 = object[1].pt.x; Y2 = object[1].pt.y;
	X3 = object[2].pt.x; Y3 = object[2].pt.y;
	X4 = object[3].pt.x; Y4 = object[3].pt.y;

	float data[8][9] = {
		{ X1, Y1, 1, 0, 0, 0, -x1*X1, -x1*Y1, -x1 },
		{ 0, 0, 0, X1, Y1, 1, -y1*X1, -y1*Y1, -y1 },
		{ X2, Y2, 1, 0, 0, 0, -x2*X2, -x2*Y2, -x2 },
		{ 0, 0, 0, X2, Y2, 1, -y2*X2, -y2*Y2, -y2 },
		{ X3, Y3, 1, 0, 0, 0, -x3*X3, -x3*Y3, -x3 },
		{ 0, 0, 0, X3, Y3, 1, -y3*X3, -y3*Y3, -y3 },
		{ X4, Y4, 1, 0, 0, 0, -x4*X4, -x4*Y4, -x4 },
		{ 0, 0, 0, X4, Y4, 1, -y4*X4, -y4*Y4, -y4 } };

	Mat A = Mat(8, 9, CV_32F, data).clone();
	return A;
}

int main()
{

	srand(time(NULL));

	Mat predone;
	const int ITEMS = 2;
	for (int obji = 0; obji < ITEMS; obji++){

		////-- Step 0: Get file name of object image and target image
		int op;
		string object_name = "empty";
		string target_name = "empty";
		do{
			cout << "(1).object_11.bmp  (2).object_12.bmp  (3).object_21.bmp  (4).object_22.bmp  (5).others (6).lakers.png" << endl;
			cout << "Enter file name of object image : ";
			cin >> op;
			switch (op){
			case 1:
				object_name = "object_11.bmp";
				break;
			case 2:
				object_name = "object_12.bmp";
				break;
			case 3:
				object_name = "object_21.bmp";
				break;
			case 4:
				object_name = "object_22.bmp";
				break;
			case 5:
				cout << "enter file name : ";
				cin >> object_name;
				break;
			case 6:
				object_name = "lakers.png";
				break;
			default:
				object_name = "empty";
			}
		} while (object_name.compare("empty") == 0);
		cout << object_name << endl << endl;

		do{
			cout << "(1).target.bmp  (2).others (3).lakers_court.jpg" << endl;
			cout << "Enter file name of target image : ";
			cin >> op;
			switch (op){
			case 1:
				target_name = "target.bmp";
				break;
			case 2:
				cout << "enter file name : ";
				cin >> target_name;
				break;
			case 3:
				target_name = "lakers_court.jpg";
				break;
			default:
				target_name = "empty";
			}
		} while (target_name.compare("empty") == 0);
		cout << target_name << endl << endl;

		Mat img_1 = imread(object_name);
		Mat img_2 = imread(target_name);


		////-- Step 1: Detect the keypoints:
		cout << endl << "detecting keypoints..." << endl;
		Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
		std::vector<KeyPoint> keypoints_1, keypoints_2;
		f2d->detect(img_1, keypoints_1);
		f2d->detect(img_2, keypoints_2);


		////-- Step 2: Calculate descriptors (feature vectors) 
		cout << endl << "caculating descriptors..." << endl;
		Mat descriptors_1, descriptors_2;
		f2d->compute(img_1, keypoints_1, descriptors_1);
		f2d->compute(img_2, keypoints_2, descriptors_2);

		Mat feat1, feat2;
		drawKeypoints(img_1, keypoints_1, feat1);
		drawKeypoints(img_2, keypoints_2, feat2);
		imwrite("Object_KeyPoint.jpg", feat1);
		imwrite("Target_KeyPoint.jpg", feat2);
		//imshow("Object_KeyPoint.jpg", feat1);
		//imshow("Target_KeyPoint.jpg", feat2);
		int key1 = keypoints_1.size();
		int key2 = keypoints_2.size();
		printf("Keypoint1=%d \nKeypoint2=%d \n", key1, key2);
		printf("Descriptor1=(%d,%d) \nDescriptor2=(%d,%d)\n", descriptors_1.size().height, descriptors_1.size().width, descriptors_2.size().height, descriptors_2.size().width);


		/////--Step 3: Find 2NN points in the target image(according to the descriptors), and exclude the ambiguous candidates
		cout << endl << "doing 2NN..." << endl;
		Mat kd1, kd2;
		vector<KeyPoint> ForwardMatches(descriptors_1.rows * K);
		vector<KeyPoint> Matches(K);
		int *match_recheck;

		match_recheck = new int[key1];

		const float MIN_ABS = 100; // 50 for lakers.png
		double dist;
		double min_list[K];
		double max_in_list;
		int    max_index;
		for (int i = 0; i < descriptors_1.rows; i++)
		{
			kd1 = descriptors_1(Rect(0, i, 128, 1));
			for (int j = 0; j < descriptors_2.rows; j++){
				kd2 = descriptors_2(Rect(0, j, 128, 1));
				dist = norm(kd1, kd2, NORM_L2);
				if (j < K){
					min_list[j] = dist;
					Matches[j] = keypoints_2[j];
				}
				else{
					max_in_list = getmax(min_list, &max_index);
					if (dist < max_in_list){
						min_list[max_index] = dist;
						Matches[max_index] = keypoints_2[j];
					}
				}
			}
			ForwardMatches[i * K] = Matches[0];
			ForwardMatches[i * K + 1] = Matches[1];

			float a = min_list[0];
			float b = min_list[1];
			if (abs(a - b) < MIN_ABS) match_recheck[i] = 0;  // 0 for ambiguous candidates
			else match_recheck[i] = 1; // 1 for good candidates
		}

		int count = 0;
		for (int i = 0; i < key1; i++)
		if (match_recheck[i] == 1) count++;

												//downsize the keypoints_1 and ForwardMatches by exclude ambiguous candidates
		vector<KeyPoint> kp1(count), kp2(count);
		int i2 = 0;
		for (int i = 0; i < key1; i++){
			if (match_recheck[i] == 1){
				kp1[i2] = keypoints_1[i];
				kp2[i2] = ForwardMatches[i * 2];
				i2++;
			}
		}
		/*
		Mat f1, f2;
		drawKeypoints(img_1, kp1, f1);
		drawKeypoints(img_2, kp2, f2);
		imwrite("kp1.jpg", f1);
		imwrite("kp2.jpg", f2);
		*/

		///--Step4: RANSAC
		const int k = 4;
		const float THRESHOLD = 0.7;
		const int MAX_TIMES = 10000;
		const float MIN_DIST = 100;
		cout << endl << "doing RANSAC..." << endl;
		cout << "Thereshold : " << THRESHOLD << endl;
		cout << "Maximum times : " << MAX_TIMES << endl;

		float inlier_percentage = 0;
		float best_percentage = -1;
		int times = 0;
		bool found = false;

		vector<KeyPoint> ConsesusSet;
		vector<KeyPoint> best_ConsesusSet;

		Mat best_ProjMat(Mat::zeros(3, 3, CV_32F));

		while (!found && times < MAX_TIMES){

			times++;

			int rand_index[k];
			nonrepeatRand(rand_index, kp1.size());

			vector<KeyPoint> object(k);
			vector<KeyPoint> target(k);
			for (int i = 0; i < k; i++){
				object[i] = kp1[rand_index[i]];
				target[i] = kp2[rand_index[i]];
			}

			//Caculate Projection Matrrix
			Mat A(Mat::zeros(8, 9, CV_32F));
			Mat EigenValues;
			Mat EigenVectors;

			A = generateA(object, target);
			eigen(A.t()*A, EigenValues, EigenVectors);

			float minEign;
			Mat minEignVec;
			minEign = EigenValues.at<float>(EigenValues.rows - 1, 0);
			minEignVec = EigenVectors(Rect(0, EigenVectors.rows - 1, 9, 1));

			Mat ProjMat(Mat::zeros(3, 3, CV_32F));
			for (int j = 0; j < minEignVec.cols; j++)
				ProjMat.at<float>((int)(j / 3), j % 3) = minEignVec.at<float>(0, j);

			//Counting Inlier points
			Mat p_obj(Mat::zeros(3, 1, CV_32F));
			Mat p_tag(Mat::zeros(3, 1, CV_32F));
			Mat p(Mat::zeros(3, 1, CV_32F));
			int inlier_count = 0;
			for (int j = 0; j < kp1.size(); j++){

				p_obj.at<float>(0, 0) = kp1[j].pt.x;
				p_obj.at<float>(1, 0) = kp1[j].pt.y;
				p_obj.at<float>(2, 0) = 1;

				p_tag = ProjMat*p_obj;
				p_tag = p_tag / p_tag.at<float>(2, 0);

				p.at<float>(0, 0) = kp2[j].pt.x;
				p.at<float>(1, 0) = kp2[j].pt.y;
				p.at<float>(2, 0) = 1;

				float dist = norm(p_tag, p, NORM_L2);
				if (dist < MIN_DIST){
					inlier_count++;
				}
			}
			inlier_percentage = (float)inlier_count / kp1.size();

			if (inlier_percentage > best_percentage){
				best_percentage = inlier_percentage;
				best_ProjMat = ProjMat;
			}

			if (best_percentage > THRESHOLD)
				found = true;

			//cout << inlier_percentage << endl;

		}

		if (found) cout << "found." << endl;
		if (times >= MAX_TIMES) cout << "reach maximum times" << endl;
		cout << "best percentage = " << best_percentage << endl;


		///--Step5: Image Warping
		cout << endl << "image warping...(forward)";
		Mat img_3 = imread(target_name);
		Mat im, imt;
		im = img_1;
		// forward warping
		for (int i = 0; i < im.rows; i++){
			for (int j = 0; j < im.cols; j++){
				imt = im(Rect(j, i, 1, 1));
				Vec3b color = imt.at<Vec3b>(Point(0, 0));
				Vec3b white = Vec3b(255, 255, 255);
				if (color != white)
				{
					Mat proj_point(Mat::zeros(3, 1, CV_32F));
					proj_point.at<float>(0, 0) = j;
					proj_point.at<float>(1, 0) = i;
					proj_point.at<float>(2, 0) = 1;
					proj_point = best_ProjMat*proj_point;
					proj_point = proj_point / proj_point.at<float>(2, 0);

					float px = proj_point.at<float>(0, 0);
					float py = proj_point.at<float>(1, 0);

					if ((px < img_3.cols) && (py < img_3.rows) && (px > 0) && (py > 0)){
						cv::Rect roi(cv::Point(px, py), imt.size());
						imt.copyTo(img_3(roi));
					}
				}
			}
		}
		//imshow("Output.jpg", img_3);
		imwrite("Output.jpg", img_3);
		cout <<  endl <<" Output.jpg" << endl;

		// backward warping
		cout << endl << "image warping...(backward)" << endl;
		if (obji == 0)
			img_3 = imread(target_name);
		else
			img_3 = predone;
		im = img_1;
		for (int i = 0; i < img_3.rows; i++){
			for (int j = 0; j < img_3.cols; j++){

				Mat proj_point(Mat::zeros(3, 1, CV_32F));
				proj_point.at<float>(0, 0) = j;
				proj_point.at<float>(1, 0) = i;
				proj_point.at<float>(2, 0) = 1;
				proj_point = best_ProjMat.inv()*proj_point;
				proj_point = proj_point / proj_point.at<float>(2, 0);

				float px = proj_point.at<float>(0, 0);
				float py = proj_point.at<float>(1, 0);

				if ((px < im.cols) && (py < im.rows) && (px > 0) && (py > 0)){
					imt = im(Rect(px, py, 1, 1));
					Vec3b color = imt.at<Vec3b>(Point(0, 0));
					Vec3b white = Vec3b(255, 255, 255);
					if (color != white){
						cv::Rect roi(cv::Point(j, i), imt.size());
						imt.copyTo(img_3(roi));
					}
				}

			}
		}
		//imshow("Output2.jpg", feat3);
		imwrite("Output2.jpg", img_3);
		cout << "Output2.jpg" << endl << endl;
		cout << "---------------------------------------------------------" << endl << endl;
		predone = img_3;
	}

	if (ITEMS > 1){
		imwrite("Final.jpg", predone);
		cout << "Final.jpg" << endl << endl;
	}

	cout << "done." << endl;

	system("pause");
	return 0;

}