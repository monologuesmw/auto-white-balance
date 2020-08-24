// main.cpp : 定义控制台应用程序的入口点。
//

#include "stdio.h"
#include <iostream>
#include <cmath>
#include <ctime>
#include<cstdlib>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <cv.h>
#include <imgproc.hpp>
using namespace cv;
using namespace std;

const int table_length = 256;


Mat load_img_cv(char* filename)
{
	Mat img = imread(filename, IMREAD_ANYCOLOR);
	return img;
}

void matMultiply(float **mat_b_inv, float mat_g[], float &a, float &b)
{
	a = mat_b_inv[0][0] * mat_g[0] + mat_b_inv[0][1] * mat_g[1];
	b = mat_b_inv[1][0] * mat_g[0] + mat_b_inv[1][1] * mat_g[1];  // 仅限二阶
}

Mat postProcess(float ub, float vb, float ur, float vr, Mat img)
{
	int row = img.rows;
	int col = img.cols;

	Mat b0 = Mat::zeros(row, col, CV_32FC1);
	//Mat_<int>(row, col);
	Mat g0 = Mat::zeros(row, col, CV_32FC1);
	//Mat_<int>(row, col);
	Mat r0 = Mat::zeros(row, col, CV_32FC1);
	//Mat_<int>(row, col);

	float b_point, r_point;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			b_point = ub * pow(img.at<Vec3b>(i, j)[0], 2) + vb * img.at<Vec3b>(i, j)[0];
			//g0[i][j] = g[i][j]
			g0.at<float>(i, j) = img.at<Vec3b>(i, j)[1];
			//int pp = img.at<Vec3b>(i, j)[1];
			//cout << pp << endl;
			r_point = ur*pow(img.at<Vec3b>(i, j)[2], 2) + vr *img.at<Vec3b>(i, j)[2];
			if (r_point > 255.0)
			{
				//r0[i][j] = 255
				r0.at<float>(i, j) = 255;
			}
			else
			{
				if (r_point < 0.0)
				{
					//r0[i][j] = 0
					r0.at<float>(i, j) = 255;
				}
				else
				{
					//r0[i][j] = r_point
					r0.at<float>(i, j) = floor(r_point);
					//int aaa = (int)floor(r_point);
					//cout << aaa << endl;
				}
			}
			if (b_point>255.0)
			{
				//b0[i][j] = 255
				b0.at<float>(i, j) = 255;
			}
			else
			{
				if (b_point<0.0)
				{
					//b0[i][j] = 0
					b0.at<float>(i, j) = 0;
				}
				else
				{
					//b0[i][j] = b_point
					b0.at<float>(i, j) = floor(b_point);
				}
			}
		}
	}
	//cout << "g0=[" << endl << b0 << endl;

	vector<Mat> newChannels = { b0, g0, r0 };
	
	Mat mergedImage;
	merge(newChannels, mergedImage);
	
	newChannels.clear();
	b0.release();
	g0.release();
	r0.release();

	return mergedImage;
}

float **matInv(float mat[2][2])
{
	
	float factor = 1 / (mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]);  // 1/(ad-bc)
	float **out;
	out = (float **)malloc(2*sizeof(float *));
	for (int i = 0; i<2; i++)
		out[i] = (float *)malloc(2 * sizeof(float));
	out[0][0] = factor * mat[1][1];  
	out[0][1] = factor * (-mat[0][1]);  
	out[1][0] = factor * (-mat[1][0]); 
	out[1][1] = factor * mat[0][0];

	return out;
}

void valueCount(Mat img, int & sum_I_r, int & sum_I_b, int & sum_I_g, uint64_t & sum_I_r_2, uint64_t & sum_I_b_2, int & max_I_r_2, int & max_I_r, int & max_I_b_2, int & max_I_b, int & max_I_g)
{
	int col = img.cols;
	int row = img.rows;
	vector<vector<int> > I_r_2(row, vector<int>(col));
	vector<vector<int> > I_b_2(row, vector<int>(col));
	
	max_I_r_2 = pow(img.at<Vec3b>(0, 0)[1], 2);
	max_I_r = img.at<Vec3b>(0, 0)[1];
	max_I_b_2 = pow(img.at<Vec3b>(0, 0)[0], 2);
	max_I_b = img.at<Vec3b>(0, 0)[0];
	max_I_g = img.at<Vec3b>(0, 0)[2];
	//cout << max_I_b << endl;
	//cout << max_I_r << endl;
	//cout << max_I_g << endl;

	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			//I_r_2[i][j] = int(r[i][j] * * 2)
			//I_b_2[i][j] = int(b[i][j] * * 2)
			I_r_2[i][j] = pow(img.at<Vec3b>(i, j)[2], 2);
			I_b_2[i][j] = pow(img.at<Vec3b>(i, j)[0], 2);
			//cout << I_r_2[i][j] << endl;
			/*
			sum_I_r_2 = I_r_2[i][j] + sum_I_r_2
			sum_I_b_2 = I_b_2[i][j] + sum_I_b_2
			sum_I_g = g[i][j] + sum_I_g
			sum_I_r = r[i][j] + sum_I_r
			sum_I_b = b[i][j] + sum_I_b
			*/
			sum_I_r_2 = I_r_2[i][j] + sum_I_r_2;
			sum_I_b_2 = I_b_2[i][j] + sum_I_b_2;
			sum_I_g = img.at<Vec3b>(i, j)[1] + sum_I_g;
			sum_I_r = img.at<Vec3b>(i, j)[2] + sum_I_r;
			sum_I_b = img.at<Vec3b>(i, j)[0] + sum_I_b;
			/*
			if max_I_r < r[i][j]:
				max_I_r = r[i][j]
			*/
			if (max_I_r < img.at<Vec3b>(i, j)[2])
			{
				max_I_r = img.at<Vec3b>(i, j)[2];
			}
			/*
			if max_I_r_2 < I_r_2[i][j]:
				max_I_r_2 = I_r_2[i][j]
			*/
			if (max_I_r_2 < I_r_2[i][j])
			{
				max_I_r_2 = I_r_2[i][j];
			}
			/*
			if max_I_g < g[i][j]:
				 max_I_g = g[i][j]
			*/
			if (max_I_g < img.at<Vec3b>(i, j)[1])
			{
				max_I_g = img.at<Vec3b>(i, j)[1];
			}
			/*
			if max_I_b_2 < I_b_2[i][j]:
				max_I_b_2 = I_b_2[i][j]
			*/
			if (max_I_b_2 < I_b_2[i][j])
			{
				max_I_b_2 = I_b_2[i][j];
			}
			/*
			if max_I_b < b[i][j]:
				max_I_b = b[i][j]
			*/
			if (max_I_b < img.at<Vec3b>(i, j)[0])
			{
				max_I_b = img.at<Vec3b>(i, j)[0];
			}
		}
	}
 	I_r_2.clear();
	I_b_2.clear();
}

struct Table
{
	int b[256];  // b的0-255
	int g[256];  // g的0-255
	int r[256];  // r的0-255        左对照表

	//int index_b[256];   // 索引

	int b0[256];  
	int g0[256];
	int r0[256];    // 右对照表

	int ptr[256];   // 当前像素点所在的位置

	int flag[256];  // 是否更改置位符
	Table()
	{
		memset(this, 0, sizeof(Table));
	}
};

Table table_init()
{
	Table table;
	for (int i = 0; i < table_length; i++)
	{
		table.b[i] = i;
		table.g[i] = i;
		table.r[i] = i;

		//table.index_b[i] = i;
		table.flag[i] = 0;

		table.b0[i] = i;
		table.g0[i] = i;
		table.r0[i] = i;

		table.ptr[i] = i;
	}
	return table;
}

Table tableReset(Table table)
{
	for (int i = 0; i < table_length; i++)
	{
		//table.index_b[i] = i;
		table.flag[i] = 0;
		table.ptr[i] = i;
	}
	return table;
}


void swap(int *a, int *b)
{
	int c;
	c = *a;
	*a = *b;
	*b = c;
}


Table singleUpdate(Mat img, Mat new_img, Table table, char type)
{
	int row = img.rows;
	int col = img.cols;
	img.convertTo(img, CV_8UC1);
	new_img.convertTo(new_img, CV_8UC1);

	int left_point, right_point;
	int left_position, right_position;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			//cout << img.at<int>(i, j) << endl;
			left_point = img.at<uchar>(i, j);
			right_point = new_img.at<uchar>(i, j);

			left_position = table.ptr[left_point];
			if (table.flag[left_position] == -1)
				continue;
			//right_position = table.ptr[right_point];
			switch (type)
			{
			case 'b':
				table.b0[left_position] = right_point;
				break;
			case 'g':
				table.g0[left_position] = right_point;
				break;
			case 'r':
				table.r0[left_position] = right_point;
				break;
			default:
				break;
			}
			table.flag[left_position] = -1;
			//cout << endl;
		}
	}

	return table;
}

Table table_update(Mat img, Mat new_img, Table table)
{

	vector<Mat> channels, channels_new;
	Mat b, g, r, b0, g0, r0;

	split(img, channels);
	split(new_img, channels_new);

	// 前后各通道
	b = channels[0]; g = channels[1]; r = channels[2];
	b0 = channels_new[0]; g0 = channels_new[1]; r0 = channels_new[2];
	
	channels.clear();
	channels_new.clear();

	// b
	char type = 'b';
	table = singleUpdate(b, b0, table, type);
	type = 'g';
	table = tableReset(table);
	table = singleUpdate(g, g0, table, type);
	type = 'r';
	table = tableReset(table);
	table = singleUpdate(r, r0, table, type);

	b.release();
	b0.release();
	g.release();
	g0.release();
	r.release();
	r0.release();

	return table;
}


int main()
{	
	clock_t start, end;

	Mat img;
	char* filename = "E:\\VT_PROJECT\\uav\\camera_photo_1.png";  // example
	
	img = load_img_cv(filename);

	start = clock();
	
	Table table = table_init();
	
	int sum_I_r = 0, sum_I_b = 0, sum_I_g = 0;
	uint64_t sum_I_r_2 = 0, sum_I_b_2 = 0;
	int max_I_r_2 = 0, max_I_r = 0, max_I_b_2 = 0, max_I_b = 0, max_I_g = 0;
	
	valueCount(img, sum_I_r, sum_I_b, sum_I_g, sum_I_r_2, sum_I_b_2, max_I_r_2, max_I_r, max_I_b_2, max_I_b, max_I_g);

	float mat_b[2][2] = { { sum_I_b_2 , sum_I_b }, {max_I_b_2 , max_I_b } };
	float mat_r[2][2] = { { sum_I_r_2 , sum_I_r }, { max_I_r_2 , max_I_r } };
	float mat_g[2] = { sum_I_g , max_I_g };

	float **mat_b_inv;
	float **mat_r_inv;

	mat_b_inv = matInv(mat_b);  // calc inv
	mat_r_inv = matInv(mat_r);

	float ub, vb;
	float ur, vr;

	matMultiply(mat_b_inv, mat_g, ub, vb);
	matMultiply(mat_r_inv, mat_g, ur, vr);   // mat multiply

	Mat new_img = postProcess(ub, vb, ur, vr, img);
	
	// update table
	table = table_update(img, new_img, table);

	end = clock();
	double endtime = (double)(end - start) / CLOCKS_PER_SEC;
	cout << "total time:" << endtime * 1000 << "ms" << endl;

	// table cout
	cout << "b:" << endl;
	for (int i = 0; i < table_length; i++)
	{
		cout << table.b[i] << ",";
	}
	cout << endl;
	
	for (int i = 0; i < table_length; i++)
	{
		cout << table.b0[i] << ",";
	}
	cout << endl;

	cout << "g:" << endl;
	for (int i = 0; i < table_length; i++)
	{
		cout << table.g[i] << ",";
	}
	cout << endl;
	for (int i = 0; i < table_length; i++)
	{
		cout << table.g0[i] << ",";
	}
	cout << endl;

	cout << "r:" << endl;
	for (int i = 0; i < table_length; i++)
	{
		cout << table.r[i] << ",";
	}
	cout << endl;
	for (int i = 0; i < table_length; i++)
	{
		cout << table.r0[i] << ",";
	}
	cout << endl;

	// img display
	Mat new_img_1;
	//cout << new_img << endl;
	new_img.convertTo(new_img_1, CV_8UC3);
	imshow("white_balance:", new_img_1);
	waitKey(0);

	img.release();
	new_img.release();
	new_img_1.release();

	return 0;
}