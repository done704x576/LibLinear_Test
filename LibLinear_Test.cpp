// LibLinear_Test.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include "CxLibLinear.h"
#include <time.h>
#include <iostream>
using namespace std;

void gen_train_sample(vector<vector<double>>& x, vector<double>& y, long sample_num, long dim, double scale);

void gen_test_sample(vector<double>& x, long sample_num, long dim, double scale);

int _tmain(int argc, _TCHAR* argv[])
{
	//��ʼ��liblinear����
	CxLibLinear    linear;
	linear.init_linear_param(linear.param);

	/*1��׼��ѵ������*/
	vector<vector<double>>    x;    //������
	vector<double>    y;            //��������ǩ��
	gen_train_sample(x, y, 2000, 288, 1);

	/*1��������֤*/
	int fold = 10;
	linear.do_cross_validation(x, y, linear.param, fold);

	/*2��ѵ��*/
	linear.train_model(x, y, linear.param);

	/*3������ģ��*/
	string model_path = "linear_model.txt";
	linear.save_linear_model(model_path);

	/*4������ģ��*/
	linear.load_linear_model(model_path);

	/*5��Ԥ��*/
	//���������������
	vector<double> x_test;
	gen_test_sample(x_test, 2000, 288, -1);
	double prob_est;
	//Ԥ��
	int value = linear.do_predict(x_test, prob_est);

	//��ӡԤ�����͸���
	printf("label: %d ", value);

	return 0;
}

void gen_train_sample(vector<vector<double>>& x, vector<double>& y, long sample_num, long dim, double scale)
{
	srand((unsigned)time(NULL));//�����
	//�����������������
	for (int i = 0; i < sample_num; i++)
	{
		vector<double> rx;
		for (int j = 0; j < dim; j++)
		{
			rx.push_back(scale*(rand() % dim));
		}
		x.push_back(rx);
		y.push_back(1);
		//printf("y = %d \n",(int)y[i]);
	}

	//��������ĸ�������
	for (int m = 0; m < sample_num; m++)
	{
		vector<double> rx;
		for (int n = 0; n < dim; n++)
		{
			rx.push_back(-scale*(rand() % dim));
		}
		x.push_back(rx);
		y.push_back(2);
		//printf("y = %d \n",(int)y[sample_num + m]);
	}
}

void gen_test_sample(vector<double>& x, long sample_num, long dim, double scale)
{
	srand((unsigned)time(NULL));//�����
	//�����������������
	for (int j = 0; j < dim; j++)
	{
		x.push_back(-scale*(rand() % dim));
	}
}

