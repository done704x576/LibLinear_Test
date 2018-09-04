#ifndef _CXLIBLINEAR_H_H_
#define _CXLIBLINEAR_H_H_

#include <string>
#include <vector>
#include <iostream>
#include "linear.h"

using namespace std;

//ÄÚ´æ·ÖÅä
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

class CxLibLinear
{
public:

	struct parameter param;

private:

	struct feature_node *x_space;

	struct problem prob;

	struct model* model_;

public:

	CxLibLinear();

	~CxLibLinear();

	void init_linear_param(struct parameter& param);

	void train_model(const vector<vector<double>>&  x, const vector<double>& y, const struct parameter& param);

	int do_predict(const vector<double>& x,double& prob_est);

	void do_cross_validation(const vector<vector<double>>& x, const vector<double>& y, const struct parameter& param, const int & nr_fold);

	int load_linear_model(string model_path);

	int save_linear_model(string model_path);

	void free_model();
};

#endif
