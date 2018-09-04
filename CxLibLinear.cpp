#include "stdafx.h"
#include "CxLibLinear.h"

CxLibLinear::CxLibLinear()
{
	model_ = NULL;
}

CxLibLinear::~CxLibLinear()
{
	free_model();
}

void CxLibLinear::init_linear_param(struct parameter& param)
{
	//参数初始化，参数调整部分在这里修改即可
	// 默认参数
	param.solver_type = L2R_L2LOSS_SVC;
	param.eps = 0.001;
	param.C = 2;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	param.p = 0.001;
	param.init_sol = NULL;
}

void CxLibLinear::train_model(const vector<vector<double>>& x, const vector<double>& y, const struct parameter& param)
{
	if (x.size() == 0)
	{
		return;
	}

	//释放先前的模型
	free_model();

	/*初始化*/        
	long    len = x.size();
	long    dim = x[0].size();
	long    elements = len * dim;

	//转换数据为liblinear格式
	prob.l = len;
	prob.n = dim;
	prob.bias = -1.0;
	prob.y = Malloc(double, prob.l);
	prob.x = Malloc(struct feature_node *, prob.l);
	x_space = Malloc(struct feature_node, elements + len);
	int j = 0;
	for (int l = 0; l < len; l++)
	{
		prob.x[l] = &x_space[j];
		for (int d = 0; d < dim; d++)
		{                
			x_space[j].index = d+1;
			x_space[j].value = x[l][d];    
			j++;
		}
		x_space[j++].index = -1;
		prob.y[l] = y[l];
	}

	/*训练*/
	model_ = train(&prob, &param);    
}

int CxLibLinear::do_predict(const vector<double>& x,double& prob_est)
{
	//int nr_class=get_nr_class(model_);
	//double *prob_estimates=NULL;
	//int n;
	//int nr_feature=get_nr_feature(model_);

	//if(model_->bias>=0)
	//	n=nr_feature+1;
	//else
	//	n=nr_feature;

	//double predict_label;
	//feature_node* x_test = Malloc(struct feature_node, x.size()+1);
	//for (unsigned int i = 0; i < x.size(); i++)
	//{
	//	if(model_->bias>=0)
	//	{
	//		x_test[i].index = n;
	//		x_test[i].value = model_->bias;
	//	}
	//	else
	//	{
	//		x_test[i].index = i + 1;
	//		x_test[i].value = x[i];
	//	}
	//
	//}
	//x_test[x.size()].index = -1;
	//
	//predict_label = predict(model_,x_test);
	//
	//return (int)predict_label;

	//数据转换
	feature_node* x_test = Malloc(struct feature_node, x.size()+1);
	for (unsigned int i = 0; i < x.size(); i++)
	{
		x_test[i].index = i + 1;
		x_test[i].value = x[i];
	}
	x_test[x.size()].index = -1;

	double *probs = new double[model_->nr_class];//存储了所有类别的概率
	//预测类别和概率
	int value = (int)predict_values(model_, x_test, probs);
	for (int k = 0; k < model_->nr_class; k++)
	{//查找类别相对应的概率
		if (model_->label[k] == value)
		{
			prob_est = probs[k];
			break;
		}
	}
	delete[] probs;
	return value;
}

void CxLibLinear::do_cross_validation(const vector<vector<double>>& x, const vector<double>& y, const struct parameter& param, const int & nr_fold)
{
	if (x.size() == 0)
		return;

	/*初始化*/
	long    len = x.size();
	long    dim = x[0].size();
	long    elements = len*dim;

	//转换数据为liblinear格式
	prob.l = len;
	prob.n = dim;
	prob.y = Malloc(double, prob.l);
	prob.x = Malloc(struct feature_node *, prob.l);
	x_space = Malloc(struct feature_node, elements + len);
	int j = 0;
	for (int l = 0; l < len; l++)
	{
		prob.x[l] = &x_space[j];
		for (int d = 0; d < dim; d++)
		{
			x_space[j].index = d + 1;
			x_space[j].value = x[l][d];
			j++;
		}
		x_space[j++].index = -1;
		prob.y[l] = y[l];
	}

	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double, prob.l);

	cross_validation(&prob, &param, nr_fold, target);
	if(param.solver_type == L2R_L2LOSS_SVR ||
	   param.solver_type == L2R_L1LOSS_SVR_DUAL ||
	   param.solver_type == L2R_L2LOSS_SVR_DUAL)
	{
		for (i = 0; i < prob.l; i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v - y)*(v - y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
		printf("Cross Validation Mean squared error = %g\n", total_error / prob.l);
		printf("Cross Validation Squared correlation coefficient = %g\n",
			((prob.l*sumvy - sumv*sumy)*(prob.l*sumvy - sumv*sumy)) /
			((prob.l*sumvv - sumv*sumv)*(prob.l*sumyy - sumy*sumy))
			);
	}
	else
	{
		for (i = 0; i < prob.l; i++)
			if (target[i] == prob.y[i])
				++total_correct;
		printf("Cross Validation Accuracy = %g%%\n", 100.0*total_correct / prob.l);
	}
	free(target);
}

int CxLibLinear::load_linear_model(string model_path)
{
	//释放原来的模型
	free_model();

	//导入模型
	model_ = load_model(model_path.c_str());
	if (model_ == NULL)
		return -1;

	return 0;
}

int CxLibLinear::save_linear_model(string model_path)
{
	int flag = save_model(model_path.c_str(), model_);
	return flag;
}

void CxLibLinear::free_model()
{
	if (model_ != NULL)
	{
		free_and_destroy_model(&model_);
		destroy_param(&param);

		if (prob.y != NULL)
		{
			free(prob.y);
			prob.y = NULL;
		}

		if (prob.x != NULL)
		{
			free(prob.x);
			prob.x = NULL;
		}

		if (x_space != NULL)
		{
			free(x_space);
			x_space = NULL;
		}
	}
}