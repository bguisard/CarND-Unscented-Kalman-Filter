#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

  // Initializing rmse vector
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // We dinamically dimsension rmse to allow for higher dimensional
  // spaces. The decision to use Ground Truth size instead of estimation
  // size is to ensure it won't crash if our estimation is not working.
  //VectorXd rmse = VectorXd::Zero(ground_truth[0].size());

  // Check dimensions of inputs against three constrains
  // The size of estimations must match the size of ground_truth
  // They cannot be zero - because we will also check they are equal
  // we just need to make sure one of them is non-zero.
  if (estimations.size() != ground_truth.size() 
        || estimations.size() == 0) {
    std::cout << "Invalid input data - check est or gt data." << endl;
    return rmse;  // returns the zero-initialized vector
  }

  // iterate through estimations and calculate the error
  for (unsigned int i=0; i < estimations.size(); ++i) {

    // calculate the error in our estimation
    VectorXd residual = estimations[i] - ground_truth[i];

    // convert error to squared error
    residual = residual.array() * residual.array();

    // accumulate square residuals
    rmse += residual;
  }

  // calculate the mean
  rmse = rmse / estimations.size();

  // calculate the squared root
  rmse = rmse.array().sqrt();

  // return the root mean squared error
  return rmse;
}