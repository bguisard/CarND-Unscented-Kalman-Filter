#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
public:

  ///* initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  ///* if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  ///* if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  ///* state covariance matrix
  MatrixXd P_;

  ///* predicted sigma points matrix
  MatrixXd Xsig_pred_;

  ///* time when the state is true, in us
  long long time_us_;

  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  ///* Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  ///* Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  ///* Radar measurement noise standard deviation radius in m
  double std_radr_;

  ///* Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  ///* Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  ///* Weights of sigma points
  VectorXd weights_;

  ///* State dimension
  int n_x_;

  ///* Augmented state dimension
  int n_aug_;

  ///* Sigma point spreading parameter
  double lambda_;

  ///* intermediate matrix used to calc state covariance
  MatrixXd X_diff_;

  ///* intermediate matrix used to calc state cross covariance
  MatrixXd Z_diff_;

  ///* laser covariance matrix
  MatrixXd R_laser_;

  ///* radar covariance matrix
  MatrixXd R_radar_;

  ///* if this is true, the program will output files for NIS for laser and radar
  bool write_NIS_;

  ///* if this is true, the program will print x_ and P_ to a file instead of screen
  bool write_state_to_file_;

  int k_iteration;

  ///* if this is true, the program will print debugging outputs to a file
  bool write_debug_file_;

  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(const VectorXd &z);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(const VectorXd &z);

/**
 * Generates matrix of augmented Sigma points
 * @param (pointer to Xsig_aug)
 */
  void GenerateAugmentedSigmaPoints(MatrixXd* Xsig_aug_);

/**
 * Predicts Augmented Sigma points for a given delta_t
 * @param delta_t
 */
  void PredictSigmaPoints(double delta_t);

/**
 * Predicts State Mean and Covariance
 */
  void PredictMeanAndCovariance();

/**
 * Normalizes angles between -PI and +PI
 * @param (double angle)
 */
  double NormalizeAngle(double angle);
};

#endif /* UKF_H */
