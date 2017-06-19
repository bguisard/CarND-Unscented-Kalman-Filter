#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <fstream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is true, the program will output debug files
  write_debug_file_ = false;

  // if this is true, the program will output files for NIS for laser and radar
  write_NIS_ = false;

  ///* if this is true, the program will print x_ and P_ to a file instead of screen
  write_state_to_file_ = true;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // initial state vector
  x_ = VectorXd::Zero(5);

  is_initialized_ = false;

  time_us_ = 0;

  n_x_ = 5;

  n_aug_ = n_x_ + 2;

  lambda_ = 3 - n_aug_;

  // initializing R matrices for Laser and Radar
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << std_laspx_ * std_laspx_, 0,
              0, std_laspy_ * std_laspy_;

  R_radar_ = MatrixXd(3, 3);
  R_radar_ << std_radr_ * std_radr_, 0, 0,
              0, std_radphi_ * std_radphi_, 0,
              0, 0, std_radrd_ * std_radrd_;

  // initial state covariance matrix P
  P_ = MatrixXd::Identity(5, 5) * 0.15;

  Xsig_pred_ = MatrixXd::Zero(n_x_, 2 * n_aug_ + 1);

  weights_ = VectorXd(2 * n_aug_ + 1);

  // set weights
  // since every weight but the first have the same value
  // we can fill the whole weight vector with the constant
  // value and then substitute the value only on the first
  // entry.
  weights_.fill(1 / (2 * (lambda_ + n_aug_)));
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  // counter of timesteps
  k_iteration = 0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {


  /*****************************************************************************
   *  Initialization                                                           *
   *****************************************************************************/

  if (!is_initialized_) {

    // first measurement
    cout << "UKF: " << endl;

    // timestep counting
    k_iteration = 0;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {

      // raw measurements from RADAR (range (rho), bearing (phi) and range rate (rhodot))
      // can be converted to cartesian coordinates using the equations:
      // px = cos(phi) * rho
      // py = sin(phi) * rho
      std::cout << "Initializing with Radar" << std::endl;
      x_ << std::cos(meas_package.raw_measurements_[1]) * meas_package.raw_measurements_[0],
            std::sin(meas_package.raw_measurements_[1]) * meas_package.raw_measurements_[0],
            0.0001,
            0.0001,
            0.0001;

      // store previous timestamp value in (microseconds - us)
      time_us_ = meas_package.timestamp_;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {

      // read raw measurements from LASER (only px and py in cartesian space)
      std::cout << "Initializing with LIDAR" << std::endl;
      x_ << meas_package.raw_measurements_[0],
            meas_package.raw_measurements_[1],
            0.0001,
            0.0001,
            0.0001;

      // store previous timestamp value in (microseconds - us)
      time_us_ = meas_package.timestamp_;
    }

    // clean NIS_radar.csv and NIS_laser.csv if write_NIS_ is true
    if (write_NIS_) {
      std::fstream f;

      if (use_radar_) {
        f.open("../NIS_radar.csv", std::fstream::out | std::fstream::trunc);
        f << "NIS radar" << std::endl;
        f.close();
      }

      if (use_laser_) {
        f.open("../NIS_laser.csv", std::fstream::out | std::fstream::trunc);
        f << "NIS laser" << std::endl;
        f.close();
      }

    }

    // clean state_output.csv if write_state_to_file_ is true
    if (write_state_to_file_) {
      std::fstream f;
      f.open("../state_output.csv", std::fstream::out | std::fstream::trunc);
      f.close();
    }

    // clean and start debug.csv if write_debug_file_ is true
    if (write_debug_file_) {
      std::fstream f;
      f.open("../debug.csv", std::fstream::out | std::fstream::trunc);

      f << "Initialization" <<endl;
      f << "\nx_," << k_iteration << endl << x_ << endl;
      f << "\nP_," << k_iteration << endl << P_ << endl;
      f << "\nweights_," << k_iteration << endl << weights_ << endl;

      f.close();
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;  
    return;
  }

  /*****************************************************************************
   *  Prediction                                                               *
   *****************************************************************************/

  if ((meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) ||
      (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)) {

    // compute the time elapsed between the current and previous measurements
    double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in microseconds
    time_us_ = meas_package.timestamp_;

    // Predict our state
    Prediction(dt);

    if (write_debug_file_) {
      std::fstream f;
      f.open("../debug.csv", std::fstream::out | std::fstream::app);
      f << "\nDelta t: " << dt << endl;
      f.close();
    }

  }

  /*****************************************************************************
   *  Update                                                                   *
   *****************************************************************************/

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    if (use_radar_) UpdateRadar(meas_package.raw_measurements_);
  } else {
    // Laser updates
    if (use_laser_) UpdateLidar(meas_package.raw_measurements_);
  }

  /*****************************************************************************
   *  Output                                                                   *
   *****************************************************************************/

  // count last iteration
  k_iteration++;

  // print the output
  if (write_state_to_file_) {
    std::fstream f;
    f.open("../state_output.csv", std::fstream::out | std::fstream::app);
    f << "x_," << k_iteration;
    for (int j = 0; j < x_.rows(); j++) {
      f << "," << x_(j);
    }
    f << endl;
    f << "P_," << k_iteration;
    for (int j = 0; j < P_.rows(); j++) {
      for (int k = 0; k < P_.cols(); k++) {
        f << "," << P_(j, k); 
      }
    }
    f << endl;
    f.close();
  }
  else{
  cout << "x_ = " << x_ << endl;
  cout << "P_ = " << P_ << endl;
  }

  // debugging prints
  if (write_debug_file_) {
    std::fstream f;
    f.open("../debug.csv", std::fstream::out | std::fstream::app);
    f << "\nx_," << k_iteration << endl << x_ << endl;
    f << "\nP_," << k_iteration << endl << P_ << endl; 
    f.close();
  }

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

  // First we predict the sigma points on T + delta_t
  PredictSigmaPoints(delta_t);

  // then we predict the mean and covariance of our new state in T + delta_t
  PredictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package.raw_measurements_
 */
void UKF::UpdateLidar(const VectorXd &z) {

  // set measurement dimension, laser can measure px and py
  int n_z_ = 2;

  // create matrix for sigma points in measurement space
  MatrixXd Zsig_ = MatrixXd::Zero(n_z_, 2 * n_aug_ + 1);

  // transform sigma points into measurement space
  // iterate through all sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // store variables for easy understanding of formulas
    Zsig_(0, i)  = Xsig_pred_(0, i);
    Zsig_(1, i)  = Xsig_pred_(1, i);
  }

  //calculate mean predicted measurement
  VectorXd z_pred_ = VectorXd::Zero(n_z_);
  z_pred_ = Zsig_ * weights_;

  // create covariance matrix S
  MatrixXd S_ = MatrixXd::Zero(n_z_, n_z_);

  // create matrix for cross correlation Tc
  MatrixXd Tc_ = MatrixXd::Zero(n_x_, n_z_);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {

    // residuals
    VectorXd z_diff_ = Zsig_.col(i) - z_pred_;
    z_diff_(1) = NormalizeAngle(z_diff_(1));

    S_ += weights_(i) * z_diff_ * z_diff_.transpose();

    // state diff
    VectorXd x_diff_ = Xsig_pred_.col(i) - x_;
    x_diff_(3) = NormalizeAngle(x_diff_(3));

    Tc_ += weights_(i) * x_diff_ * z_diff_.transpose();
  
  }

  // debugging prints
  if (write_debug_file_) {
    std::fstream f;
    f.open("../debug.csv", std::fstream::out | std::fstream::app);
    f << "\nDebugging steps for  UPDATE LASER.";
    f << "\nz: " << z << endl;
    f << "\nZsig_," << k_iteration << endl << Zsig_ << endl;
    f << "\nz_pred_," << k_iteration << endl << z_pred_ << endl;
    f << "\nS_," << k_iteration << endl << S_ << endl;
    f << "\nTc_," << k_iteration << endl << Tc_ << endl;
    f.close();
  }

  // add measurement noise covariance matrix
  S_ += R_laser_;

  // calculate Kalman gain K;
  MatrixXd K_ = Tc_ * S_.inverse();

  // store residual
  VectorXd y_ = z - z_pred_;
  //y_(1) = NormalizeAngle(y_(1));

  // debugging prints
  if (write_debug_file_) {
    std::fstream f;
    f.open("../debug.csv", std::fstream::out | std::fstream::app);
    f << "\nS_ with noise," << k_iteration << endl << S_ << endl;
    f << "\nK_," << k_iteration << endl << K_ << endl;
    f << "\ny_," << k_iteration << endl << y_ << endl;
    f.close();
  }

  //update state mean and covariance matrix
  x_ += K_ * y_;
  P_ -= K_ * S_ * K_.transpose();

  // NIS calculation
  double NIS_laser_ = y_.transpose() * S_.inverse() * y_;
  if (write_NIS_) {
    std::fstream f;
    f.open("../NIS_laser.csv", std::fstream::out | ios::app);
    f << NIS_laser_ << std::endl;
    f.close();
  }
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package.raw_measurements_
 */
void UKF::UpdateRadar(const VectorXd &z) {

  // set measurement dimension, radar can measure r, phi, and r_dot
  int n_z_ = 3;

  // create matrix for sigma points in measurement space
  MatrixXd Zsig_ = MatrixXd::Zero(n_z_, 2 * n_aug_ + 1);

  // transform sigma points into measurement space
  // iterate through all sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // store variables for easy understanding of formulas
    // can be supressed to save space in production.
    double px  = Xsig_pred_(0, i);
    double py  = Xsig_pred_(1, i);
    double v   = Xsig_pred_(2, i);
    double phi = Xsig_pred_(3, i);

    // calculate vx and vy
    double vx = cos(phi) * v;
    double vy = sin(phi) * v;

    // store values that will be reused for efficiency
    double divisor = sqrt(px*px + py*py);

    // measurement model for radar
    Zsig_(0, i) = divisor;
    Zsig_(1, i) = atan2(py, px);
    Zsig_(2, i) = (px*vx + py*vy) / divisor;

  }

  //calculate mean predicted measurement
  VectorXd z_pred_ = VectorXd::Zero(n_z_);
  z_pred_ = Zsig_ * weights_;

  // covariance matrix S
  MatrixXd S_ = MatrixXd::Zero(n_z_, n_z_);

  //create matrix for cross correlation Tc
  MatrixXd Tc_ = MatrixXd::Zero(n_x_, n_z_);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd z_diff_ = Zsig_.col(i) - z_pred_;
    z_diff_(1) = NormalizeAngle(z_diff_(1));

    S_ += weights_(i) * z_diff_ * z_diff_.transpose();

    VectorXd x_diff_ = Xsig_pred_.col(i) - x_;
    x_diff_(3) = NormalizeAngle(x_diff_(3));

  //calculate cross correlation matrix
    Tc_ += weights_(i) * x_diff_ * z_diff_.transpose();
  }

  // debugging prints
  if (write_debug_file_) {
    std::fstream f;
    f.open("../debug.csv", std::fstream::out | std::fstream::app);
    f << "\nDebugging steps for  UPDATE RADAR.";
    f << "\nz: " << z << endl;
    f << "\nZsig_," << k_iteration << endl << Zsig_ << endl;
    f << "\nz_pred_," << k_iteration << endl << z_pred_ << endl;
    f << "\nS_," << k_iteration << endl << S_ << endl;
    f << "\nTc_," << k_iteration << endl << Tc_ << endl;
    f.close();
  }

  // add measurement noise covariance matrix
  S_ += R_radar_;

  // calculate Kalman gain K;
  MatrixXd K_ = Tc_ * S_.inverse();

  // store residual
  VectorXd y_ = z - z_pred_;

  // normalize angle
  y_(1) = NormalizeAngle(y_(1));

  // debugging prints
  if (write_debug_file_) {
    std::fstream f;
    f.open("../debug.csv", std::fstream::out | std::fstream::app);
    f << "\nS_ with noise," << k_iteration << endl << S_ << endl;
    f << "\nK_," << k_iteration << endl << K_ << endl;
    f << "\ny_," << k_iteration << endl << y_ << endl;
    f.close();
  }

  //update state mean and covariance matrix
  x_ += K_ * y_;
  P_ -= K_ * S_ * K_.transpose();

  // NIS calculation
  double NIS_radar_ = y_.transpose() * S_.inverse() * y_;
  if (write_NIS_) {
    std::fstream f;
    f.open("../NIS_radar.csv", std::fstream::out | ios::app);
    f << NIS_radar_ << std::endl;
    f.close();
  }
}

/**
 * Predicts Augmented Sigma points for a given delta_t
 * @param delta_t
 */
void UKF::PredictSigmaPoints(double delta_t) {
  // Create a matrix for the augmented sigma points
  MatrixXd Xsig_aug_ = MatrixXd::Zero(n_aug_, 2 * n_aug_ + 1);

  // Call our augmented sigma points generator function
  GenerateAugmentedSigmaPoints(&Xsig_aug_);

  // store delta_t^2 to avoid multiple calcs
  double delta_t2 = delta_t * delta_t;

  // debugging prints
  if (write_debug_file_) {
    std::fstream f;
    f.open("../debug.csv", std::fstream::out | std::fstream::app);
    f << "\nXsig_aug_: " << endl << Xsig_aug_ << endl;
    f << "\n\nDebugging steps of PREDICT SIGMA POINTS." << endl;
    f << "\nDelta t: " << delta_t << endl;
    f << "\nDelta t2: " << delta_t2 << endl;
    f.close();
  }

  // iterate through columns
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // store variables for easy understanding of formulas
    // can be supressed to save space in production.
    double px        = Xsig_aug_(0, i);
    double py        = Xsig_aug_(1, i);
    double v         = Xsig_aug_(2, i);
    double phi       = Xsig_aug_(3, i);
    double phi_d     = Xsig_aug_(4, i);
    double nu_a      = Xsig_aug_(5, i);
    double nu_phi_dd = Xsig_aug_(6, i);

    // declare new predicted state variables
    double px_p;
    double py_p;

    // These values below are specific for our CTRV model
    // (constant turn rate and velocity )
    double v_p = v;
    double phi_p = (phi + (phi_d * delta_t));
    double phi_d_p = phi_d;

    //phi_p = NormalizeAngle(phi_p);

    // if phi dot is not zero (we are using 1e-4 as zero here)
    if (fabs(phi_d) > 0.001) {
      px_p = px + (v / phi_d) * (sin(phi_p) - sin(phi));
      py_p = py + (v / phi_d) * (cos(phi) - cos(phi_p));
    } // if phi dot is zero we use the simplified version
    else {
      px_p = px + (v * cos(phi) * delta_t);
      py_p = py + (v * sin(phi) * delta_t);
    }

    //add noise
    px_p += 0.5 * (delta_t2 * cos(phi) * nu_a);
    py_p += 0.5 * (delta_t2 * sin(phi) * nu_a);
    v_p += (delta_t * nu_a);
    phi_p += (0.5 * delta_t2 * nu_phi_dd);
    phi_d_p += (delta_t * nu_phi_dd);

    //phi_p = NormalizeAngle(phi_p);

    // pass calculated values to Xsig_pred
    Xsig_pred_.col(i) << px_p, py_p, v_p, phi_p, phi_d_p;
  }

  // debugging prints
  if (write_debug_file_) {
    std::fstream f;
    f.open("../debug.csv", std::fstream::out | std::fstream::app);
    f << "\nXsig_pred_: " << endl << Xsig_pred_ << endl;
    f.close();
  }
}

/**
 * Generates matrix of augmented Sigma points
 * @param (pointer to Xsig_aug)
 */
void UKF::GenerateAugmentedSigmaPoints(MatrixXd* Xsig_aug_) {
  // Create augmented mean vector
  VectorXd x_aug_ = VectorXd(n_aug_);
  x_aug_.head(5) = x_;
  x_aug_(5)     = 0;
  x_aug_(6) = 0;

  // Create augmented state covariances
  MatrixXd P_aug_ = MatrixXd::Zero(7, 7);
  P_aug_.topLeftCorner(5, 5) = P_;
  P_aug_(5, 5) = std_a_ * std_a_;
  P_aug_(6, 6) = std_yawdd_ * std_yawdd_;

  //create square root matrix
  MatrixXd A_aug_ = P_aug_.llt().matrixL();

    // debugging prints
  if (write_debug_file_) {
    std::fstream f;
    f.open("../debug.csv", std::fstream::out | std::fstream::app);
    f << "\n\nDebugging steps of GENERATE AUGMENTED SIGMA POINTS." << endl;
    f << "\nP_aug_: \n" << P_aug_ << endl;
    f << "\nA_aug_: \n" << A_aug_ << endl;
    f.close();
  }

  // Create augmented sigma point matrix
  (*Xsig_aug_).col(0) = x_aug_;

  for (int i = 0; i < n_aug_; i++) {
    (*Xsig_aug_).col(i + 1)          = x_aug_ + sqrt(lambda_ + n_aug_) * A_aug_.col(i);
    (*Xsig_aug_).col(i + 1 + n_aug_) = x_aug_ - sqrt(lambda_ + n_aug_) * A_aug_.col(i);
  }
}

/**
 * Predicts State Mean and Covariance
 */
void UKF::PredictMeanAndCovariance() {

  // predict state mean
  // we use vectorization instead of loops for efficiency
  x_ = Xsig_pred_ * weights_;

  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    x_diff(3) = NormalizeAngle(x_diff(3));

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }

    // debugging prints
  if (write_debug_file_) {
    std::fstream f;
    f.open("../debug.csv", std::fstream::out | std::fstream::app);
    f << "\n\nDebugging steps of PREDICT MEAN AND COVARIANCE." << endl;
    f << "\nx_: \n" << x_ << endl;
    f << "\nP_: \n" << P_ << endl;
    f.close();
  }

}

/**
 * Normalizes angles between -PI and +PI
 * @param (double angle)
 */
double UKF::NormalizeAngle(double angle) {
  double n_angle;

  if (fabs(angle) > M_PI) {
    n_angle = angle - round(angle / (2.0d * M_PI)) * (2.0d * M_PI);

    // debugging prints
    if (write_debug_file_) {
      std::fstream f;
      f.open("../debug.csv", std::fstream::out | std::fstream::app);
      f << "\nAngle before normalizing: " << angle << endl;
      f << "\nAngle after normalizing: " << n_angle << endl;
      f.close();
    }
  } else {
    return angle;
  }

  return n_angle;
}