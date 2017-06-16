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
  // if this is true, the program will output files for NIS for laser and radar
  write_NIS_ = true;

  ///* if this is true, the program will print x_ and P_ to a file instead of screen
  write_state_to_file_ = true;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.2; // copied from last project for now

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2; // copied from last project for now

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
  x_ = VectorXd(5);

  is_initialized_ = false;

  time_us_ = 0;

  n_x_ = 5;

  n_aug_ = n_x_ + 2;

  lambda_ = 3 - n_aug_;

  // initializing R and H matrices for Laser and Radar
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << std_laspx_ * std_laspx_, 0,
              0, std_laspy_ * std_laspy_;

  H_laser_ = MatrixXd(2, 5);
  H_laser_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0;

  R_radar_ = MatrixXd(3, 3);
  R_radar_ << std_radr_ * std_radr_, 0, 0,
              0, std_radphi_ * std_radphi_, 0,
              0, 0, std_radrd_ * std_radrd_;

  // initial state covariance matrix P
  P_ = MatrixXd::Identity(5, 5);

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  weights_ = VectorXd(2 * n_aug_ + 1);

  // set weights
  // since every weight but the first have the same value
  // we can fill the whole weight vector with the constant
  // value and then substitute the value only on the first
  // entry.
  weights_.fill(1 / (2 * (lambda_ + n_aug_)));
  weights_(0) = lambda_ / (lambda_ + n_aug_);

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

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {

      // raw measurements from RADAR (range (rho), bearing (phi) and range rate (rhodot))
      // can be converted to cartesian coordinates using the equations:
      // px = cos(phi) * rho
      // py = sin(phi) * rho
      std::cout << "Initializing with Radar" << std::endl;
      x_ << std::cos(meas_package.raw_measurements_[1]) * meas_package.raw_measurements_[0],
            std::sin(meas_package.raw_measurements_[1]) * meas_package.raw_measurements_[0],
            0,
            0,
            0;

      // store previous timestamp value in (microseconds - us)
      time_us_ = meas_package.timestamp_ / 1000000.0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {

      // read raw measurements from LASER (only px and py in cartesian space)
      std::cout << "Initializing with LIDAR" << std::endl;
      x_ << meas_package.raw_measurements_[0],
            meas_package.raw_measurements_[1],
            0,
            0,
            0;

      // store previous timestamp value in (microseconds - us)
      time_us_ = meas_package.timestamp_ / 1000000.0;
    }

    // clean NIS_radar.csv and NIS_laser.csv if write_NIS_ is true
    if (write_NIS_ == true) {
      std::fstream f;
      f.open("../NIS_radar.csv", std::fstream::out | std::fstream::trunc);
      f << "NIS radar" << std::endl;
      f.close();

      f.open("../NIS_laser.csv", std::fstream::out | std::fstream::trunc);
      f << "NIS laser" << std::endl;
      f.close();
    }

    // clean state_output.csv if write_state_to_file_ is true
    if (write_state_to_file_ == true) {
      std::fstream f;
      f.open("../state_output.csv", std::fstream::out | std::fstream::trunc);
      f.close();
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;  
    return;
  }

  /*****************************************************************************
   *  Prediction                                                               *
   *****************************************************************************/

  // compute the time elapsed between the current and previous measurements
  double dt = (meas_package.timestamp_ / 1000000.0 - time_us_);	//dt - expressed in microseconds
  time_us_ = meas_package.timestamp_  / 1000000.0;

  // Predict our state
  Prediction(dt);

  /*****************************************************************************
   *  Update                                                                   *
   *****************************************************************************/

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    if (use_radar_ == true) UpdateRadar(meas_package.raw_measurements_);
  } else {
    // Laser updates
    if (use_laser_ == true) UpdateLidar(meas_package.raw_measurements_);
  }

  // print the output
  if (write_state_to_file_ == true) {
    std::fstream f;
    f.open("../state_output.csv", std::fstream::out | std::fstream::app);
    f << "x_" << endl;
    for (int j = 0; j < x_.rows(); j++) {
      f << x_(j) << endl;
    }
    f << "P_" << endl;
    for (int j = 0; j < P_.rows(); j++) {
      for (int k = 0; k < P_.cols(); k++) {
        f << P_(j, k) << ","; 
      }
      f << endl;
    }
    f.close();
  }
  else{
  cout << "x_ = " << x_ << endl;
  cout << "P_ = " << P_ << endl;
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
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(const VectorXd &z) {
  // Because this model is linear, we don't need to worry
  // about loss of precision and can use the same
  // method as in EKF.

  VectorXd y_ = z - (H_laser_ * x_);
  MatrixXd Ht = H_laser_.transpose();           // storing Ht matrix for efficiency
  MatrixXd S_ = H_laser_ * P_ * Ht + R_laser_;
  MatrixXd Si = S_.inverse();                   // storing Si matrix for efficiency
  MatrixXd K_ = P_ * Ht * Si;

  // New Estimate
  x_ = x_ + (K_ * y_);
  MatrixXd I_ = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I_ - K_ * H_laser_) * P_;

  // NIS calculation
  double NIS_laser_ = y_.transpose() * Si * y_;
  if (write_NIS_ == true) {
    std::fstream f;
    f.open("../NIS_laser.csv", std::fstream::out | ios::app);
    f << NIS_laser_ << std::endl;
    f.close();
  }
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(const VectorXd &z) {

  // set measurement dimension, radar can measure r, phi, and r_dot
  int n_z_ = 3;

  // create matrix for sigma points in measurement space
  MatrixXd Zsig_ = MatrixXd(n_z_, 2 * n_aug_ + 1);

  // transform sigma points into measurement space
  // iterate through all sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // store variables for easy understanding of formulas
    // can be supressed to save space in production.
    double px  = Xsig_pred_(0,i);
    double py  = Xsig_pred_(1,i);
    double v   = Xsig_pred_(2,i);
    double phi = Xsig_pred_(3,i);

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
  VectorXd z_pred_ = VectorXd(n_z_);
  z_pred_ = Zsig_ * weights_;

  //calculate measurement covariance matrix S
  Z_diff_ = Zsig_.colwise() - z_pred_;

  // Normalize angles on Z_diff
  for (int i = 0; i < Z_diff_.cols(); i++) {
    if (Z_diff_(1, i) >  M_PI) {
      //std::cout << "Angle before norm1: " << Z_diff_(1, i) << endl;
      Z_diff_(1, i) = (int(Z_diff_(1, i) - M_PI) % int(2. * M_PI) - M_PI);
      //std::cout << "Angle after norm1: " << Z_diff_(1, i) << endl;
    }
    if (Z_diff_(1, i) < -M_PI) {
      //std::cout << "Angle before norm1: " << Z_diff_(1, i) << endl;
      Z_diff_(1, i) = (int(Z_diff_(1, i) + M_PI) % int(2. * M_PI) - M_PI);
      //std::cout << "Angle after norm1: " << Z_diff_(1, i) << endl;
    }
   }

  // vectorized calculation of covariance matrix S
  MatrixXd S_ = MatrixXd(n_z_, n_z_);
  S_ = (Z_diff_.array().rowwise() * weights_.transpose().array()).matrix() * Z_diff_.transpose();

  // add measurement noise covariance matrix
  S_ = S_ + R_radar_;

  //create matrix for cross correlation Tc
  MatrixXd Tc_ = MatrixXd(n_x_, n_z_);

  //calculate cross correlation matrix
  // vectorized calculation of cross correlation matrix Tc
  X_diff_ = Xsig_pred_.colwise() - x_;
  Z_diff_ = Zsig_.colwise() - z_pred_;
  Tc_ = (X_diff_.array().rowwise() * weights_.transpose().array()).matrix() * Z_diff_.transpose();

  // calculate Kalman gain K;
  MatrixXd K_ = Tc_ * S_.inverse();

  // store residual
  VectorXd z_diff_ = z - z_pred_;

  // normalize angle
  if (z_diff_(1) >  M_PI) {
    //std::cout << "Angle before norm2: " << z_diff_(1) << endl;
    z_diff_(1) = (int(z_diff_(1) - M_PI) % int(2. * M_PI) - M_PI);
    //std::cout << "Angle after norm2: " << z_diff_(1) << endl;
  }
  if (z_diff_(1) < -M_PI) {
    //std::cout << "Angle before norm2: " << z_diff_(1) << endl;
    z_diff_(1) = (int(z_diff_(1) + M_PI) % int(2. * M_PI) - M_PI);
    //std::cout << "Angle after norm2: " << Z_diff_(1) << endl;
  }

  //update state mean and covariance matrix
  x_ += K_ * z_diff_;
  P_ -= K_ * S_ * K_.transpose();

  // NIS calculation
  double NIS_radar_ = z_diff_.transpose() * S_.inverse() * z_diff_;
  if (write_NIS_ == true) {
    std::fstream f;
    f.open("../NIS_radar.csv", std::fstream::out | ios::app);
    f << NIS_radar_ << std::endl;
    f.close();
  }
}

void UKF::PredictSigmaPoints(double delta_t) {
  // Create a matrix for the augmented sigma points
  MatrixXd Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // Call our augmented sigma points generator function
  GenerateAugmentedSigmaPoints(&Xsig_aug_);

  // store delta_t^2 to avoid multiple calcs
  double delta_t2 = delta_t * delta_t;

  // iterate through columns
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // store variables for easy understanding of formulas
    // can be supressed to save space in production.
    double px        = Xsig_aug_(0,i);
    double py        = Xsig_aug_(1,i);
    double v         = Xsig_aug_(2,i);
    double phi       = Xsig_aug_(3,i);
    double phi_d     = Xsig_aug_(4,i);
    double nu_a      = Xsig_aug_(5,i);
    double nu_phi_dd = Xsig_aug_(6,i);

    // declare new state variables
    double px_p;
    double py_p;

    // These values below are specific for our CTRV model
    // (constant turn rate and velocity )
    double v_p = v;
    double phi_p = phi + phi_d * delta_t;
    double phi_d_p = phi_d;

    // if phi dot is not zero (we are using 1e-4 as zero here)
    if (fabs(phi_d) > 0.001) {
      px_p = px + v / phi_d * (  sin(phi + phi_d * delta_t) - sin(phi));
      py_p = py + v / phi_d * (- cos(phi + phi_d * delta_t) + cos(phi));
    } // if phi dot is zero we use the simplified version
    else {
      px_p = px + v * delta_t * cos(phi);
      py_p = py + v * delta_t * sin(phi);
    }

    //add noise
    px_p = px_p + 0.5 * delta_t2 * cos(phi) * nu_a;
    py_p = py_p + 0.5 * delta_t2 * sin(phi) * nu_a;
    v_p = v_p + delta_t * nu_a;
    phi_p = phi_p + 0.5 * delta_t2 * nu_phi_dd;
    phi_d_p = phi_d_p + delta_t * nu_phi_dd;

    // pass calculated values to Xsig_pred
    Xsig_pred_.col(i) << px_p, py_p, v_p, phi_p, phi_d_p;
  }
}

void UKF::GenerateAugmentedSigmaPoints(MatrixXd* Xsig_aug_) {
  // Create augmented mean vector
  VectorXd x_aug_ = VectorXd(n_aug_);
  x_aug_.head(n_x_) = x_;
  x_aug_(n_x_)     = 0;
  x_aug_(n_x_ + 1) = 0;

  // Create augmented state covariances
  MatrixXd P_aug_ = MatrixXd(7, 7);
  P_aug_.fill(0.0);
  P_aug_.topLeftCorner(5, 5) = P_;
  P_aug_(5, 5) = std_a_ * std_a_;
  P_aug_(6, 6) = std_yawdd_ * std_yawdd_;

  //create square root matrix
  MatrixXd A_aug_ = P_aug_.llt().matrixL();

  // Create augmented sigma point matrix
  (*Xsig_aug_).col(0) = x_aug_;

  for (int i = 0; i < n_aug_; i++) {
    (*Xsig_aug_).col(i + 1)          = x_aug_ + sqrt(lambda_ + n_aug_) * A_aug_.col(i);
    (*Xsig_aug_).col(i + 1 + n_aug_) = x_aug_ - sqrt(lambda_ + n_aug_) * A_aug_.col(i);
  }
}


void UKF::PredictMeanAndCovariance() {

  // predict state mean
  // we use vectorization instead of loops for efficiency
  x_ = Xsig_pred_ * weights_;

  // predict state covariance matrix
  // we use vectorization instead of loops for efficiency
  X_diff_ = Xsig_pred_.colwise() - x_;

  // now we iterate through the angles to normalize between -pi and +pi
  for (int i = 0; i < X_diff_.cols(); i++) {
    if (X_diff_(3, i) >  M_PI) {
      //std::cout << "Angle before norm3: " << X_diff_(3, i) << endl;
      X_diff_(3, i) = (int(X_diff_(3, i) - M_PI) % int(2. * M_PI) - M_PI);
      //std::cout << "Angle after norm3: " << X_diff_(3, i) << endl;
    }
    if (X_diff_(3, i) < -M_PI) {
      //std::cout << "Angle before norm3: " << X_diff_(3, i) << endl;
      X_diff_(3, i) = (int(X_diff_(3, i) + M_PI) % int(2. * M_PI) - M_PI);
      //std::cout << "Angle after norm3: " << X_diff_(3, i) << endl;
    }
   }

  P_ = (X_diff_.array().rowwise() * weights_.transpose().array()).matrix() * X_diff_.transpose();

}
