/******************************************************************************
MIT License

Copyright (c) 2022 Jihang Li (jihangli AT duck DOT com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Last update: 2022-03-21 10:16
Version: V0.1.0
******************************************************************************/
#ifndef NEU_CV_H
#define NEU_CV_H

#include <cmath>

#if defined(NEUCVH_USE_EIGEN)
#include "Eigen/Dense"
#endif

#if defined(NEUCVH_USE_OPENCV)
#include "opencv2/opencv.hpp"
#include "opencv2/video/tracking.hpp"
#endif


//************************************************ NKalmanFilter Declaration{{{
template <class T>
class NKalmanFilter
{
  static_assert(std::is_class<T>::value, "Non-class type is not supported yet.");

public:
  virtual ~NKalmanFilter() {}

  virtual T Correct(const T &measurement) = 0;

  virtual T Filter(const T &measurement, const T &control = T()) = 0;

  virtual T Peek(const unsigned int &steps, const T &control = T()) = 0;

  virtual T Predict(const T &control = T()) = 0;

  virtual void Reset(
      const int &state_dimension, const int &measurement_dimension,
      const int &control_dimension, const float &q_scale = std::nanf(""),
      const float &r_scale = std::nanf("")) = 0;

  virtual T GetControlMatrix() const = 0;

  virtual T GetMeasurementMatrix() const = 0;

  virtual T GetMeasurementNoise() const = 0;

  virtual T GetProcessNoise() const = 0;

  virtual T GetTransitionMatrix() const = 0;

  virtual void SetControlMatrix(const T &B) = 0;

  virtual void SetMeasurementMatrix(const T &H) = 0;

  virtual void SetMeasurementNoise(const T &R) = 0;

  virtual void SetProcessNoise(const T &Q) = 0;

  virtual void SetTransitionMatrix(const T &A) = 0;
};  // class NKalmanFilter

#ifdef NEUCVH_USE_EIGEN
template <class T>
class NKalmanFilterEigen: public NKalmanFilter<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
{
  static_assert(
      std::is_same<T, float>::value || std::is_same<T, double>::value,
      "This template only accepts float or double.");

  using MatrixXT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

public:
  NKalmanFilterEigen(
      const int &state_dimension, const int &measurement_dimension,
      const int &control_dimension, const float &r_scale = 1.0f,
      const float &q_scale = 1.0f);

  ~NKalmanFilterEigen();

  MatrixXT Correct(const MatrixXT &measurement);

  MatrixXT Filter(
      const MatrixXT &measurement, const MatrixXT &control = MatrixXT());

  MatrixXT Peek(const unsigned int &steps, const MatrixXT &control = MatrixXT());

  MatrixXT Predict(const MatrixXT &control = MatrixXT());

  void Reset(
      const int &state_dimension, const int &measurement_dimension,
      const int &control_dimension, const float &r_scale = std::nanf(""),
      const float &q_scale = std::nanf(""));

  MatrixXT GetControlMatrix() const;

  MatrixXT GetMeasurementMatrix() const;

  MatrixXT GetMeasurementNoise() const;

  MatrixXT GetProcessNoise() const;

  MatrixXT GetTransitionMatrix() const;

  void SetControlMatrix(const MatrixXT &control);

  void SetMeasurementMatrix(const MatrixXT &H);

  void SetMeasurementNoise(const MatrixXT &R);

  void SetProcessNoise(const MatrixXT &Q);

  void SetTransitionMatrix(const MatrixXT &transition);

private:
  void MeasurementToState(const MatrixXT &measurement, MatrixXT &state);

  MatrixXT m_transition;         // A
  MatrixXT m_control;            // B
  MatrixXT m_measurement;        // H
  MatrixXT m_measurement_noise;  // R
  MatrixXT m_process_noise;      // Q
  MatrixXT m_gain;               // K
  MatrixXT m_state_post;
  MatrixXT m_state_pre;
  MatrixXT m_error_post;
  MatrixXT m_error_pre;
  MatrixXT m_temp;
  bool m_initialized;
  float m_r_scale;
  float m_q_scale;
};  // class NKalmanFilterEigen
#endif  // ifdef NEUCVH_USE_EIGEN

#ifdef NEUCVH_USE_OPENCV
class NKalmanFilterOpenCV: public NKalmanFilter<cv::Mat>
{
public:
  NKalmanFilterOpenCV(
      const int &state_dimension, const int &measurement_dimension,
      const int &control_dimension, const float &r_scale = 1.0f,
      const float &q_scale = 1.0f);

  ~NKalmanFilterOpenCV();

  cv::Mat Correct(const cv::Mat &measurement);

  cv::Mat Filter(
      const cv::Mat &measurement, const cv::Mat &control = cv::Mat());

  cv::Mat Predict(const cv::Mat &control = cv::Mat());

  cv::Mat Peek(const unsigned int &steps, const cv::Mat &control = cv::Mat());

  void Reset(
      const int &state_dimension, const int &measurement_dimension,
      const int &control_dimension, const float &r_scale = std::nanf(""),
      const float &q_scale = std::nanf(""));

  cv::Mat GetControlMatrix() const;

  cv::Mat GetMeasurementMatrix() const;

  cv::Mat GetMeasurementNoise() const;

  cv::Mat GetProcessNoise() const;

  cv::Mat GetTransitionMatrix() const;

  void SetControlMatrix(const cv::Mat &control);

  void SetMeasurementMatrix(const cv::Mat &H);

  void SetMeasurementNoise(const cv::Mat &R);

  void SetProcessNoise(const cv::Mat &Q);

  void SetTransitionMatrix(const cv::Mat &transition);

private:
  void MeasurementToState(const cv::Mat &measurement, cv::Mat &state);

  cv::KalmanFilter m_kf;
  bool m_initialized;
  float m_r_scale;
  float m_q_scale;
};  // class NKalmanFilterOpenCV
#endif  // ifdef NEUCVH_USE_OPENCV
//************************************************ NKalmanFilter Declaration}}}

//********************************************* NKalmanFilter Implementation{{{
#ifdef NEUCVH_USE_EIGEN
template <class T>
NKalmanFilterEigen<T>::NKalmanFilterEigen(
    const int &state_dimension, const int &measurement_dimension,
    const int &control_dimension, const float &r_scale, const float &q_scale)
  :m_initialized(false), m_r_scale(1.0f), m_q_scale(1.0f)
{
  this->m_transition = MatrixXT();
  this->m_control = MatrixXT();
  this->m_measurement = MatrixXT();
  this->m_measurement_noise = MatrixXT();
  this->m_process_noise = MatrixXT();
  this->m_gain = MatrixXT();
  this->m_state_post = MatrixXT();
  this->m_state_pre = MatrixXT();
  this->m_error_post = MatrixXT();
  this->m_error_pre = MatrixXT();
  this->m_temp = MatrixXT();

  this->Reset(
      state_dimension, measurement_dimension, control_dimension,
      r_scale, q_scale);
}

template <class T>
NKalmanFilterEigen<T>::~NKalmanFilterEigen()
{
}

template <class T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
NKalmanFilterEigen<T>::Correct(const MatrixXT &measurement)
{
  this->m_temp = this->m_error_pre * this->m_measurement.transpose();

  this->m_gain =
    this->m_temp
    * (this->m_measurement * this->m_temp + this->m_measurement_noise).inverse();

  this->m_state_post =
    this->m_state_pre
    + this->m_gain * (measurement - this->m_measurement * this->m_state_pre);

  this->m_error_post = this->m_error_pre - this->m_gain * this->m_temp.transpose();

  return this->m_state_post;
}

template <class T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
NKalmanFilterEigen<T>::Filter(
    const MatrixXT &measurement, const MatrixXT &control)
{
  if (!this->m_initialized)
  {
    this->MeasurementToState(measurement, this->m_state_post);
    this->m_initialized = true;
    return measurement;
  }

  this->Predict(control);
  return this->Correct(measurement);
}

template <class T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
NKalmanFilterEigen<T>::Predict(const MatrixXT &control)
{
  this->m_state_pre = this->m_transition * this->m_state_post;

  if (control.rows() > 0 && control.cols() > 0)
  {
    this->m_state_pre += this->m_control * control;
  }

  this->m_error_pre =
    this->m_transition * this->m_error_post * this->m_transition.transpose()
    + this->m_process_noise;

  this->m_state_post = this->m_state_pre;
  this->m_error_post = this->m_error_pre;

  return this->m_state_pre;
}

template <class T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
NKalmanFilterEigen<T>::Peek(const unsigned int &steps, const MatrixXT &control)
{
  NKalmanFilterEigen<T> kf = *this;

  for (unsigned int i = 0; i < steps; ++i)
  {
    kf.Predict(control);
  }

  return kf.m_state_pre;
}

template <class T>
void NKalmanFilterEigen<T>::Reset(
    const int &state_dimension, const int &measurement_dimension,
    const int &control_dimension, const float &r_scale, const float &q_scale)
{
  if (state_dimension <= 0 || measurement_dimension <= 0)
  {
    fprintf(stderr, " Dimensions of state and measurement must be positive.\n");
    exit(EXIT_FAILURE);
  }

  this->m_initialized = false;

  // A
  if (this->m_transition.rows() != state_dimension
      || this->m_transition.cols() != state_dimension)
  {
    this->m_transition.resize(state_dimension, state_dimension);
  }
  this->m_transition.setZero();
  // NOTE: The function 'setIndentity' is invalid for dynamic size matrix.
  for (int i = 0; i < state_dimension; ++i)
  {
    this->m_transition(i, i) = (T)1.0;
  }

  // B
  if (this->m_control.rows() != state_dimension
      || this->m_control.cols() != control_dimension)
  {
    this->m_control.resize(state_dimension, control_dimension);
  }
  this->m_control.setZero();

  // H
  if (this->m_measurement.rows() != measurement_dimension
      || this->m_measurement.cols() != state_dimension)
  {
    this->m_measurement.resize(measurement_dimension, state_dimension);
  }
  this->m_measurement.setZero();
  // NOTE: The function 'setIndentity' is invalid for dynamic size matrix.
  for (int i = 0; i < std::min(measurement_dimension, state_dimension); ++i)
  {
    this->m_measurement(i, i) = (T)1.0;
  }

  // R
  if (this->m_measurement_noise.rows() != measurement_dimension
      || this->m_measurement_noise.cols() != measurement_dimension)
  {
    this->m_measurement_noise.resize(measurement_dimension, measurement_dimension);
  }
  this->m_measurement_noise.setZero();
  // NOTE: The function 'setIndentity' is invalid for dynamic size matrix.
  for (int i = 0; i < measurement_dimension; ++i)
  {
    this->m_measurement_noise(i, i) = (T)1.0;
  }

  // Q
  if (this->m_process_noise.rows() != state_dimension
      || this->m_process_noise.cols() != state_dimension)
  {
    this->m_process_noise.resize(state_dimension, state_dimension);
  }
  this->m_process_noise.setZero();
  // NOTE: The function 'setIndentity' is invalid for dynamic size matrix.
  for (int i = 0; i < this->m_process_noise.rows(); ++i)
  {
    this->m_process_noise(i, i) = (T)1.0;
  }

  // K
  if (this->m_gain.rows() != state_dimension
      || this->m_gain.cols() != measurement_dimension)
  {
    this->m_gain.resize(state_dimension, measurement_dimension);
  }
  this->m_gain.setZero();

  if (this->m_state_post.rows() != state_dimension
      || this->m_state_post.cols() != 1)
  {
    this->m_state_post.resize(state_dimension, 1);
  }
  this->m_state_post.setZero();

  if (this->m_state_pre.rows() != state_dimension
      || this->m_state_pre.cols() != 1)
  {
    this->m_state_pre.resize(state_dimension, 1);
  }
  this->m_state_pre.setZero();

  if (this->m_error_post.rows() != state_dimension
      || this->m_error_post.cols() != state_dimension)
  {
    this->m_error_post.resize(state_dimension, state_dimension);
  }
  this->m_error_post.setZero();

  if (this->m_error_pre.rows() != state_dimension
      || this->m_error_pre.cols() != state_dimension)
  {
    this->m_error_pre.resize(state_dimension, state_dimension);
  }
  this->m_error_pre.setZero();

  if (this->m_temp.rows() != state_dimension
      || this->m_temp.cols() != measurement_dimension)
  {
    this->m_temp.resize(state_dimension, measurement_dimension);
  }
  this->m_temp.setZero();

  if (!std::isnan(r_scale))
  {
    this->m_r_scale = r_scale;
  }

  if (!std::isnan(q_scale))
  {
    this->m_q_scale = q_scale;
  }

  this->m_measurement_noise *= this->m_r_scale;
  this->m_process_noise *= this->m_q_scale;
}

template <class T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
NKalmanFilterEigen<T>::GetControlMatrix() const
{
  return this->m_control;
}

template <class T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
NKalmanFilterEigen<T>::GetMeasurementMatrix() const
{
  return this->m_measurement;
}

template <class T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
NKalmanFilterEigen<T>::GetMeasurementNoise() const
{
  return this->m_measurement_noise;
}

template <class T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
NKalmanFilterEigen<T>::GetProcessNoise() const
{
  return this->m_process_noise;
}

template <class T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
NKalmanFilterEigen<T>::GetTransitionMatrix() const
{
  return this->m_transition;
}

template <class T>
void NKalmanFilterEigen<T>::SetControlMatrix(const MatrixXT &B)
{
  this->m_control = std::move(B);
}

template <class T>
void NKalmanFilterEigen<T>::SetMeasurementMatrix(const MatrixXT &H)
{
  this->m_measurement = std::move(H);
}

template <class T>
void NKalmanFilterEigen<T>::SetMeasurementNoise(const MatrixXT &R)
{
  this->m_measurement_noise = std::move(R);
}

template <class T>
void NKalmanFilterEigen<T>::SetProcessNoise(const MatrixXT &Q)
{
  this->m_process_noise = std::move(Q);
}

template <class T>
void NKalmanFilterEigen<T>::SetTransitionMatrix(const MatrixXT &A)
{
  this->m_transition = std::move(A);
}

template <class T>
void NKalmanFilterEigen<T>::MeasurementToState(
    const MatrixXT &measurement, MatrixXT &state)
{
  Eigen::Index mr = measurement.rows();
  Eigen::Index mc = measurement.cols();
  Eigen::Index sr = state.rows();
  Eigen::Index sc = state.cols();

  if (mr > sr || mc > sc)
  {
    fprintf(
        stderr,
        "Measurement (%td x %td) has a bigger size than state (%td x %td).",
        mr, mc, sr, sc);
    exit(EXIT_FAILURE);
  }

  state.block(0, 0, mr, mc) = measurement;
}
#endif  // ifdef NEUCVH_USE_EIGEN

#ifdef NEUCVH_USE_OPENCV
inline NKalmanFilterOpenCV::NKalmanFilterOpenCV(
    const int &state_dimension, const int &measurement_dimension,
    const int &control_dimension, const float &r_scale, const float &q_scale)
  :m_initialized(false), m_r_scale(1.0f), m_q_scale(1.0f)
{
  this->Reset(
      state_dimension, measurement_dimension, control_dimension,
      r_scale, q_scale);
}

inline NKalmanFilterOpenCV::~NKalmanFilterOpenCV()
{
}

inline cv::Mat NKalmanFilterOpenCV::Correct(const cv::Mat &measurement)
{
  this->m_kf.correct(measurement);
  return this->m_kf.statePost;
}

inline cv::Mat NKalmanFilterOpenCV::Filter(
    const cv::Mat &measurement, const cv::Mat &control)
{
  if (!this->m_initialized)
  {
    this->MeasurementToState(measurement, this->m_kf.statePost);
    this->m_initialized = true;
    return measurement;
  }

  this->Predict(control);
  return this->Correct(measurement);
}

inline cv::Mat NKalmanFilterOpenCV::Predict(const cv::Mat &control)
{
  this->m_kf.predict(control);
  return this->m_kf.statePre;
}

inline cv::Mat NKalmanFilterOpenCV::Peek(
    const unsigned int &steps, const cv::Mat &control)
{
  cv::KalmanFilter kf = this->m_kf;

  for (int i = 0; i < steps; ++i)
  {
    kf.predict(control);
  }

  return kf.statePre;
}

inline void NKalmanFilterOpenCV::Reset(
    const int &state_dimension, const int &measurement_dimension,
    const int &control_dimension, const float &r_scale, const float &q_scale)
{
  if (state_dimension <= 0 || measurement_dimension <= 0)
  {
    fprintf(stderr, " Dimensions of state and measurement must be positive.\n");
    exit(EXIT_FAILURE);
  }

  this->m_initialized = false;

  this->m_kf.init(state_dimension, measurement_dimension, control_dimension);

  this->m_kf.measurementMatrix =
    cv::Mat::eye(measurement_dimension, state_dimension, CV_32F);

  if (!std::isnan(r_scale))
  {
    this->m_r_scale = r_scale;
  }

  if (!std::isnan(q_scale))
  {
    this->m_q_scale = q_scale;
  }

  this->m_kf.measurementNoiseCov *= this->m_r_scale;
  this->m_kf.processNoiseCov *= this->m_q_scale;
}

inline cv::Mat NKalmanFilterOpenCV::GetControlMatrix() const
{
  return this->m_kf.controlMatrix;
}

inline cv::Mat NKalmanFilterOpenCV::GetMeasurementMatrix() const
{
  return this->m_kf.measurementMatrix;
}

inline cv::Mat NKalmanFilterOpenCV::GetMeasurementNoise() const
{
  return this->m_kf.measurementNoiseCov;
}

inline cv::Mat NKalmanFilterOpenCV::GetProcessNoise() const
{
  return this->m_kf.processNoiseCov;
}

inline cv::Mat NKalmanFilterOpenCV::GetTransitionMatrix() const
{
  return this->m_kf.transitionMatrix;
}

inline void NKalmanFilterOpenCV::SetControlMatrix(const cv::Mat &B)
{
  this->m_kf.controlMatrix = std::move(B);
}

inline void NKalmanFilterOpenCV::SetMeasurementMatrix(const cv::Mat &H)
{
  this->m_kf.measurementMatrix = std::move(H);
}

inline void NKalmanFilterOpenCV::SetMeasurementNoise(const cv::Mat &R)
{
  this->m_kf.measurementNoiseCov = std::move(R);
}

inline void NKalmanFilterOpenCV::SetProcessNoise(const cv::Mat &Q)
{
  this->m_kf.processNoiseCov = std::move(Q);
}

inline void NKalmanFilterOpenCV::SetTransitionMatrix(const cv::Mat &A)
{
  this->m_kf.transitionMatrix = std::move(A);
}

inline void NKalmanFilterOpenCV::MeasurementToState(
    const cv::Mat &measurement, cv::Mat &state)
{
  int mr = measurement.rows;
  int mc = measurement.cols;
  int sr = state.rows;
  int sc = state.cols;

  if (mr > sr || mc > sc)
  {
    fprintf(
        stderr,
        "Measurement (%d x %d) has a bigger size than state (%d x %d).",
        mr, mc, sr, sc);
    exit(EXIT_FAILURE);
  }

  measurement.copyTo(state(cv::Rect(0, 0, mc, mr)));
}
#endif  // ifdef NEUCVH_USE_OPENCV
//********************************************* NKalmanFilter Implementation}}}

#endif  // NEU_CV_H
