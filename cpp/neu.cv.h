/******************************************************************************
Copyright (c) 2022 Jihang Li
neu.cv.h is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan
PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.


Last update: 2022-07-21 14:53
Version: v0.1.1
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


// From https://github.com/Neur1n/neu.h
#ifndef NEU_H
#include <time.h>

#if defined(_MSC_VER)
#define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#define __PRETTY_FUNCTION__ __FUNCSIG__
#else
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

#ifdef NDEBUG
#define n_assert(expr) do { \
  if (!(expr)) { \
    char ts[26] = {0}; \
    fprintf(stderr, "\n[ASSERTION FAILURE %s | %s - %s - %d] \n%s", \
        n_timestamp(ts, 26), n_full_path(__FILENAME__, ts), __FUNCTION__, __LINE__, #expr); \
    exit(EXIT_FAILURE); } \
} while (false)
#else
#define n_assert(expr) do {assert(expr);} while (false)
#endif

#define n_pi(T) (T)(3.141592653589793238462643383279502884197169399375)

double n_duration(
    const struct timespec start, const struct timespec end, const char* unit);

#define n_release(x) do { \
  if (x != nullptr) { \
    delete x; \
    x = nullptr; \
  } \
} while (false)

#define n_release_array(x) do { \
  if (x != nullptr) { \
    delete[] x; \
    x = nullptr; \
  } \
} while (false)
#endif


//*********************************************** NLowPassFilter Declaration{{{
template<class T = float>
class NLowPassFilter
{
public:
  NLowPassFilter(const size_t& length, const T* alpha = nullptr);

  ~NLowPassFilter();

  void Filter(
      const T* x, T* hatx, size_t* length = nullptr, const T* alpha = nullptr);

  void GetLastValue(T* hatxprev, size_t* length = nullptr);

  void Reset(const size_t* length = nullptr, const T* alpha = nullptr);

  void SetAlpha(const T* alpha);

private:
  bool m_initialized;
  size_t m_length;
  T* m_alpha;
  T* m_hatxprev;
};  // class NLowPassFilter
//}}}

//*********************************************** NOneEuroFilter Declaration{{{
template<class T = float>
class NOneEuroFilter
{
public:
  NOneEuroFilter(
      const size_t& length, const T& frequency, const T& beta = (T)0,
      const T& mincutoff = (T)0, const T& dcutoff = (T)0);

  ~NOneEuroFilter();

  void Filter(
      const T* x, T* hatx, size_t* length = nullptr,
      const struct timespec* timestamp = nullptr);

  void Reset(
      const size_t* length = nullptr, const T* frequency = nullptr,
      const T* beta = nullptr, const T* mincutoff = nullptr,
      const T* dcutoff = nullptr);

private:
  void UpdateAlpha(const T* cutoff, T* alpha);

  bool m_initialized;
  struct timespec m_timestamp;
  size_t m_length;
  T m_frequency;
  T m_beta;
  T* m_dcutoff;
  T* m_mincutoff;
  NLowPassFilter<T>* m_lpf;
  NLowPassFilter<T>* m_dlpf;
};  // class OneEuroFilter
//}}}

//************************************************ NKalmanFilter Declaration{{{
template <class T>
class NKalmanFilter
{
  static_assert(std::is_class<T>::value, "Non-class type is not supported yet.");

public:
  virtual ~NKalmanFilter() {}

  virtual T Correct(const T& measurement) = 0;

  virtual T Filter(const T& measurement, const T& control = T()) = 0;

  virtual T Peek(const unsigned int& steps, const T& control = T()) = 0;

  virtual T Predict(const T& control = T()) = 0;

  virtual void Reset(
      const int& state_dimension, const int& measurement_dimension,
      const int& control_dimension, const float& q_scale = std::nanf(""),
      const float& r_scale = std::nanf("")) = 0;

  virtual T GetControlMatrix() const = 0;

  virtual T GetLastState() const = 0;

  virtual T GetMeasurementMatrix() const = 0;

  virtual T GetMeasurementNoise() const = 0;

  virtual T GetProcessNoise() const = 0;

  virtual T GetTransitionMatrix() const = 0;

  virtual void SetControlMatrix(const T& B) = 0;

  virtual void SetMeasurementMatrix(const T& H) = 0;

  virtual void SetMeasurementNoise(const T& R) = 0;

  virtual void SetProcessNoise(const T& Q) = 0;

  virtual void SetTransitionMatrix(const T& A) = 0;
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
      const int& state_dimension, const int& measurement_dimension,
      const int& control_dimension, const float& r_scale = 1.0f,
      const float& q_scale = 1.0f);

  ~NKalmanFilterEigen();

  MatrixXT Correct(const MatrixXT& measurement);

  MatrixXT Filter(
      const MatrixXT& measurement, const MatrixXT& control = MatrixXT());

  MatrixXT Peek(const unsigned int& steps, const MatrixXT& control = MatrixXT());

  MatrixXT Predict(const MatrixXT& control = MatrixXT());

  void Reset(
      const int& state_dimension, const int& measurement_dimension,
      const int& control_dimension, const float& r_scale = std::nanf(""),
      const float& q_scale = std::nanf(""));

  MatrixXT GetControlMatrix() const;

  MatrixXT GetLastState() const;

  MatrixXT GetMeasurementMatrix() const;

  MatrixXT GetMeasurementNoise() const;

  MatrixXT GetProcessNoise() const;

  MatrixXT GetTransitionMatrix() const;

  void SetControlMatrix(const MatrixXT& control);

  void SetMeasurementMatrix(const MatrixXT& H);

  void SetMeasurementNoise(const MatrixXT& R);

  void SetProcessNoise(const MatrixXT& Q);

  void SetTransitionMatrix(const MatrixXT& transition);

private:
  void MeasurementToState(const MatrixXT& measurement, MatrixXT& state);

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
      const int& state_dimension, const int& measurement_dimension,
      const int& control_dimension, const float& r_scale = 1.0f,
      const float& q_scale = 1.0f);

  ~NKalmanFilterOpenCV();

  cv::Mat Correct(const cv::Mat& measurement);

  cv::Mat Filter(
      const cv::Mat& measurement, const cv::Mat& control = cv::Mat());

  cv::Mat Predict(const cv::Mat& control = cv::Mat());

  cv::Mat Peek(const unsigned int& steps, const cv::Mat& control = cv::Mat());

  void Reset(
      const int& state_dimension, const int& measurement_dimension,
      const int& control_dimension, const float& r_scale = std::nanf(""),
      const float& q_scale = std::nanf(""));

  cv::Mat GetControlMatrix() const;

  cv::Mat GetLastState() const;

  cv::Mat GetMeasurementMatrix() const;

  cv::Mat GetMeasurementNoise() const;

  cv::Mat GetProcessNoise() const;

  cv::Mat GetTransitionMatrix() const;

  void SetControlMatrix(const cv::Mat& control);

  void SetMeasurementMatrix(const cv::Mat& H);

  void SetMeasurementNoise(const cv::Mat& R);

  void SetProcessNoise(const cv::Mat& Q);

  void SetTransitionMatrix(const cv::Mat& transition);

private:
  void MeasurementToState(const cv::Mat& measurement, cv::Mat& state);

  cv::KalmanFilter m_kf;
  bool m_initialized;
  float m_r_scale;
  float m_q_scale;
};  // class NKalmanFilterOpenCV
#endif  // ifdef NEUCVH_USE_OPENCV
//}}}

//******************************************** NLowPassFilter Implementation{{{
template<class T>
NLowPassFilter<T>::NLowPassFilter(const size_t& length, const T* alpha)
  :m_initialized(false), m_length(0), m_alpha(nullptr), m_hatxprev(nullptr)
{
  this->Reset(&length, alpha);
}

template<class T>
NLowPassFilter<T>::~NLowPassFilter()
{
  n_release_array(this->m_alpha);
  n_release_array(this->m_hatxprev);
}

template<class T>
void NLowPassFilter<T>::Filter(
    const T* x, T* hatx, size_t* length, const T* alpha)
{
  n_assert(x != nullptr && hatx != nullptr);

  if (alpha != nullptr)
  {
    this->SetAlpha(alpha);
  }

  size_t size = this->m_length * sizeof(T);

  if (!this->m_initialized)
  {
    std::memcpy(this->m_hatxprev, x, size);
    this->m_initialized = true;
  }
  else
  {
    for (size_t i = 0; i < this->m_length; ++i)
    {
      this->m_hatxprev[i] =
        this->m_alpha[i] * x[i] + (1.0 - this->m_alpha[i]) * this->m_hatxprev[i];
    }
  }

  std::memcpy(hatx, this->m_hatxprev, size);

  if (length != nullptr)
  {
    *length = this->m_length;
  }
}

template<class T>
void NLowPassFilter<T>::GetLastValue(T* hatxprev, size_t* length)
{
  n_assert(hatxprev != nullptr);

  std::memcpy(hatxprev, this->m_hatxprev, this->m_length * sizeof(T));

  if (length != nullptr)
  {
    *length = this->m_length;
  }
}

template<class T>
void NLowPassFilter<T>::Reset(const size_t* length, const T* alpha)
{
  if (length != nullptr && this->m_length != *length)
  {
    this->m_length = *length;

    n_release_array(this->m_alpha);
    n_release_array(this->m_hatxprev);

    this->m_alpha = new T[this->m_length]{(T)0};
    this->m_hatxprev = new T[this->m_length]{(T)0};
  }

  if (alpha != nullptr)
  {
    this->SetAlpha(alpha);
  }

  this->m_initialized = false;
}

template<class T>
void NLowPassFilter<T>::SetAlpha(const T* alpha)
{
  n_assert(alpha != nullptr);

  for (size_t i = 0; i < this->m_length; ++i)
  {
    n_assert(alpha[i] > (T)0 && alpha[i] <= (T)1);
    this->m_alpha[i] = alpha[i];
  }
}
//}}}

//******************************************** NOneEuroFilter Implementation{{{
template<class T>
NOneEuroFilter<T>::NOneEuroFilter(
    const size_t& length, const T& frequency, const T& beta,
    const T& mincutoff, const T& dcutoff)
  :m_initialized(false), m_length(length), m_frequency(frequency), m_beta(beta),
  m_mincutoff(nullptr), m_dcutoff(nullptr), m_lpf(nullptr), m_dlpf(nullptr)
{
  this->Reset(&length, &frequency, &beta, &mincutoff, &dcutoff);
}

template<class T>
NOneEuroFilter<T>::~NOneEuroFilter()
{
  n_release_array(this->m_mincutoff);
  n_release_array(this->m_dcutoff);
  n_release(this->m_lpf);
  n_release(this->m_dlpf);
}

template<class T>
void NOneEuroFilter<T>::Filter(
    const T* x, T* hatx, size_t* length, const struct timespec* timestamp)
{
  if (timestamp != nullptr)
  {
    this->m_frequency = (T)(1.0 / n_duration(*timestamp, this->m_timestamp, "s"));
    this->m_timestamp = *timestamp;
  }

  T* dx = new T[this->m_length]{(0)};
  T* hatxprev = new T[this->m_length]{(0)};

  if (!this->m_initialized)
  {
    this->m_initialized = true;
  }
  else
  {
    this->m_lpf->GetLastValue(hatxprev);

    for (size_t i = 0; i < this->m_length; ++i)
    {
      dx[i] = (x[i] - hatxprev[i]) * this->m_frequency;
    }
  }

  T* edx = new T[this->m_length]{(0)};
  T* cutoff = new T[this->m_length]{(0)};
  T* alpha = new T[this->m_length]{(0)};

  this->m_dlpf->Filter(dx, edx);

  for (size_t i = 0; i < this->m_length; ++i)
  {
    if (edx[i] < (T)0)
    {
      edx[i] = - edx[i];
    }

    cutoff[i] = this->m_mincutoff[i] + this->m_beta * edx[i];
  }

  this->UpdateAlpha(cutoff, alpha);

  this->m_lpf->Filter(x, hatx, nullptr, alpha);

  if (length != nullptr)
  {
    *length = this->m_length;
  }

  n_release_array(dx);
  n_release_array(hatxprev);
  n_release_array(edx);
  n_release_array(cutoff);
  n_release_array(alpha);
}

template<class T>
void NOneEuroFilter<T>::Reset(
    const size_t* length, const T* frequency, const T* beta,
    const T* mincutoff, const T* dcutoff)
{
  if (length != nullptr)
  {
    n_assert(*length > 0);

    if (this->m_length != *length)
    {
      this->m_length = *length;

      n_release(this->m_dcutoff);
      n_release(this->m_mincutoff);
      n_release(this->m_lpf);
      n_release(this->m_dlpf);
    }
  }

  if (frequency != nullptr)
  {
    n_assert(*frequency >= (T)0);
    this->m_frequency = *frequency;
  }

  if (beta != nullptr)
  {
    n_assert(*beta >= (T)0);
    this->m_beta = *beta;
  }

  if (mincutoff != nullptr)
  {
    n_assert(*mincutoff >= (T)0);

    if (this->m_mincutoff == nullptr)
    {
      this->m_mincutoff = new T[this->m_length]{(T)*mincutoff};
    }

    std::fill_n(this->m_mincutoff, this->m_length, *mincutoff);
  }

  if (dcutoff != nullptr)
  {
    n_assert(*dcutoff >= (T)0);

    if (this->m_dcutoff == nullptr)
    {
      this->m_dcutoff = new T[this->m_length]{(T)*dcutoff};
    }

    std::fill_n(this->m_dcutoff, this->m_length, *dcutoff);
  }

  T* alpha = new T[this->m_length]{T(0)};

  this->UpdateAlpha(this->m_mincutoff, alpha);

  if (this->m_lpf == nullptr)
  {
    this->m_lpf = new NLowPassFilter<T>(this->m_length, alpha);
  }
  else
  {
    this->m_lpf->SetAlpha(alpha);
  }

  this->UpdateAlpha(this->m_dcutoff, alpha);

  if (this->m_dlpf == nullptr)
  {
    this->m_dlpf = new NLowPassFilter<T>(this->m_length, alpha);
  }
  else
  {
    this->m_dlpf->SetAlpha(alpha);
  }

  n_release_array(alpha);

  this->m_initialized = false;
}

template<class T>
void NOneEuroFilter<T>::UpdateAlpha(const T* cutoff, T* alpha)
{
  n_assert(cutoff != nullptr);

  T tau = (T)0;

  for (size_t i = 0; i < this->m_length; ++i)
  {
    tau = (T)1.0 / ((T)2.0 * n_pi(T) * cutoff[i]);
    alpha[i] = (T)1.0 / ((T)1.0 + tau * this->m_frequency);
  }
}
//}}}

//********************************************* NKalmanFilter Implementation{{{
#ifdef NEUCVH_USE_EIGEN
template <class T>
NKalmanFilterEigen<T>::NKalmanFilterEigen(
    const int& state_dimension, const int& measurement_dimension,
    const int& control_dimension, const float& r_scale, const float& q_scale)
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
NKalmanFilterEigen<T>::Correct(const MatrixXT& measurement)
{
  this->m_temp = this->m_error_pre * this->m_measurement.transpose();

  this->m_gain =
    this->m_temp
    * (this->m_measurement * this->m_temp + this->m_measurement_noise).inverse();

  this->m_state_post =
    this->m_state_pre
    + this->m_gain * (measurement - this->m_measurement * this->m_state_pre);

  this->m_error_post = this->m_error_pre - this->m_gain * this->m_temp.transpose();

  return this->m_state_post.block(0, 0, measurement.rows(), measurement.cols());
}

template <class T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
NKalmanFilterEigen<T>::Filter(
    const MatrixXT& measurement, const MatrixXT& control)
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
NKalmanFilterEigen<T>::Predict(const MatrixXT& control)
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
NKalmanFilterEigen<T>::Peek(const unsigned int& steps, const MatrixXT& control)
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
    const int& state_dimension, const int& measurement_dimension,
    const int& control_dimension, const float& r_scale, const float& q_scale)
{
  if (state_dimension <= 0 || measurement_dimension <= 0)
  {
    fprintf(stderr, "Dimensions of state and measurement must be positive.\n");
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
NKalmanFilterEigen<T>::GetLastState() const
{
  return this->m_state_post;
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
void NKalmanFilterEigen<T>::SetControlMatrix(const MatrixXT& B)
{
  this->m_control = B;
}

template <class T>
void NKalmanFilterEigen<T>::SetMeasurementMatrix(const MatrixXT& H)
{
  this->m_measurement = H;
}

template <class T>
void NKalmanFilterEigen<T>::SetMeasurementNoise(const MatrixXT& R)
{
  this->m_measurement_noise = R;
}

template <class T>
void NKalmanFilterEigen<T>::SetProcessNoise(const MatrixXT& Q)
{
  this->m_process_noise = Q;
}

template <class T>
void NKalmanFilterEigen<T>::SetTransitionMatrix(const MatrixXT& A)
{
  this->m_transition = A;
}

template <class T>
void NKalmanFilterEigen<T>::MeasurementToState(
    const MatrixXT& measurement, MatrixXT& state)
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
    const int& state_dimension, const int& measurement_dimension,
    const int& control_dimension, const float& r_scale, const float& q_scale)
  :m_initialized(false), m_r_scale(1.0f), m_q_scale(1.0f)
{
  this->Reset(
      state_dimension, measurement_dimension, control_dimension,
      r_scale, q_scale);
}

inline NKalmanFilterOpenCV::~NKalmanFilterOpenCV()
{
}

inline cv::Mat NKalmanFilterOpenCV::Correct(const cv::Mat& measurement)
{
  this->m_kf.correct(measurement);
  return this->m_kf.statePost(cv::Rect(0, 0, measurement.cols, measurement.rows));
}

inline cv::Mat NKalmanFilterOpenCV::Filter(
    const cv::Mat& measurement, const cv::Mat& control)
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

inline cv::Mat NKalmanFilterOpenCV::Predict(const cv::Mat& control)
{
  this->m_kf.predict(control);
  return this->m_kf.statePre;
}

inline cv::Mat NKalmanFilterOpenCV::Peek(
    const unsigned int& steps, const cv::Mat& control)
{
  cv::KalmanFilter kf = this->m_kf;

  for (int i = 0; i < steps; ++i)
  {
    kf.predict(control);
  }

  return kf.statePre;
}

inline void NKalmanFilterOpenCV::Reset(
    const int& state_dimension, const int& measurement_dimension,
    const int& control_dimension, const float& r_scale, const float& q_scale)
{
  if (state_dimension <= 0 || measurement_dimension <= 0)
  {
    fprintf(stderr, "Dimensions of state and measurement must be positive.\n");
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

inline cv::Mat NKalmanFilterOpenCV::GetLastState() const
{
  return this->m_kf.statePost;
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

inline void NKalmanFilterOpenCV::SetControlMatrix(const cv::Mat& B)
{
  this->m_kf.controlMatrix = B;
}

inline void NKalmanFilterOpenCV::SetMeasurementMatrix(const cv::Mat& H)
{
  this->m_kf.measurementMatrix = H;
}

inline void NKalmanFilterOpenCV::SetMeasurementNoise(const cv::Mat& R)
{
  this->m_kf.measurementNoiseCov = R;
}

inline void NKalmanFilterOpenCV::SetProcessNoise(const cv::Mat& Q)
{
  this->m_kf.processNoiseCov = Q;
}

inline void NKalmanFilterOpenCV::SetTransitionMatrix(const cv::Mat& A)
{
  this->m_kf.transitionMatrix = A;
}

inline void NKalmanFilterOpenCV::MeasurementToState(
    const cv::Mat& measurement, cv::Mat& state)
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
//}}}

#ifndef NEU_H
inline double n_duration(
    const struct timespec start, const struct timespec end, const char* unit)
{
  double diff = (double)(
      (end.tv_sec - start.tv_sec) * 1000000000LL + end.tv_nsec - start.tv_nsec);

  if (strcmp(unit, "h") == 0)
  {
    return diff / 3600000000000.0;
  }
  else if (strcmp(unit, "m") == 0)
  {
    return diff / 60000000000.0;
  }
  else if (strcmp(unit, "s") == 0)
  {
    return diff / 1000000000.0;
  }
  else if (strcmp(unit, "ms") == 0)
  {
    return diff / 1000000.0;
  }
  else if (strcmp(unit, "us") == 0)
  {
    return diff / 1000.0;
  }
  else  // if (strcmp(unit, "ns") == 0)
  {
    return diff;
  }
}
#endif

#endif  // NEU_CV_H
