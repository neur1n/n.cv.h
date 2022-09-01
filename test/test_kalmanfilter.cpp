#include "n.cv.h"

#include <iostream>
#include <vector>


int main(int argc, char **argv)
{
  std::vector<std::pair<float, float>> raw(10);
  for (size_t i = 0; i < raw.size(); ++i)
  {
    raw[i].first = (float)i + static_cast<float>(rand() % 10) / 10.0f;
    raw[i].second = (float)i +static_cast<float>(i + rand() % 10) / 10.0f;
    printf("%f, %f\n\n", raw[i].first, raw[i].second);
  }

  NKalmanFilter<Eigen::MatrixXf> *ekf = new NKalmanFilterEigen<float>(4, 2, 0, 0.1f, 1.0f);
  NKalmanFilter<cv::Mat> *okf = new NKalmanFilterOpenCV(4, 2, 0, 0.1f, 1.0f);

  Eigen::Matrix<float, 2, 1> em;
  cv::Mat om(2, 1, CV_32F, 0.0f);

  for (size_t i = 0; i < raw.size(); ++i)
  {
    std::cout << ">>>>>>>>>>>>>>>>>>>>" << std::endl;

    em(0, 0) = raw[i].first;
    em(1, 0) = raw[i].second;
    om.at<float>(0, 0) = raw[i].first;
    om.at<float>(1, 0) = raw[i].second;

    Eigen::MatrixXf eo = ekf->Filter(em);
    std::cout << "Eigen output:\n" << eo << std::endl << std::endl;

    cv::Mat oo = okf->Filter(om);
    std::cout << "OpenCV output:\n" << oo << std::endl;

    std::cout << "<<<<<<<<<<<<<<<<<<<<" << std::endl << std::endl;
  }

  delete ekf;
  ekf = nullptr;

  delete okf;
  okf = nullptr;

  return 0;
}
