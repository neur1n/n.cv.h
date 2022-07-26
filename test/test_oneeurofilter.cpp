#include "neu.cv.h"

#include <iostream>
#include <vector>


int main(int argc, char **argv)
{
  float input[10]{0.0f};
  for (size_t i = 0; i < 5; ++i)
  {
    input[i] = (float)i + static_cast<float>(rand() % 10) / 10.0f;
    input[i + 1] = (float)i +static_cast<float>(i + rand() % 10) / 10.0f;
    printf("Input %zu: %f, %f\n\n", i, input[i], input[i + 1]);
  }

  NOneEuroFilter<float> oe(10, 30.0f, 0.001f, 0.01f, 1.0f);

  float output[10]{0.0f};

  oe.Filter(input, output);

  printf(">>>>>>>>>>>>>>>>>>>>\n");
  for (size_t i = 0; i < 5; ++i)
  {
    printf("Output %zu: %f, %f\n", i, output[i], output[i + 1]);
  }
  printf("<<<<<<<<<<<<<<<<<<<<\n");

  return 0;
}
