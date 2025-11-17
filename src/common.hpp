#pragma once

#include <iostream>
#include <vector>
#include <filesystem>

#include <opencv4/opencv2/opencv.hpp>

namespace paacman{
using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::map;
namespace fs = std::filesystem;



enum {STEP_ZERO, STEP1, STEP2};
enum {LABELLING_NOT_REQUIRED=-1, 
  LABELLING_TYPE_0_PREV, LABELLING_TYPE_0_NEXT, LABELLING_TYPE_1};
  // labelling type 0: 4 pts form rectangle
  // labelling type 1: there exist 2 pair with label distance 1, 
    // or 1 pair with label distance 2

const int N_STEP = 2;
const int N_KEYPOINT_IN_MAKRER = 6;
static constexpr int N_MIN_REQUIRED_PTS = 4;
const string STEP1_STR="step1";
const string STEP2_STR="step2";
static constexpr int MAX_ALLOWED_THREADS = 500;
static constexpr int MAX_ALLOWED_N_FEATURE_CANDIDATES = 30; // 30C4 == 27405.
static constexpr int MAX_COMBINATORIAL_OPTIMIZATION_TRIALS = 10000;
  // 30C4 is the number of combinations of 30 features taken 4 at a time.
  // if there are more than 30 features, we will not check all combinations
constexpr static int STEP2_MIN_RADIUS_PIXELS[] = {7, 15, 23};

inline void my_error(string msg){
  throw std::runtime_error(msg.c_str());
}
inline bool check_valid_step(int step){
  return step >= STEP_ZERO && step <= STEP2;
}
} // namespace paacman