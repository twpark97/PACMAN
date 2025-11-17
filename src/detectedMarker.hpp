#pragma once

#include "common.hpp"

namespace paacman{


struct DetectedFeature{
  struct {
    int id;
    cv::Point2f ce;//center
    cv::Point2f cec; // center in cell, meaning that cec == ce / 8
    float conf;
    bool supressed_by_nms;
    
    cv::Point2f feature_center_subpix;
  }step1; // these are the results from step1

  struct {
    int seq_dnn; // sequence number of the feature in the dnn output
    cv::Point2f feature_center_subpix_undistorted;
    int label;
  } step2;

  bool operator<(const DetectedFeature& other) const{
    return step1.conf < other.step1.conf;
  }
};

struct DetectedMarker{
  int id;

  // from regular shape check
  vector<cv::Point2f> pts_clockwise_distorted, pts_clockwise_undistorted;
  cv::Point2f center_2d_undistorted;
  
  // from direction check
  cv::Mat rvec, tvec;
  vector<int> labels; 

  double getYaw(){
    cv::Mat rvec;
    cv::Rodrigues(this->rvec, rvec);
    double yaw = atan2(rvec.at<double>(1,0), rvec.at<double>(0,0));
    return yaw;
  }
  double getYawDeg(){
    cv::Mat rvec;
    cv::Rodrigues(this->rvec, rvec);
    double yaw = atan2(rvec.at<double>(1,0), rvec.at<double>(0,0));
    double yaw_deg = yaw * 180 / M_PI;
    if (yaw_deg < 0) yaw_deg += 360;
    return yaw_deg;
  }
};

} // namespace paacman