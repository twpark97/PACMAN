#pragma once

#include <mutex>
#include <chrono>
#include <fstream>

#include "common.hpp"
#include "parameters.hpp"
#include "detectedMarker.hpp"

// this one contains debugging info
// and visualize step output
namespace paacman{

class TimeLogger{
public:
  void init(){}

  void flush(){
    out_file.open(logging_file);
    if (false == out_file.is_open()){
      my_error(string() + "failed to open " + logging_file);
      return;
    }

    out_file << STEP1 << "(us)," << STEP2 << "(us)," << endl;
    for(auto& time_log : time_logs){
      out_file 
        << time_log[STEP1_STR] << "," 
        << time_log[STEP2_STR] << '\n';
    }
    out_file.close();
  }
  int getFlushSeq() const {
    return flush_seq_;
  }
  void setLoggingFile(string f, int flush_seq){
    logging_file = f;
    flush_seq_ = flush_seq;
  }
  void setDoPrint(bool do_print){
    do_print_ = do_print;
  }
  void initNewTiming(){
    time_logs.push_back(std::map<string, int>());
  }
  void tic(){
    start = std::chrono::high_resolution_clock::now();
  }
  void toc(string step_name, string n){
    end = std::chrono::high_resolution_clock::now();
    duration_count_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    time_logs.back()[step_name] = duration_count_us;

    if (do_print_){
      cout << n << "(time elapsed: " << duration_count_us << " us" << ") "<< endl;
    }
  }
private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  int duration_count_us;
  vector<std::map<string, int>> time_logs; //[seq]["step_name"] -> time in ms
  bool do_print_;

  string logging_file;
  std::ofstream out_file;
  int flush_seq_;
};


class PaacmanDebugDetail{
public:
  // debugging msg
  void updateLastTriedStep(string msg, int step = -1){
    if (step != -1) last_tried_step = step;
    msgs.push_back(msg);
  }
  void printTriedSteps(){
    std::lock_guard<std::mutex> lock(mtx);
    cout << "Detection failed - last tried step: " << last_tried_step << endl;
    for(auto& msg : msgs){
      cout << "  " << msg << endl;
    }
  }
  void clear(){
    std::lock_guard<std::mutex> lock(mtx);
    last_tried_step = -1;
    msgs.clear();
  }

  // visualization_debugging
  void setDebugDir(
    string debug_root, 
    std::map<int, string> debug_dir_map, 
    bool clear_dir=false)
  {
    debug_root_ = debug_root;
    debug_dir_map_ = debug_dir_map;

    // clear tree if clear_dir == true
    if (true == clear_dir){
      fs::remove_all(debug_root_);
    }
    
    // create tree with error checking
    fs::create_directories(debug_root_);
    if (false == fs::is_directory(debug_root_))
      my_error(string() + "failed to create " + debug_root_);
    for(auto& dir : debug_dir_map_){
      fs::path dir_path = fs::path(debug_root_) / dir.second;
      fs::create_directories(dir_path);
      if (false == fs::is_directory(dir_path))
        my_error(string() + "failed to create " + dir_path.string());
    }
  }

  void setDebugParams(const DebugInfo& debug_info){
    debug_info_ = debug_info;
  }

  void saveImage(
    const cv::Mat* img_ptr,
    int step,
    string name,
    unsigned int seq,
    bool called_internally=false
  ){
    if (false == called_internally)
      std::lock_guard<std::mutex> lock(mtx);
    
    if (false == check_valid_step(step)){
      my_error(string() + "invalid step in saveImage: " + std::to_string(step));
    }
    else if (nullptr == img_ptr){
      cout << "image_ptr is nullptr in saveImage" << endl;
      return;
    }
    else if (img_ptr->rows == 0 || img_ptr->cols == 0){
      my_error("invalid image in saveImage");
    }
    string dbg_dir = debug_dir_map_[step];
    fs::path img_path = 
      fs::path(debug_root_)  / dbg_dir / 
        (std::to_string(seq) + "_" + name + ".png");
    cv::imwrite(img_path.string(), *img_ptr);
  }

  void saveImageRectified(
    const cv::Mat* img_ptr,
    int step,
    string name,
    unsigned int seq,
    const PaacmanParameters& params,
    bool called_internally=false
  ){
    if (false == called_internally)
      std::lock_guard<std::mutex> lock(mtx);
    
    if (false == check_valid_step(step)){
      my_error(string() + "invalid step in saveImage: " + std::to_string(step));
    }
    else if (nullptr == img_ptr){
      cout << "image_ptr is nullptr in saveImage" << endl;
      return;
    }
    else if (img_ptr->rows == 0 || img_ptr->cols == 0){
      my_error("invalid image in saveImage");
    }
    string dbg_dir = debug_dir_map_[step];
    fs::path img_path = 
      fs::path(debug_root_)  / dbg_dir / 
        (std::to_string(seq) + "_" + name + ".png");
    cv::Mat img_undistorted;
    cv::undistort(*img_ptr, img_undistorted, 
      params.getCameraInfo().K, params.getCameraInfo().D, 
      params.getCameraInfo().new_K
    );
    cv::imwrite(img_path.string(), img_undistorted);
  }

  // step1 related
  void step1_visualizeDetectedFeature(
    const cv::Mat* img_ptr, 
    const vector<DetectedFeature>& features,
    int seq
  ){
    std::lock_guard<std::mutex> lock(mtx);
    cv::Mat img_copy;
    cv::cvtColor(*img_ptr, img_copy, cv::COLOR_GRAY2BGR);

    // draw vague circle near feature one the RGB image
    cv::Mat img_rgb = cv::Mat::zeros(img_copy.rows, img_copy.cols, CV_8UC3);
    for (const auto& f: features){
      cv::circle(
        // img_rgb, f.step1.ce, debug_info_.step1_debug_feature_radius, 
        // debug_info_.id_to_color(f.step1.id), -1);
        img_rgb, f.step1.ce, debug_info_.step1_debug_feature_radius, 
        debug_info_.id_to_color(f.step1.id), 10); //twtw
    }
    // cv::addWeighted(img_copy, 1.0, img_rgb, 2.0, 0, img_rgb);

    vector<cv::Point> pts;
    vector<int> ids;
    for (const auto& f : features){
      pts.push_back(f.step1.ce);
      ids.push_back(f.step1.id);
    }
    drawCirclesNearPts(&img_copy, pts, ids, debug_info_);
    saveImage(&img_copy, STEP1, "step1_feature_circle", seq, true);
  }

  void step1_visualizeDecoderOut(
    const cv::Mat* img_ptr,
    const vector<float>& raw_decoder_out,
    const vector<DetectedFeature>& features,
    int seq
  )
  {
    // construct decoder image, meeaning that
    std::lock_guard<std::mutex> lock(mtx);
    int H = img_ptr->rows;
    int Hc = H/8;
    int W = img_ptr->cols;
    int Wc = W/8;
    cv::Mat img_dec(H, W, CV_32FC1);

    // (64, H/8 W/8) -> (H, W)
    for (int pixel = 0 ; pixel < 64; ++pixel){
      for (int hc = 0 ; hc < Hc; ++hc){
        for (int wc = 0; wc < Wc; ++wc){
          float val = raw_decoder_out[pixel*Hc*Wc + hc*Wc + wc];
          
          int x = wc*8 + pixel%8;
          int y = hc*8 + pixel/8;

          img_dec.at<float>(y, x) = val;
        }
      }
    }

    // normalize to 0 ~ 1
    double min, max;
    cv::minMaxLoc(img_dec, &min, &max);
    img_dec = (img_dec - min) / (max - min);

    // convert to 0 ~ 255
    img_dec = img_dec * 255;
    img_dec.convertTo(img_dec, CV_8UC1);

    // convert to JET
    cv::Mat img_dec_jet = cv::Mat::ones(H, W, CV_8UC3);
    // set scalar
    // img_dec_jet.setTo(cv::Scalar(255, 204, 204));
    cv::applyColorMap(img_dec, img_dec_jet, cv::COLORMAP_TWILIGHT);


    // draw boxes
    vector<cv::Point> pts;
    vector<int> ids;
    for (const auto& f : features){
      pts.push_back(f.step1.ce);
      ids.push_back(f.step1.id);
    }
    drawCirclesNearPts(&img_dec_jet, pts, ids, debug_info_);


    // save
    saveImage(&img_dec_jet, STEP1, "step1_decoder_out", seq, true);
  }

  void step2_visualizeUndistortedPoints(
    const cv::Mat* img_ptr,
    const map<int, vector<DetectedFeature>> id_to_features,
    const map<int, bool> id_to_is_valid,
    const PaacmanParameters& params,
    int seq
  )
  {
    std::lock_guard<std::mutex> lock(mtx);

    int n_desc = params.getDnnInfo().n_desc;

    for (int id = 0; id < n_desc; ++id){
      if (false == id_to_is_valid.at(id)) continue;

      //rectify image
      cv::Mat img_undistorted;
      cv::undistort(*img_ptr, img_undistorted, 
        params.getCameraInfo().K, params.getCameraInfo().D, 
        params.getCameraInfo().new_K
      );
      cv::cvtColor(img_undistorted, img_undistorted, cv::COLOR_GRAY2BGR);

      // draw feature in undistorted image
      for (const auto& f : id_to_features.at(id)){
        int x_int = f.step2.feature_center_subpix_undistorted.x;
        int y_int = f.step2.feature_center_subpix_undistorted.y;

        auto c = params.getDebugInfo().id_to_color(id);
        img_undistorted.at<cv::Vec3b>(y_int, x_int)[0] = c[0];
        img_undistorted.at<cv::Vec3b>(y_int, x_int)[1] = c[1];
        img_undistorted.at<cv::Vec3b>(y_int, x_int)[2] = c[2];
      }

      // draw boxes
      vector<cv::Point> pts;
      vector<int> ids;
      for (const auto& f : id_to_features.at(id)){
        pts.push_back(f.step2.feature_center_subpix_undistorted);
        ids.push_back(f.step1.id);
      }
      drawCirclesNearPts(&img_undistorted, pts, ids, params.getDebugInfo());

      saveImage(&img_undistorted, STEP2, 
        string() + "step2_undistorted_id" + std::to_string(id), seq, true);
    }
  }

  void step2_visualizeDiagIntersectionTrg(
    int width, int height, vector<cv::Point2f> pts, cv::Point2f pt,
    float minmax_rate, int labelling_type, 
    vector<int> pseudo_labels,
    vector<int> right_triangle,
    cv::Point3f pt_other_perfect = cv::Point3f(0, 0, 0)
  )
  {
    std::lock_guard<std::mutex> lock(mtx);
    static int seq = 0;
    cv::Mat img = cv::Mat::zeros(height, width, CV_8UC3);
    for (auto& p : pts){
      cv::circle(img, p, 5, cv::Scalar(0, 0, 255), -1);
    }

    // insert minmax_rate and labelling_type
    cv::Scalar cc;
    switch(labelling_type){
    case LABELLING_NOT_REQUIRED: cc = cv::Scalar(0, 0, 255); break;
    case LABELLING_TYPE_0_PREV: cc = cv::Scalar(0, 255, 0); break;
    case LABELLING_TYPE_0_NEXT: cc = cv::Scalar(255, 0, 0); break;
    case LABELLING_TYPE_1: cc = cv::Scalar(127, 127, 0); break;
    default: my_error("invalid labelling_type");
    }
    cv::putText(img, std::to_string(minmax_rate), cv::Point(0, 100), cv::FONT_HERSHEY_SIMPLEX, 2, cc, 2);
    cv::putText(img, std::to_string(labelling_type), cv::Point(0, 200), cv::FONT_HERSHEY_SIMPLEX, 2, cc, 2);

    cv::circle(img, pt, 5, cv::Scalar(0, 255, 0), -1);
    //add label and seq in text
    for (int i = 0 ; i < (int)pseudo_labels.size(); ++i){
      cv::putText(img, std::to_string(pseudo_labels[i]), pts[i], cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 1);
    }

    // visualize right triangle
    const string strs[] = {"A", "B", "C"};
    for (int i = 0 ; i < (int)right_triangle.size(); ++i){
      int pt_idx = right_triangle[i];
      cv::putText(img, strs[i], 
        pts[pt_idx] + cv::Point2f(-30, 0), 
        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 1);
    }

    // draw arrow from C to pt_other_perfect
    if (cv::norm(pt_other_perfect) > 0.00001){
      cv::Point2f pt_other_perfect_image 
        = cv::Point2f(pt_other_perfect.x, pt_other_perfect.y);
      cv::arrowedLine(img, pts[right_triangle[2]], 
        pt_other_perfect_image, cv::Scalar(0, 255, 0), 2);
      cv::circle(img, pt_other_perfect_image, 5, cv::Scalar(0, 0, 255), -1);
    }

    saveImage(&img, STEP2, 
      string() + "step2_diag_intersection" + std::to_string(seq++), STEP2, true);
  }

  void step2_visualizeDiagIntersection(
    int width, int height, vector<cv::Point2f> pts, cv::Point2f pt,
    float minmax_rate, int labelling_type, 
    vector<int> pseudo_labels,
    cv::Point3f pt_other_perfect = cv::Point3f(0, 0, 0)
  )
  {
    std::lock_guard<std::mutex> lock(mtx);
    static int seq = 0;
    cv::Mat img = cv::Mat::zeros(height, width, CV_8UC3);
    for (auto& p : pts){
      cv::circle(img, p, 5, cv::Scalar(0, 0, 255), -1);
    }

    // insert minmax_rate and labelling_type
    cv::Scalar cc;
    switch(labelling_type){
    case LABELLING_NOT_REQUIRED: cc = cv::Scalar(0, 0, 255); break;
    case LABELLING_TYPE_0_PREV: cc = cv::Scalar(0, 255, 0); break;
    case LABELLING_TYPE_0_NEXT: cc = cv::Scalar(255, 0, 0); break;
    case LABELLING_TYPE_1: cc = cv::Scalar(127, 127, 0); break;
    default: my_error("invalid labelling_type");
    }
    cv::putText(img, std::to_string(minmax_rate), cv::Point(0, 100), cv::FONT_HERSHEY_SIMPLEX, 2, cc, 2);
    cv::putText(img, std::to_string(labelling_type), cv::Point(0, 200), cv::FONT_HERSHEY_SIMPLEX, 2, cc, 2);

    cv::circle(img, pt, 5, cv::Scalar(0, 255, 0), -1);
    //add label and seq in text
    for (int i = 0 ; i < (int)pseudo_labels.size(); ++i){
      cv::putText(img, std::to_string(pseudo_labels[i]), pts[i], cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 1);
    }

    // draw arrow from C to pt_other_perfect
    if (cv::norm(pt_other_perfect) > 0.00001){
      cv::Point2f pt_other_perfect_image 
        = cv::Point2f(pt_other_perfect.x, pt_other_perfect.y);
      cv::circle(img, pt_other_perfect_image, 5, cv::Scalar(0, 0, 255), -1);
    }
    
    saveImage(&img, STEP2, 
      string() + "step2_diag_intersection" + std::to_string(seq++), STEP2, true);
  }

  void step2_visualizeDirectionCheck(
    const cv::Mat* img_ptr,
    const PaacmanParameters& params,
    vector<cv::Point2f> pts_clockwise_undistorted,
    vector<int> labels,
    float rad_min,
    vector<float> directions_degree,
    cv::Mat rvec,
    cv::Mat tvec,
    vector<cv::Point2f> reprojected_pts,
    int id
    )
  {
    cv::Mat img_undistorted;
    cv::undistort(*img_ptr, img_undistorted, 
      params.getCameraInfo().K, params.getCameraInfo().D, 
      params.getCameraInfo().new_K
    );
    cv::cvtColor(img_undistorted, img_undistorted, cv::COLOR_GRAY2BGR);

    // draw circles considering radius
    auto c = params.getDebugInfo().id_to_color(id);

    for (int i = 0 ; i < (int)pts_clockwise_undistorted.size(); ++i){
      cv::circle(img_undistorted, pts_clockwise_undistorted[i], 
        rad_min, c, 1);
    }

    // write labels
    // for (int i = 0 ; i < (int)pts_clockwise_undistorted.size(); ++i){
    //   const auto& pt = pts_clockwise_undistorted[i];
    //   int x_int = pt.x 
    //     + params.getDebugInfo().enclosed_circle_radius;
    //   int y_int = pt.y
    //     - params.getDebugInfo().enclosed_circle_radius;

    //   int label = labels[i];
    //   cv::putText(img_undistorted, std::to_string(label), 
    //     cv::Point(x_int, y_int), cv::FONT_HERSHEY_SIMPLEX, 
    //     params.getDebugInfo().font_scale, 
    //     c, 1);
    // }

    // visualize direction vector
    for (int i = 0 ; i < (int)pts_clockwise_undistorted.size(); ++i){
      const auto& pt = pts_clockwise_undistorted[i];
      float rad = rad_min;
      float dir = directions_degree[i] * CV_PI / 180.f;

      cv::Point2f pt2 = pt + cv::Point2f(rad*cos(dir), rad*sin(dir));
      cv::arrowedLine(img_undistorted, pt, pt2, c, 10);
    }

    const auto& m_name = params.getMarkerNames().at(id);
    const auto& marker_config = params.getMarkerConfigs().at(m_name);
    // cv::drawFrameAxes(
    //   img_undistorted, params.getCameraInfo().new_K, 
    //   cv::noArray(), rvec, tvec, 
    //   marker_config.hexagon_radius_meter_
    // );

    // visualize reprojected points
    // for (int i = 0 ; i < (int)reprojected_pts.size(); ++i){
    //   c = cv::Scalar(200, 100, 50);

    //   cv::circle(img_undistorted, reprojected_pts[i], 
    //     3, c, 1);
    // }

    static int seq = 0;
    saveImage(&img_undistorted, STEP2, 
      string() + "step2_direction_check" + std::to_string(seq++), STEP2, true);
  }

  void step2_visualizeDirectionCheckSimple(
    const cv::Mat* img_ptr,
    const PaacmanParameters& params,
    vector<cv::Point2f> pts_clockwise_undistorted,
    float rad_min,
    vector<float> directions_degree,
    int id
    )
  {
    cv::Mat img_undistorted;
    cv::undistort(*img_ptr, img_undistorted, 
      params.getCameraInfo().K, params.getCameraInfo().D, 
      params.getCameraInfo().new_K
    );
    cv::cvtColor(img_undistorted, img_undistorted, cv::COLOR_GRAY2BGR);

    // draw circles considering radius
    auto c = params.getDebugInfo().id_to_color(id);

    // for (int i = 0 ; i < (int)pts_clockwise_undistorted.size(); ++i){
    //   cv::circle(img_undistorted, pts_clockwise_undistorted[i], 
    //     rad_min, c, 1);
    // }

    // visualize direction vector
    for (int i = 0 ; i < (int)pts_clockwise_undistorted.size(); ++i){
      const auto& pt = pts_clockwise_undistorted[i];
      float dir = directions_degree[i] * CV_PI / 180.f;

      // cv::Point2f pt2 = pt + cv::Point2f(rad_min*cos(dir), rad_min*sin(dir));
      // cv::arrowedLine(img_undistorted, pt, pt2, c, 2);

      cv::Point2f pt2 = pt + cv::Point2f(30*cos(dir), 30*sin(dir));
      cv::arrowedLine(img_undistorted, pt, pt2, c, 10, 8, 0, 0.3);

    }

    // vector<int> ids(pts_clockwise_undistorted.size());
    // std::fill_n(ids.begin(), pts_clockwise_undistorted.size(), id);
    // cout << "twtw id: " << id << endl;
    // vector<cv::Point2i> pts;
    // for (int i = 0 ; i < (int)pts_clockwise_undistorted.size(); ++i){
    //   pts.push_back(cv::Point2i(pts_clockwise_undistorted[i].x, pts_clockwise_undistorted[i].y));
    // }
    // drawCirclesNearPts(&img_undistorted, pts, ids, params.getDebugInfo());


    static int seq = 0;
    saveImage(&img_undistorted, STEP2, 
      string() + "step2_direction_simple_check" + std::to_string(seq++), STEP2, true);
  }

private:
  void drawCirclesNearPts(
    cv::Mat* img_ptr,
    const vector<cv::Point>& pts,
    vector<int> ids,
    const DebugInfo& debug_info
  )
  {
    // draw rectangle around pts
    for (size_t i = 0 ; i < pts.size(); ++i){
      auto pt = pts[i];
      int id = ids[i];

      
      auto c = debug_info.id_to_color(id);
      cv::circle(*img_ptr, pt, debug_info.enclosed_circle_radius, 
      c, debug_info.enclosed_circle_thickness);

      // draw pixel
      img_ptr->at<cv::Vec3b>(pt.y, pt.x)[0] = c[0];
      img_ptr->at<cv::Vec3b>(pt.y, pt.x)[1] = c[1];
      img_ptr->at<cv::Vec3b>(pt.y, pt.x)[2] = c[2];
    }
  }
private:
  // debugging msg 
  int last_tried_step;
  vector<string> msgs;
  std::mutex mtx; //thread safe

  // visualization debugging
  DebugInfo debug_info_;
  string debug_root_;
  std::map<int, string> debug_dir_map_;
};

} // namespace paacman