#include "paacman.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>
#include <random>

namespace paacman{
inline bool is_pt_in_image(int x, int y, const cv::Mat* mat){
  return x >= 0 && x < mat->cols && y >= 0 && y < mat->rows;
}

inline int dist_clock_a_b(int a, int b, int clock_size){
  int dist = a - b;
  if (dist < 0) dist += clock_size;
  dist %= clock_size;
  return dist;
}

Paacman::Paacman(string cfg_path){
  params.loadParameters(cfg_path);
  params.printParams();
  
  initPaacman();
}

void Paacman::initPaacman(){
  // utilites
  enable_time_logging = false;
  enable_debugging = false;
  seq = 0;

  /* step1 */
  trt_mgr_ptr = new Tensorrt10Manager(params.getDnnInfo().trt_engine_path);
  //warm up network
  cv::Mat dummy = cv::Mat::zeros(params.getCameraInfo().height, params.getCameraInfo().width, CV_8UC1);
  vector<vector<float>> net_output_vectors;
  for (int i = 0 ; i < 5; ++i)
    trt_mgr_ptr->runInference(dummy, net_output_vectors);


  /* step2 */
  found_marker = false;

  // contains (idx of (a, b, c), a < b). C is angle angle(ABC) forms 90 degree
  // (a,b,c) = label_type_to_right_triangle[labelling_type][candidate_idx]
  //    with n(candidate) == 2
  // always assumes a < b
  label_type_to_pseudo_labels[LABELLING_TYPE_0_PREV] 
    = {{0, 2, 3, 4}, {0, 3, 4, 5}};
  label_type_to_right_triangle[LABELLING_TYPE_0_PREV] 
    = {{2, 0, 3}, {1, 0, 2}};
  label_type_to_right_triangle_other_idx[LABELLING_TYPE_0_PREV] 
    = {1, 3};
  label_type_to_inverse_area[LABELLING_TYPE_0_PREV] 
    = {.4f/.6f, .5f};
  
  label_type_to_pseudo_labels[LABELLING_TYPE_0_NEXT] 
    = {{0, 2, 4, 5}, {0, 3, 4, 5}};
  label_type_to_right_triangle[LABELLING_TYPE_0_NEXT] 
    = {{3, 1, 0}, {1, 0, 2}};
  label_type_to_right_triangle_other_idx[LABELLING_TYPE_0_NEXT]
    = {2, 3};
  label_type_to_inverse_area[LABELLING_TYPE_0_NEXT] 
    = {.4f/.6f, .5f};

  label_type_to_pseudo_labels[LABELLING_TYPE_1] 
    = {{0, 1, 3, 4}, {0, 2, 3, 5}};
  label_type_to_right_triangle[LABELLING_TYPE_1] 
    = {{2, 0, 3}, {3, 1, 0}}; 
  label_type_to_right_triangle_other_idx[LABELLING_TYPE_1]
    = {1, 2};
  label_type_to_inverse_area[LABELLING_TYPE_1] 
    = {.4f/.6f, .4f/.6f};
  
  clock_dist_diff_to_length_3d_ratio.push_back(0);
  clock_dist_diff_to_length_3d_ratio.push_back(1);
  clock_dist_diff_to_length_3d_ratio.push_back(std::sqrt(3));
  clock_dist_diff_to_length_3d_ratio.push_back(2);
  clock_dist_diff_to_length_3d_ratio.push_back(std::sqrt(3));
  clock_dist_diff_to_length_3d_ratio.push_back(1);

  /* precompute all nC4 combinations */
  // c++ does not provide general combination function. 
  // combination can be implemented by boolean array and std::next_permutation
  // see https://whiteherv.tistory.com/category/C%2B%2B/STL
  combination_helper_vec3d.emplace_back(); // empty vectors since nC4 with n<4 does not make sense
  combination_helper_vec3d.emplace_back();
  combination_helper_vec3d.emplace_back();
  combination_helper_vec3d.emplace_back();
  for (int n_detected_features = 4 ; n_detected_features < MAX_ALLOWED_N_FEATURE_CANDIDATES;
   ++n_detected_features){
    vector<bool> combination_helper_vec(n_detected_features, true);
    std::fill_n(combination_helper_vec.begin(), N_MIN_REQUIRED_PTS, false);
    // false, false, ..., false, true, true, ..., true

    // assemble all combinations
    combination_helper_vec3d.emplace_back();
    auto& combination_helper_vec2d = combination_helper_vec3d.back();
    do{
      combination_helper_vec2d.emplace_back();
      for(size_t i = 0; i < combination_helper_vec.size(); ++i){
        if(false == combination_helper_vec[i]) {
          combination_helper_vec2d.back().emplace_back(i);
        }
      }
    } while (std::next_permutation(combination_helper_vec.begin(), combination_helper_vec.end()));

    std::shuffle(combination_helper_vec2d.begin(), combination_helper_vec2d.end(),
      std::default_random_engine(12345));
  }
}

bool Paacman::step1_network_inference(
  const cv::Mat& img,  //input: image that may contain paacman marker
  STEP1_OUT& step1_out) 
{
  step1_out.step_type = STEP1;
  vector<DetectedFeature>& detected_features = step1_out.detected_features;

  int W = params.getCameraInfo().width;
  int H = params.getCameraInfo().height;
  int Wc = W/8;
  int Hc = H/8;

  // network inference
  static bool is_file_open = false;
  static std::ofstream ofs;
  if (false == is_file_open){
    ofs.open("/workspace/estimate/debug/net_out.txt");
    is_file_open = true;
  }
  auto ts = std::chrono::high_resolution_clock::now();
  bool success = trt_mgr_ptr->runInference(img, step1_out.net_output_vectors);
  auto te = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(te - ts).count();
  ofs << duration << '\n';

  if (false == success) {
    debug_detail.updateLastTriedStep("TensorrtManager::runInference() failed", STEP1);
    return false;
  }

  //multithreaded lookup
  std::vector<size_t> feature_cand_indicies;

  static std::mutex mtx_step1;
  float conf_thresh = params.getDnnInfo().conf_thresh;
  auto find_positive = [&feature_cand_indicies, conf_thresh]
    (float* ptr, int start, int end){
    std::vector<size_t> index;
    for (int i = start ; i < end ; ++i){
      if (ptr[i] >= conf_thresh){
        mtx_step1.lock();
        feature_cand_indicies.push_back(i);
        mtx_step1.unlock();
      }
    }
  };


  auto& decoder_out = step1_out.net_output_vectors[0];

  // create search threads
  std::vector<std::thread> threads_;
  int n_threads = params.getAlgoParams().max_available_threads;
  for (int i = 0 ; i < n_threads; ++i){
    int start = i * decoder_out.size() / n_threads;
    int end = (i+1) * decoder_out.size() / n_threads;

    std::thread t(find_positive, decoder_out.data(), start, end);
    threads_.push_back(std::move(t));
  }
  for (auto& t : threads_) t.join();

  // find features with high confidence
  std::priority_queue<DetectedFeature> features_sorted;
  for (auto index: feature_cand_indicies){
    DetectedFeature f;
    int pixel = index / (Hc*Wc);
    int yc = (index % (Hc*Wc)) / Wc;
    int xc = index % Wc;

    // border check
    if (xc < params.getAlgoParams().border_cell) continue;
    if (xc >= W - params.getAlgoParams().border_cell) continue;
    if (yc < params.getAlgoParams().border_cell) continue;
    if (yc >= H - params.getAlgoParams().border_cell) continue;

    f.step1.id = -1;
    f.step1.conf = decoder_out[index];
    f.step1.ce.x = xc*8 + pixel%8;
    f.step1.ce.y = yc*8 + pixel/8;
    f.step1.cec.x = xc;
    f.step1.cec.y = yc;
    f.step1.supressed_by_nms = false;
    features_sorted.push(f);
  }

  if (features_sorted.size() >= 500){
    static bool warned_too_many_features_cands = false;
    if (false == warned_too_many_features_cands){
      cout << "Too many corner candidates(" << features_sorted.size() 
        << "), this would make further processings very slow" << endl;
      cout << "try tuning nms conf thresh or retrain your network" << endl;
      warned_too_many_features_cands = true;
    }
  }
  else if (features_sorted.size() == 0){
    debug_detail.updateLastTriedStep("No features detected", STEP1);
    return false;
  }

  // sort by confidnce
  vector<DetectedFeature> features_not_suppressed;
  while (false == features_sorted.empty()){
    features_not_suppressed.push_back(features_sorted.top());
    features_sorted.pop();
  }

  // nms
  size_t n_cand = features_not_suppressed.size();
  for(size_t i = 0; i < n_cand; ++i){
    if (true == features_not_suppressed[i].step1.supressed_by_nms) continue;

    for(size_t j = i + 1; j < n_cand; ++j){
      if (true == features_not_suppressed[j].step1.supressed_by_nms) continue;

      int dist_l1 = abs(features_not_suppressed[i].step1.ce.x - features_not_suppressed[j].step1.ce.x)
        + abs(features_not_suppressed[i].step1.ce.y - features_not_suppressed[j].step1.ce.y);
      if (dist_l1 <= params.getDnnInfo().nms_suppress_dist) 
        features_not_suppressed[j].step1.supressed_by_nms = true;
    }
  }

  // exclude suppressed features
  detected_features.resize(0);
  for (const auto& f: features_not_suppressed){
    if (false == f.step1.supressed_by_nms) detected_features.push_back(f);
  }
  

  // find id
  const auto& descriptor_out = step1_out.net_output_vectors[1]; // 4, Hc, Wc
  for (auto& feature: detected_features){
    int xc = feature.step1.cec.x;
    int yc = feature.step1.cec.y;

    // need to run argmax to find descriptor(id)
    float max_des_val = -1000.f;
    int max_des_id = params.getDnnInfo().n_desc;
    for (int id_cand = 0; id_cand < params.getDnnInfo().n_desc; ++id_cand){
      float des_val = descriptor_out[id_cand*Hc*Wc + yc*Wc + xc];
      if (des_val > max_des_val){
        max_des_val = des_val;
        max_des_id = id_cand;
      }
    }
    feature.step1.id = max_des_id;
  }

  //twtw 
  // static std::ofstream ofs2;
  // if (false == ofs2.is_open()){
  //   ofs2.open("/workspace/estimate/nms_tp_fp.csv");
  // }
  // static std::vector<cv::Point2i> tp_pts = {
  //   {598, 486},
  //   {598, 396},
  //   {491, 351},
  //   {385, 483},
  //   {492, 529},
  //   {385, 393},
  // };

  // // count # of tp or fp
  // float dist_thresh = 5.f;
  // int tp = 0, fp = 0;
  // for (const auto& f: detected_features){
  //   if (f.step1.id != 2) continue;
  //   bool is_tp = false;
  //   for (const auto& tp_pt: tp_pts){
  //     if (cv::norm(f.step1.ce - cv::Point2f(tp_pt)) < dist_thresh){
  //       is_tp = true;
  //       break;
  //     }
  //   }
  //   if (is_tp) tp++;
  //   else fp++;
  // }
  // ofs2 << tp << ',' << fp << '\n';
  // ofs2.flush();
  // scanf("%*d");

  // twtw paper done

 
  // localize feature center with 'cornerSubpix' asynchrounously
  /// define job
  auto cornerSubpix_job = [](const cv::Mat* img, vector<DetectedFeature*> features_ptr)
  {
    for(size_t i = 0 ; i < features_ptr.size(); ++i){
      DetectedFeature* feature_ptr = features_ptr[i];
      // vector<cv::Point2f> corner_inout = {feature_ptr->step1.ce};
      // cv::cornerSubPix(
      //   *img, 
      //   corner_inout,
      //   cv::Size(5,5), 
      //   cv::Size(-1,-1), 
      //   cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 20, 0.01)
      // );
      // feature_ptr->step1.feature_center_subpix = corner_inout[0];
      feature_ptr->step1.feature_center_subpix = feature_ptr->step1.ce;
    }
  };
  
  vector<DetectedFeature*> detected_features_ptr;
  for (auto& f: detected_features) detected_features_ptr.push_back(&f);
  cornerSubpix_job(&img, detected_features_ptr);
  
  
  
  /* 
    sequential processing is much better, as we processing very small image size and few points
    in my computer, processing 8 points w/
    32 thread takes 671 us while w/ sequential processing takes 35us 
  */

  return true;
}

bool Paacman::step2_shape_check(
  const cv::Mat& img,
  const STEP1_OUT& step1_out, 
  STEP2_OUT& step2_out)
{
  // I assumpt that a marker consists of n features with identical descriptor(id)
  // and also assumpt that (id of each feature == id of each marker)
  // if anyone want to mix id in one feature, then this code must be modified. 

  int n_desc = params.getDnnInfo().n_desc;

  // collect samples with same id
  step2_out.id_to_features.clear();
  for (const auto& f: step1_out.detected_features){
    step2_out.id_to_features[f.step1.id].push_back(f);
    step2_out.id_to_features[f.step1.id].back().step2.seq_dnn 
      = step2_out.id_to_features[f.step1.id].size() - 1;
  }

  step2_out.id_to_is_valid.clear();
  for (int id = 0 ; id < n_desc; ++id){
    step2_out.id_to_is_valid[id] = false;
  }
  for (int id = 0 ; id < n_desc; ++id){ 
    // current implementation assumpt that if marker has n_features, then
    // allowed min number of detected points from network is max(n_feature//2+1, 4)
    // and # of features in marker must be even.
    // thus, max # allowed occluded pts is (n_feature_in_marker - min_pts_required)
    // e.g.1: marker with 4 feature -> 4 points need to be detected from step1, 0 pts can be occluded
    // e.g.2: marker with 6 feature -> 4 points need to be detected from step1, 2 pts can be occluded

    int n_feature_in_marker = N_KEYPOINT_IN_MAKRER;
    if (n_feature_in_marker != 4 && n_feature_in_marker != 6) 
      my_error("not implemented for n_feature != 4, 6");
    if (step2_out.id_to_features[id].size() < N_MIN_REQUIRED_PTS){
      string msg = "network detected too few features with id " + std::to_string(id) 
        + " (required: " + std::to_string(N_MIN_REQUIRED_PTS) + ", detected: " 
        + std::to_string(step2_out.id_to_features[id].size()) + ")";
      debug_detail.updateLastTriedStep(msg);
      step2_out.id_to_is_valid[id] = false;
      continue;
    }

    auto& id_to_features = step2_out.id_to_features[id];

    /* get undistorted points */
    // collect distorted points
    vector<cv::Point2f> distorted_points;
    for (const auto& f: id_to_features){
      distorted_points.push_back(f.step1.feature_center_subpix);
    }

    // undistort points
    vector<cv::Point2f> undistorted_points;
    cv::undistortPoints(
      distorted_points, 
      undistorted_points, 
      params.getCameraInfo().K, 
      params.getCameraInfo().D,
      cv::noArray(),
      params.getCameraInfo().new_K
    );
    for (auto& f: id_to_features){
      f.step2.feature_center_subpix_undistorted 
        = undistorted_points[f.step2.seq_dnn];
    }

    // create distance map between each features
    step2_out.id_to_undistorted_pts_distmap.clear();

    int n_detected_features = id_to_features.size();
    if (n_detected_features > MAX_ALLOWED_N_FEATURE_CANDIDATES) {
      string msg = "network detected too many features with id " + std::to_string(id) 
        + " (detected: " + std::to_string(n_detected_features) + ", " 
        + "MAX_ALLOWED_N_FEATURE_CANDIDATES: " 
        + std::to_string(MAX_ALLOWED_N_FEATURE_CANDIDATES) + ")";
      debug_detail.updateLastTriedStep(msg);
      step2_out.id_to_is_valid[id] = false;
      continue;
    }
    step2_out.id_to_undistorted_pts_distmap[id].resize(n_detected_features*n_detected_features + 1);
    for (int i = 0 ; i < n_detected_features; ++i){
      for (int j = i; j < n_detected_features; ++j){
        float dist = cv::norm(
          step2_out.id_to_features[id][i].step2.feature_center_subpix_undistorted
          - step2_out.id_to_features[id][j].step2.feature_center_subpix_undistorted
        );
        step2_out.id_to_undistorted_pts_distmap[id][i*n_detected_features + j] = dist;
        step2_out.id_to_undistorted_pts_distmap[id][j*n_detected_features + i] = dist;
      }
    }
    step2_out.id_to_undistorted_pts_distmap[id].back() = n_detected_features;

    /* get all seq_dnn combinations with n(pts) == N_MIN_REQUIRED_PTS*/

    found_marker = false; 

    vector<std::thread> ts;
    int n_max_thread = params.getAlgoParams().max_available_threads;
    // so far(2023.11.23), no computer has >500 threads....
    // we'll use above array til ~(n_max_thread-1) index
    for (int i = 0 ; i < n_max_thread; ++i)
      detail_ary[i].clear();
    
    const auto& combination_helper_vec2d 
      = combination_helper_vec3d[n_detected_features];
    for (int i = 0 ; i < n_max_thread; ++i)
    {
      int check_start = i * combination_helper_vec2d.size() / n_max_thread;
      int check_end = (i+1) * combination_helper_vec2d.size() / n_max_thread;
      if ((check_end - check_start) == 0) continue;
      // abort creating thread if already found regular shape
      {
        std::lock_guard<std::mutex> lock(step2_check_if_regular_mtx);
        if (true == found_marker) break;
      }

      // create thread input
      info_ary[i].params_ptr = &params;
      info_ary[i].id_to_features_ptr = &step2_out.id_to_features;
      info_ary[i].combinations_helper_vec2d_ptr = &combination_helper_vec2d;
      info_ary[i].helper_start_idx = check_start;
      info_ary[i].helper_end_idx = check_end;
      info_ary[i].id = id;

      // std::thread t(&Paacman::step2_check_if_regular, this, 
      std::thread t(&Paacman::step2_check_if_equilateral, this, 
        &img, info_ary + i, detail_ary + i);

      ts.push_back(std::move(t));

      if (check_end >= (int)combination_helper_vec2d.size()) break;
    }
    for (auto& t : ts) t.join();


    // check if marker is found 
    if (false == found_marker){ // no need to get mtx here, since only one thread can be here
      string msg = "no marker found (id: " + std::to_string(id) + ")";
      debug_detail.updateLastTriedStep(msg);
      continue;
    }
    
    CheckDetail* detail_ptr = nullptr;
    for (int i = 0 ; i < n_max_thread; ++i){
      if (true == detail_ary[i].success){
        detail_ptr = detail_ary + i;
        break;
      }
    }// in multithreaded env, there can be many solutions, but we just pick first one

    // organize results
    if (nullptr == detail_ptr) my_error("nullptr == detail_ptr");
    step2_out.id_to_detected_markers[id] = detail_ptr->toDetectedMarker(id);
    step2_out.id_to_is_valid[id] = true; 
  }

  for (const auto& valid: step2_out.id_to_is_valid)
    if (true == valid.second) return true;
  return false;
}

void Paacman::step2_check_if_equilateral
  (
  // called internally by step2_shape_check()
    const cv::Mat* img_ptr,
    const CheckInfo* info_ptr,
    CheckDetail* detail_ptr
  )
{
  const EquilateralShapeCheckInfo* l_and_dir_info_ptr =
    dynamic_cast<const EquilateralShapeCheckInfo*>(info_ptr);
  LabellingAndDirectionCheckDetail * l_and_dir_check_detail_ptr =
    dynamic_cast<LabellingAndDirectionCheckDetail*>(detail_ptr);
  l_and_dir_check_detail_ptr->success = false;

  // unpack members
  const PaacmanParameters& params = *(l_and_dir_info_ptr->params_ptr);
  const map<int, vector<DetectedFeature>>& id_to_features = *(l_and_dir_info_ptr->id_to_features_ptr);
  const vector<vector<int>>& combinations_helper_vec2d = *(l_and_dir_info_ptr->combinations_helper_vec2d_ptr);
  const int helper_start_idx = l_and_dir_info_ptr->helper_start_idx;
  const int helper_end_idx = l_and_dir_info_ptr->helper_end_idx;
  const int id = l_and_dir_info_ptr->id;

  const auto& m_name = params.getMarkerNames().at(id);
  auto marker_config = params.getMarkerConfigs().at(m_name); //copy

  // this algorithms runs only when feature forms regular hexagon.
  // otherwise, it breaks
  int n_feature = N_KEYPOINT_IN_MAKRER;
  if (6 != n_feature){
    my_error("not implemented for n_feature != 6");
  }

  for (int helper_idx = helper_start_idx;
    helper_idx < helper_end_idx; ++helper_idx)
  {
    {
      std::lock_guard<std::mutex> lock(step2_check_if_regular_mtx);
      if (true == found_marker) break;
    }

    const auto& comb = combinations_helper_vec2d.at(helper_idx);
    
    // assemble pts
    vector<cv::Point2f> pts;
    for (size_t i = 0 ; i < comb.size(); ++i){
      int deq_dnn = comb[i];
      const cv::Point2f& pt = 
        id_to_features.at(id).at(deq_dnn).step2.feature_center_subpix_undistorted;
      pts.push_back(pt);
    }

    // run convexhull to check nested pts and arrange in clockwise order
    vector<int> pt_indicies_in_clockwise;
    cv::convexHull(pts, pt_indicies_in_clockwise, false, false);
    if (pts.size() > pt_indicies_in_clockwise.size()){ // check nested
      continue;
    }
    vector<cv::Point2f> pts_clockwise;
    vector<int> seq_dnn_in_clockwise;
    for (int idx: pt_indicies_in_clockwise){
      pts_clockwise.push_back(pts[idx]);
      seq_dnn_in_clockwise.push_back(comb[idx]);
    }

    /* label each point considering area */
    // get intersection of square diagonals
    cv::Point2f diag_intersection;
    {
      const auto& pt1 = pts_clockwise[0];
      const auto& pt2 = pts_clockwise[2];
      const auto& pt3 = pts_clockwise[1];
      const auto& pt4 = pts_clockwise[3];

      float A1 = pt2.y - pt1.y;
      float B1 = pt1.x - pt2.x;
      float C1 = A1*pt1.x + B1*pt1.y;

      float A2 = pt4.y - pt3.y;
      float B2 = pt3.x - pt4.x;
      float C2 = A2*pt3.x + B2*pt3.y;

      float det = A1*B2 - A2*B1;
      if (std::fabs(det) < 0.00001){
        my_error("det == 0");
      }
      diag_intersection.x = (B2*C1 - B1*C2) / det;
      diag_intersection.y = (A1*C2 - A2*C1) / det;
    }

    vector<float> dists_from_center;
    // calc dist from center to each
    for (int ii = 0 ; ii < N_MIN_REQUIRED_PTS; ++ii){
      float&& dist = cv::norm(pts_clockwise[ii] - diag_intersection);
      dists_from_center.emplace_back(dist);
    }
    // calc area scale (not exact area)
    vector<float> areas_clockwise;
    for (int ii = 0 ; ii < N_MIN_REQUIRED_PTS; ++ii){
      int ii_n = (ii + 1) % N_MIN_REQUIRED_PTS;
      float&& area = dists_from_center[ii]*dists_from_center[ii_n];     
      areas_clockwise.emplace_back(area);
    }

    // calc min/max area rate
    float min = params.getCameraInfo().width * params.getCameraInfo().height;
    float max = -1;
    int big_area_start_idx = -1; // do not touch...
    for (int ii = 0 ; ii < N_MIN_REQUIRED_PTS; ++ii){
      if (areas_clockwise[ii] < min) min = areas_clockwise[ii];
      if (areas_clockwise[ii] > max) {
        max = areas_clockwise[ii];
        big_area_start_idx = ii;
      }
    }
    
    float area_rate = min / max;
    int labelling_type;
    if (area_rate < params.getAlgoParams().step2_labelling_thresh1) 
      labelling_type = LABELLING_NOT_REQUIRED; // big distortion
    else if (area_rate < params.getAlgoParams().step2_labelling_thresh2) {
      // need to find 2nd biggest area to clarify labelling type
      int big_2nd_area_start_idx = -1;
      float max_2nd = -1;

      for (int ii = 0 ; ii < N_MIN_REQUIRED_PTS; ++ii){
        if (ii == big_area_start_idx) continue;
        if (areas_clockwise[ii] > max_2nd) {
          max_2nd = areas_clockwise[ii];
          big_2nd_area_start_idx = ii;
        }
      }

      int dist = dist_clock_a_b(big_area_start_idx, big_2nd_area_start_idx, N_MIN_REQUIRED_PTS);
      if (dist == 1) // 2nd big area in prev index
        labelling_type= LABELLING_TYPE_0_PREV;
      else if (dist == 3)
        labelling_type = LABELLING_TYPE_0_NEXT;
      else {
        my_error("invalid big_area_start_idx - big_2nd_area_start_idx");
      }
    }
    else labelling_type = LABELLING_TYPE_1;
    
    // abandon if labelling is not required
    if (labelling_type == LABELLING_NOT_REQUIRED) continue;

    // equilateral shape check
    vector<LabellingAndDirectionCheckDetail::LabellingDetail> 
      labelling_details(label_type_to_pseudo_labels[labelling_type].size());
    
    // fill labelling details
    for (size_t label_cand_idx = 0 ; 
        label_cand_idx < label_type_to_pseudo_labels[labelling_type].size(); 
        ++label_cand_idx)
    {
      // infer pseudo labels
      const vector<int>& labels = label_type_to_pseudo_labels[labelling_type][label_cand_idx];
      vector<int> pseudo_label_clockwise(N_MIN_REQUIRED_PTS);
      for (int iii = 0; iii < N_MIN_REQUIRED_PTS; ++iii){
        int idx_cur = (big_area_start_idx + iii) % N_MIN_REQUIRED_PTS;
        pseudo_label_clockwise[idx_cur] = labels[iii];
      }

      // infer abc of right triangle
      auto right_triangle = label_type_to_right_triangle[labelling_type][label_cand_idx];
      for (auto& idx: right_triangle){
        idx = (big_area_start_idx + idx) % N_MIN_REQUIRED_PTS;
      }

      labelling_details[label_cand_idx].pseudo_labels = pseudo_label_clockwise;
      labelling_details[label_cand_idx].pts_clockwise_undistorted 
        = pts_clockwise;
      // collect pts_distorted
      for (int ii = 0 ; ii < N_MIN_REQUIRED_PTS; ++ii){
        int seq_dnn = seq_dnn_in_clockwise[ii];
        labelling_details[label_cand_idx].pts_clockwise_distorted.emplace_back(
          id_to_features.at(id).at(seq_dnn).step1.feature_center_subpix
        );
      }
      // center
      labelling_details[label_cand_idx].hexagon_center_undistorted
        = (pts_clockwise[right_triangle[0]] + pts_clockwise[right_triangle[1]]) / 2;
      
      // if (params.getDebugInfo().enable_debug){
      //   cout << "right triangle..." << endl;
      //   cout << right_triangle[0] << endl;
      //   cout << right_triangle[1] << endl;

      //   cout << "pseudo labels - " << endl;
      //   for (auto p: pseudo_label_clockwise)
      //     cout << p << ' ';
      //   cout << endl;
      // }

      // debug....
      if (true == params.getDebugInfo().enable_debug){
        if (true == params.getDebugInfo().enable_debug_combinatorial_optimization){
          debug_detail.step2_visualizeDiagIntersection(
            params.getCameraInfo().width, params.getCameraInfo().height, 
            pts_clockwise, labelling_details[label_cand_idx].hexagon_center_undistorted,
            area_rate, labelling_type, pseudo_label_clockwise
          );
        }
      }
    }// labelling_details

    constexpr static int max_regular_shape_details = 2;
    vector<LabellingAndDirectionCheckDetail::DirectionCheckDetail> 
      direction_check_results(max_regular_shape_details);
    for (auto& d: direction_check_results) d.success = false;

    for (int idx_regular_shape_detail = 0; 
      idx_regular_shape_detail < max_regular_shape_details;
      idx_regular_shape_detail++)
    {
      {
        std::lock_guard<std::mutex> lock(step2_check_if_regular_mtx);
        if (true == found_marker) break;
      }

      const auto& labelling_detail = labelling_details[idx_regular_shape_detail];

      LabellingAndDirectionCheckDetail::DirectionCheckDetail direction_check_detail;

      /* check radius of hexagon is long enough*/
      // enumerate all possible radiuses in image and check all >= min_radius
      float radius_over_hexagon_radius = 
        marker_config.keypoint_circle_rad_over_hexagon_rad_;
      vector<float> rad_pixel_cands;

      for (size_t pt_idx = 0; 
        pt_idx < labelling_detail.pts_clockwise_distorted.size(); ++pt_idx)
      {
        const auto& pt1 = labelling_detail.pts_clockwise_distorted[pt_idx];
        for (size_t pt_idx2 = 1; 
          pt_idx2 < labelling_detail.pts_clockwise_distorted.size(); ++pt_idx2)
        {
          const auto& pt2 = labelling_detail.pts_clockwise_distorted[pt_idx2];

          int label_diff = 
            dist_clock_a_b(labelling_detail.pseudo_labels[pt_idx2], 
              labelling_detail.pseudo_labels[pt_idx], n_feature);
          float line_f1_to_f2_2d = cv::norm(pt1 - pt2);
          float label_diff_to_line_factor = 
            clock_dist_diff_to_length_3d_ratio[label_diff];
          float radius_pixel = line_f1_to_f2_2d / label_diff_to_line_factor 
            * radius_over_hexagon_radius;
          rad_pixel_cands.push_back(radius_pixel);
        }
      }

      direction_check_detail.min_radius_pixel_img = rad_pixel_cands[0];
      for (size_t rad_idx = 1; rad_idx < rad_pixel_cands.size(); ++rad_idx){
        if (rad_pixel_cands[rad_idx] < direction_check_detail.min_radius_pixel_img)
          direction_check_detail.min_radius_pixel_img = rad_pixel_cands[rad_idx];
      }

      int idx_rad_possible = -1;
      const float RADIUS_DECREASING_FACTOR=0.9;
      for (int idx_rad = 2; idx_rad >= 0; idx_rad--){
        if (direction_check_detail.min_radius_pixel_img*RADIUS_DECREASING_FACTOR 
          >= STEP2_MIN_RADIUS_PIXELS[idx_rad] - 0.5)
        {
          idx_rad_possible = idx_rad;
          break;
        }
      }
      if (idx_rad_possible==-1)
      {
        if (params.getDebugInfo().enable_debug)
          cout << "too small rad" << endl;
        continue;
      }

      /* get keypoint patch descriptor */
      // find direction angle
      int n_descriptor_length = marker_config.keypoint_binary_descriptors.size(idx_rad_possible);
      vector<KeypointBinaryDescriptors> keypoint_descriptor_targets(N_MIN_REQUIRED_PTS);
      vector<vector<int>> intensities_vec;
      for (int i = 0 ; i < N_MIN_REQUIRED_PTS; ++i){
        keypoint_descriptor_targets[i].reset();
        intensities_vec.emplace_back(n_descriptor_length, 0);
      }

      // calc white area angle ratio
      vector<int> k_descripting_discretization_error_pts;
      bool is_pt_out_of_image = false;
      for (size_t pt_idx = 0; 
        pt_idx < labelling_detail.pts_clockwise_distorted.size(); ++pt_idx){
          
        // get intensities
        const float& pt_x = labelling_detail.pts_clockwise_distorted[pt_idx].x;
        const float& pt_y = labelling_detail.pts_clockwise_distorted[pt_idx].y;

        vector<int> hist(256, 0); // find binarized intensity
        for (int circle_pt_idx = 0; 
          circle_pt_idx < n_descriptor_length; ++circle_pt_idx){
          auto dxdy = marker_config.keypoint_descriptor_dxdys_vec_[idx_rad_possible][circle_pt_idx];

          int x = pt_x + dxdy.x;
          int y = pt_y + dxdy.y;

          if (false == is_pt_in_image(x, y, img_ptr)){
            is_pt_out_of_image = true;
            break;
          }
          int intensity = img_ptr->at<uchar>(y, x);
          intensities_vec[pt_idx][circle_pt_idx] = intensity;
          hist[intensity]++;
        }
        if (is_pt_out_of_image) {
          if (params.getDebugInfo().enable_debug)
            cout << "is_pt_out_of_image" << endl;
          break;
        }

        // calc cumulated hist
        vector<int> cum_hist(256, 0);
        cum_hist[0] = hist[0];
        int intensity_threshold = 0;
        const int n_intensity_threshold_pts = 
          n_descriptor_length * (1-marker_config.white_area_ratio_);
        for (int intensity = 1 ; intensity < 256; ++intensity){
          cum_hist[intensity] = cum_hist[intensity-1] + hist[intensity];
          if (cum_hist[intensity] >= n_intensity_threshold_pts){
            intensity_threshold = intensity;
            k_descripting_discretization_error_pts.push_back(hist[intensity]);
            break;
          }
        }

        // calc binarized descriptor
        for (int circle_pt_idx = 0;
          circle_pt_idx < (int)n_descriptor_length; ++circle_pt_idx){
          keypoint_descriptor_targets[pt_idx].set(idx_rad_possible, circle_pt_idx, 
            intensities_vec[pt_idx][circle_pt_idx] >= intensity_threshold
          );
        }//circle_pt_idx
      }//pt_idx

      if (is_pt_out_of_image) continue;

      // iterate all descriptors and calc hamming distance
      for (int idx_descriptor = 0; idx_descriptor < N_MIN_REQUIRED_PTS; ++idx_descriptor){
        float min_hamming_dist = n_descriptor_length;

        vector<int> best_matched_desc_indicies;
        for (int idx_shifted = 0; idx_shifted < n_descriptor_length; ++idx_shifted)
        {
          int hamming_dist;
          if (idx_rad_possible == 0){
            auto&& hamming_dist_bitset = keypoint_descriptor_targets[idx_descriptor].getDescriptor0() ^
              marker_config.keypoint_binary_descriptor_shifted.getBitsetVec0()[idx_shifted];
            hamming_dist = hamming_dist_bitset.count();
            // cout << hamming_dist << ' ' << endl;
            // cout << keypoint_descriptor_targets[idx_descriptor].getDescriptor0() << endl;
            // cout << marker_config.keypoint_binary_descriptor_shifted.getBitsetVec0()[idx_shifted] << endl;
          }
          else if (idx_rad_possible == 1){
            auto&& hamming_dist_bitset = keypoint_descriptor_targets[idx_descriptor].getDescriptor1() ^
              marker_config.keypoint_binary_descriptor_shifted.getBitsetVec1()[idx_shifted];
            hamming_dist = hamming_dist_bitset.count();
            // cout << hamming_dist << ' ' << endl;
            // cout << keypoint_descriptor_targets[idx_descriptor].getDescriptor1() << endl;
            // cout << marker_config.keypoint_binary_descriptor_shifted.getBitsetVec1()[idx_shifted] << endl;
          }
          else if (idx_rad_possible == 2){
            auto&& hamming_dist_bitset = keypoint_descriptor_targets[idx_descriptor].getDescriptor2() ^
              marker_config.keypoint_binary_descriptor_shifted.getBitsetVec2()[idx_shifted];
            hamming_dist = hamming_dist_bitset.count();
            // cout << hamming_dist << ' ' << endl;
            // cout << keypoint_descriptor_targets[idx_descriptor].getDescriptor2() << endl;
            // cout << marker_config.keypoint_binary_descriptor_shifted.getBitsetVec2()[idx_shifted] << endl;
          }
          else my_error("invalid idx_rad_possible");

          if (hamming_dist == min_hamming_dist){
            best_matched_desc_indicies.push_back(idx_shifted);
          }
          else if (hamming_dist < min_hamming_dist){
            min_hamming_dist = hamming_dist;
            best_matched_desc_indicies.resize(0);
            best_matched_desc_indicies.push_back(idx_shifted);
          }
        }

        // pick up median bitarray index
        int median_bitarray_idx;
        if (best_matched_desc_indicies.size() == 1) 
          median_bitarray_idx = best_matched_desc_indicies[0];
        else
          median_bitarray_idx = best_matched_desc_indicies[best_matched_desc_indicies.size()/2];

        // pick up orientation
        float rotated_angle = marker_config.dedicated_angles_vec[idx_rad_possible][median_bitarray_idx];
        direction_check_detail.directions_clockwise_degree.push_back(rotated_angle);
      }

      if (params.getDebugInfo().enable_debug){
        // cout << "rotated angles: ";
        // for (auto r: direction_check_detail.directions_clockwise_degree) cout << r << ' ';
        // cout << endl;
      }

      // find median. This is considered as rotated yaw of marker
      std::vector<float> rotated_angles_deg_sorted 
        = direction_check_detail.directions_clockwise_degree;
      std::sort(rotated_angles_deg_sorted.begin(),
        rotated_angles_deg_sorted.end());
      float median_angle_deg
        = rotated_angles_deg_sorted[rotated_angles_deg_sorted.size() / 2];
      
      // calc error btw median and check invalidity
      bool is_descriptor_different = false;
      for(auto angle_deg: rotated_angles_deg_sorted){
        if (std::isnan(angle_deg)) {
          is_descriptor_different = true;
          break;
        }
        float error = std::fabs(median_angle_deg - angle_deg);
        if (error > 180) error = std::fabs(360 - error);
        if (error > 
          params.getAlgoParams().step2_feature_direction_angle_error_thresh){
          is_descriptor_different = true;
        }
      }

      if (true == params.getDebugInfo().enable_debug){
        if (true == params.getDebugInfo().enable_debug_combinatorial_optimization){
          debug_detail.step2_visualizeDirectionCheckSimple
          (
            img_ptr, params, labelling_detail.pts_clockwise_undistorted, 
            STEP2_MIN_RADIUS_PIXELS[idx_rad_possible], 
            direction_check_detail.directions_clockwise_degree,
            id
          );
        }
      }
      if (is_descriptor_different) {
        if (params.getDebugInfo().enable_debug){
          cout << "is_descriptor_different, rotated angles: ";
          for (auto r: direction_check_detail.directions_clockwise_degree) cout << r << ' ';
          cout << endl;
        }
        continue;
      }
      
      // get enu angle of pts_clockwise_undistorted[0]
      const cv::Point2f& center_2d_undistorted 
        = labelling_detail.hexagon_center_undistorted; 

      cv::Point2f center_to_tgt_pt = 
        labelling_detail.pts_clockwise_undistorted[0] - center_2d_undistorted;
      float angle_center_to_tgt_pt = 
        std::atan2(center_to_tgt_pt.y, center_to_tgt_pt.x) * 180 / M_PI;
      float predefined_label_1st_pt 
        = std::roundf((angle_center_to_tgt_pt - median_angle_deg + marker_config.keypoint0_yaw_deg_) /60.f);
      if (predefined_label_1st_pt < 0) predefined_label_1st_pt += n_feature;
      int d_label = predefined_label_1st_pt - labelling_detail.pseudo_labels[0];

      // if (params.getDebugInfo().enable_debug){
      //   cout << "angle_center_to_tgt_pt: " << angle_center_to_tgt_pt << endl;
      //   cout << "median_angle_deg: " << median_angle_deg << endl;
      //   cout << "pseudo_label_1st_pt: " << labelling_detail.pseudo_labels[0] << endl;
      //   cout << "predefined_label_1st_pt: " << predefined_label_1st_pt << endl;
      // }

      direction_check_detail.labels.resize(N_MIN_REQUIRED_PTS);
      for (size_t i = 0; i < N_MIN_REQUIRED_PTS; ++i){
        int predefined_label = labelling_detail.pseudo_labels[i] + d_label;
        if (predefined_label < 0) predefined_label += n_feature;
        predefined_label %= n_feature;
        direction_check_detail.labels[i] = predefined_label;
      }

      // now we can do solvePnP
      vector<cv::Point3f> pts3d;
      for (size_t i = 0; i < N_MIN_REQUIRED_PTS; ++i){
        const int& predefined_label = direction_check_detail.labels[i];
        pts3d.push_back(marker_config.label_to_pts3d[predefined_label]);
      }
    
      cv::Mat rvec, tvec;
      cv::solvePnP(pts3d, labelling_detail.pts_clockwise_undistorted, 
        params.getCameraInfo().new_K, cv::noArray(),
        rvec, tvec);

      /* so far, there can be 2 solutions with wrong label */
      /* we'll cut one by comparing yaw from image and pnp */
      // calc yaw from pnp
      float yaw_pnp = 0;;
      cv::Mat rotationMatrix;
      cv::Rodrigues(rvec, rotationMatrix);

      float sy = sqrt(rotationMatrix.at<double>(0,0) * rotationMatrix.at<double>(0,0) + 
                    rotationMatrix.at<double>(1,0) * rotationMatrix.at<double>(1,0));
      bool singular = sy < 1e-6;
      if (false == singular) {
        yaw_pnp = atan2(rotationMatrix.at<double>(1,0), rotationMatrix.at<double>(0,0));
      }
      else {
        yaw_pnp = 0;
      }

      float yaw_pnp_enu = -yaw_pnp * 180/M_PI;
      yaw_pnp_enu = yaw_pnp_enu + 360;
      if (yaw_pnp_enu > 360) yaw_pnp_enu -= 360;

      bool valid_dir = true;

      // check 1 - yaw error must be in threshold
      float yaw_marker_enu = -median_angle_deg;
      float img_and_pnp_yawdiff_error = std::fabs(yaw_pnp_enu - yaw_marker_enu);
      if (img_and_pnp_yawdiff_error > 360) img_and_pnp_yawdiff_error -= 360;
      float img_and_pnp_yawdiff_error2 = std::fabs(img_and_pnp_yawdiff_error - 360);
      if (img_and_pnp_yawdiff_error > img_and_pnp_yawdiff_error2)
        img_and_pnp_yawdiff_error = img_and_pnp_yawdiff_error2;
      if (img_and_pnp_yawdiff_error > 
        params.getAlgoParams().step2_img_and_pnp_yawdiff_error_thresh){
        if (params.getDebugInfo().enable_debug)
          cout << "yawdiff error is high: " << img_and_pnp_yawdiff_error << endl;
        valid_dir = false;
      }
      

      // check 2 - calc reprojection error (we ignore D term of camera matrix)
      cv::Mat R;
      cv::Rodrigues(rvec, R);
      cv::Mat T = tvec;
      cv::Mat RT = cv::Mat::zeros(3, 4, CV_32FC1);
      R.copyTo(RT(cv::Rect(0, 0, 3, 3)));
      T.copyTo(RT(cv::Rect(3, 0, 1, 3)));
      
      float reprojection_error_avg = 0;
      float reprojection_error_max = -1;
      vector<cv::Point2f> reprojected_pts;
      for (int pt_idx = 0; pt_idx < N_MIN_REQUIRED_PTS; ++pt_idx){
        const auto& label = direction_check_detail.labels[pt_idx];

        const auto pt = marker_config.label_to_pts3d[label];
        cv::Mat pt3d = cv::Mat::zeros(4, 1, CV_32FC1);
        pt3d.at<float>(0, 0) = pt.x;
        pt3d.at<float>(1, 0) = pt.y;
        pt3d.at<float>(2, 0) = pt.z;
        pt3d.at<float>(3, 0) = 1.0;

        //get reprojected point
        cv::Mat pt2d = params.getCameraInfo().new_K * RT * pt3d;
        pt2d /= pt2d.at<float>(2, 0);
        cv::Point2f pt2d_cv(pt2d.at<float>(0, 0), pt2d.at<float>(1, 0));
        
        // calc reprojection error
        const auto& pt_from_img 
          = labelling_detail.pts_clockwise_undistorted[pt_idx];
        float reprojection_error =
          cv::norm(pt2d_cv - pt_from_img);
        reprojection_error_avg += reprojection_error;
        if (reprojection_error_max < reprojection_error)
          reprojection_error_max = reprojection_error;
        reprojected_pts.push_back(pt2d_cv);
      }
      reprojection_error_avg /= N_MIN_REQUIRED_PTS;

      if (true == params.getDebugInfo().enable_debug){
        if (true == params.getDebugInfo().enable_debug_combinatorial_optimization){
          debug_detail.step2_visualizeDirectionCheck
          (
            img_ptr, 
            params, 
            labelling_detail.pts_clockwise_undistorted, 
            direction_check_detail.labels,
            STEP2_MIN_RADIUS_PIXELS[idx_rad_possible], 
            direction_check_detail.directions_clockwise_degree,
            rvec, 
            tvec,
            reprojected_pts,
            id          
          );
        }
      }

      if (false == valid_dir) continue;
      if (reprojection_error_max > 
        params.getAlgoParams().step2_max_reprojection_error_pixel_thresh){
        if (params.getDebugInfo().enable_debug)
          cout << "big reprojection error: " << reprojection_error_max << endl;
        continue;
      }

      direction_check_detail.tvec = tvec;
      direction_check_detail.rvec = rvec;
      direction_check_detail.success = true;
      direction_check_detail.reprojection_error = reprojection_error_avg;
      direction_check_detail.yawdiff_error = img_and_pnp_yawdiff_error;
      direction_check_detail.l_detail_ptr 
        = &labelling_details[idx_regular_shape_detail];
      direction_check_results[idx_regular_shape_detail] = direction_check_detail;

    } //idx_regular_shape_detail

    {
      std::lock_guard<std::mutex> lock(step2_check_if_regular_mtx);
      if (true == found_marker) break;
    }

    // if 2 candidate exists, choose lower reprojection error one
    // filter successed results
    vector<int> correct_detail_indicies;
    int idx = 0;
    for (auto& direction_check_detail: direction_check_results){
      if (direction_check_detail.success){
        correct_detail_indicies.push_back(idx);
      }
      idx++;
    }
    if (0 == correct_detail_indicies.size()) continue;

    if (params.getDebugInfo().enable_debug){
      cout << "size of correct_detail_indicies: " << correct_detail_indicies.size() << endl;
    }
    
    //calculate score to find best results
    int best_detail_idx = correct_detail_indicies[0];

    if (correct_detail_indicies.size() > 1){ 
      vector<double> scores(correct_detail_indicies.size());
      // reprojection error, yawdiff error
      for (size_t i = 0; i < correct_detail_indicies.size(); ++i){
        int detail_idx = correct_detail_indicies[i];
        double score = 0;
        score += -direction_check_results[detail_idx].reprojection_error;
        // score += -direction_check_results[detail_idx].yawdiff_error;
        if (params.getDebugInfo().enable_debug){
          cout << "error of reproj, yawdiff: " << 
            direction_check_results[detail_idx].reprojection_error << ' ' <<
            direction_check_results[detail_idx].yawdiff_error << endl;
        }
        scores[i] = score;
      }
      int best_score_idx = std::max_element(scores.begin(), scores.end()) - scores.begin();
      best_detail_idx = correct_detail_indicies[best_score_idx];
    }

    step2_check_if_regular_mtx.lock();
    if (true == found_marker){ // prevent overwrite by other thread
      step2_check_if_regular_mtx.unlock();
      break;
    }

    found_marker = true;
    l_and_dir_check_detail_ptr->success = true;
    l_and_dir_check_detail_ptr->labelling_detail 
      = *direction_check_results[best_detail_idx].l_detail_ptr;
    l_and_dir_check_detail_ptr->direction_check_detail 
      = direction_check_results[best_detail_idx];
    l_and_dir_check_detail_ptr->direction_check_detail.l_detail_ptr = nullptr; // cleaning memory
    step2_check_if_regular_mtx.unlock();
    break;
  } // helper idx
}

vector<DetectedFeature> Paacman::detectKeypoints(const cv::Mat &img_input){
  static STEP1_OUT step1_out;

  step1_out.success = false;

  // precheck
  if (img_input.empty()) {
    debug_detail.updateLastTriedStep("image is empty", STEP1);
    return {};
  }

  // to gray
  cv::Mat img;
  if (img_input.channels() == 3){
    cv::cvtColor(img_input, img, cv::COLOR_BGR2GRAY);
  } else img = img_input;

  int H = getParams().getCameraInfo().height;
  int W = getParams().getCameraInfo().width;
  if (img.cols != W || img.rows != H)
    cv::resize(img, img, cv::Size(W, H));

  // run utilities
  runUtilities(STEP_ZERO, &img);

  /* step1: network inference */
  step1_out.success = step1_network_inference(img, step1_out);
  runUtilities(STEP1, &img, &step1_out);

  return step1_out.detected_features;
}


bool Paacman::detectMarkers(const cv::Mat &img_input, 
  map<int, DetectedMarker> &id_to_detected_markers){
  static STEP1_OUT step1_out;
  static STEP2_OUT step2_out;
  step2_out.id_to_detected_markers.clear();

  step1_out.success = false;
  step2_out.success = false;


  // precheck
  if (img_input.empty()) {
    debug_detail.updateLastTriedStep("image is empty", STEP1);
    return false;
  }

  // to gray
  cv::Mat img;
  if (img_input.channels() == 3){
    cv::cvtColor(img_input, img, cv::COLOR_BGR2GRAY);
  } else img = img_input;

  int H = getParams().getCameraInfo().height;
  int W = getParams().getCameraInfo().width;
  if (img.cols != W || img.rows != H)
    cv::resize(img, img, cv::Size(W, H));

  // run utilities
  runUtilities(STEP_ZERO, &img);

  /* step1: network inference */
  step1_out.success = step1_network_inference(img, step1_out);
  runUtilities(STEP1, &img, &step1_out);

  /* step2 */
  step2_out.success = step2_shape_check(img, step1_out, step2_out);
  runUtilities(STEP2, &img, &step2_out);

  if (step2_out.success) id_to_detected_markers = step2_out.id_to_detected_markers;
  return step2_out.success;
}

Paacman& Paacman::setTimeLogging(bool enable){
  enable_time_logging = enable;
  if (true == enable_time_logging){
    time_logger.setLoggingFile(
      params.getDebugInfo().time_log_csv,
      params.getDebugInfo().time_log_flush_seq
    );
    time_logger.setDoPrint(
      params.getDebugInfo().time_log_do_print
    );
  }
  return *this;
}

Paacman& Paacman::setDebugging(bool enable){
  enable_debugging = enable;
  if (true == enable_debugging){
    debug_detail.setDebugDir(
      params.getDebugInfo().debug_image_dir_root,
      params.getDebugInfo().debug_image_dirs_map,
      params.getDebugInfo().clear_debug_image_dirs
    );
    debug_detail.setDebugParams(params.getDebugInfo());
  }
  return *this;
}

void Paacman::runUtilities(int step, const cv::Mat* img_ptr, STEP_OUT* step_out_ptr, bool print_separator){
  //step n: arrange results of step (n) and initialize for step (n+1)
  if (step == STEP_ZERO){ // initialize utilities
    seq++;
    if (print_separator)
      cout << "======================================" << endl;

    if (true == enable_debugging){
      debug_detail.clear();
      debug_detail.saveImage(img_ptr, STEP_ZERO, "original", seq);
      debug_detail.saveImageRectified(img_ptr, STEP_ZERO, "rectified", seq, params);
    }

    // time_logger must run at last for timing
    if (true == enable_time_logging){
      time_logger.initNewTiming();
      time_logger.tic();
    }
  }
  else if (step == STEP1){
    if ((nullptr == step_out_ptr) && step_out_ptr->step_type != STEP1){
      my_error("invalid step_out_ptr detected when step == STEP1");
    }
    STEP1_OUT& step1_out = *static_cast<STEP1_OUT*>(step_out_ptr);

    if (true == enable_time_logging){
      time_logger.toc(STEP1_STR, "step1_network_inference");
    }
    if (true == enable_debugging){
      if (false == step1_out.success){
        debug_detail.updateLastTriedStep("step1_network_inference() failed", STEP1);
      }
      // visualize if debugging is enabled
      debug_detail.step1_visualizeDetectedFeature(img_ptr, step1_out.detected_features, seq);
      debug_detail.step1_visualizeDecoderOut(
        img_ptr, 
        step1_out.net_output_vectors[0], 
        step1_out.detected_features, 
        seq
      );
      debug_detail.clear();
    }
    if (true == enable_time_logging){
      time_logger.tic();
    }
  }
  else if (step == STEP2){
    if ((nullptr == step_out_ptr) && step_out_ptr->step_type != STEP2){
      my_error("invalid step_out_ptr detected when step == STEP2");
    }
    STEP2_OUT& step2_out = *static_cast<STEP2_OUT*>(step_out_ptr);
    
    if (true == enable_time_logging){
      time_logger.toc(STEP2_STR,"step2_regular_shape_check");
    }
    if (true == enable_debugging){
      if (false == step2_out.success){
        // visualize if failed
        debug_detail.updateLastTriedStep("step2_shape_check() failed", STEP2);
        debug_detail.printTriedSteps();

        debug_detail.step2_visualizeUndistortedPoints(
          img_ptr, step2_out.id_to_features, step2_out.id_to_is_valid, params, seq);
      }
      debug_detail.clear();
    }
    if (true == enable_time_logging){
      if (seq == (int)params.getDebugInfo().time_log_flush_seq){
        time_logger.flush();
      }
    }
  }
  else my_error("invalid step");
}

bool Paacman::visualizeDetectedMarkers(
    const cv::Mat& in, cv::Mat& out, const map<int, DetectedMarker> &id_to_detected_markers,
    bool print_on_failure
  )
{
  if (in.empty()) {
    if (print_on_failure)
      cout << "empty img" << endl;
    return false;
  }
  
  //undistort image
  cv::undistort(in, out, 
    params.getCameraInfo().K, params.getCameraInfo().D, 
    params.getCameraInfo().new_K);
  if (out.channels() != 3)
    cv::cvtColor(out, out, cv::COLOR_GRAY2BGR);

  if (id_to_detected_markers.size() == 0){
    if (print_on_failure)
      cout << "no detected marker" << endl;
    return false;
  }
  
  for (const auto& [id, detected_marker]: id_to_detected_markers)
  {
    // draw feature in undistorted image
    for (const auto& pt : detected_marker.pts_clockwise_undistorted){
      int x_int = pt.x;
      int y_int = pt.y;

      auto c = params.getDebugInfo().id_to_color(id);
      out.at<cv::Vec3b>(y_int, x_int)[0] = c[0];
      out.at<cv::Vec3b>(y_int, x_int)[1] = c[1];
      out.at<cv::Vec3b>(y_int, x_int)[2] = c[2];
    }

    // draw circles
    for (const auto& pt : detected_marker.pts_clockwise_undistorted){
      int x_int = pt.x;
      int y_int = pt.y;

      cv::circle(out, cv::Point(x_int, y_int), 
        20,params.getDebugInfo().id_to_color(id), 10); //normal
        // 10,params.getDebugInfo().id_to_color(id), 3);
    }

    // write labels
    for (int i = 0 ; i < (int)detected_marker.pts_clockwise_undistorted.size(); ++i){
      const auto& pt = detected_marker.pts_clockwise_undistorted[i];
      int x_int = pt.x 
        + params.getDebugInfo().enclosed_circle_radius*1.2 //normal
         - 140;
      int y_int = pt.y
        - params.getDebugInfo().enclosed_circle_radius
         + 80;
      // if (detected_marker.labels[i] == 2) {
      //   x_int -= 20;
      //   y_int -= 10;
      // }
      // else if (detected_marker.labels[i] == 1) {
      //   x_int -= 20;
      //   y_int += 50;
      // }
      // else if (detected_marker.labels[i] == 0) {
      //   x_int -= 20;
      //   y_int += 50;
      // }

      int label = detected_marker.labels[i];
      // cv::putText(out, std::to_string(label), 
      //   cv::Point(x_int, y_int), cv::FONT_HERSHEY_SIMPLEX,
      //   params.getDebugInfo().font_scale, 
      //   params.getDebugInfo().id_to_color(id), 10); //normal
        // 1, 
        // params.getDebugInfo().id_to_color(id), 3);
    }

    // draw axes of marker
    const auto& m_name = params.getMarkerNames().at(id);
    const auto& marker_config = params.getMarkerConfigs().at(m_name);
    cv::drawFrameAxes(
      out, params.getCameraInfo().new_K, 
      cv::noArray(), detected_marker.rvec, detected_marker.tvec, 
      marker_config.hexagon_radius_meter_*1, 10
    );
  }
  return true;
}

CheckInfo::~CheckInfo(){}

void LabellingAndDirectionCheckDetail::clear()
{
  success = false;

  labelling_detail.pseudo_labels.clear();

  labelling_detail.hexagon_center_undistorted = cv::Point2f(0,0);
  labelling_detail.pts_clockwise_distorted.clear();
  labelling_detail.pts_clockwise_undistorted.clear();

  direction_check_detail.success = false;
  direction_check_detail.min_radius_pixel_img = 0;
  direction_check_detail.max_radius_pixel_img = 0;
  direction_check_detail.directions_clockwise_degree.clear();
}

DetectedMarker LabellingAndDirectionCheckDetail::toDetectedMarker(int id) const 
{
  DetectedMarker detected_marker;
  detected_marker.id = id;

  // from regular shape
  detected_marker.pts_clockwise_distorted = 
    labelling_detail.pts_clockwise_distorted;
  detected_marker.pts_clockwise_undistorted =
    labelling_detail.pts_clockwise_undistorted;

  detected_marker.center_2d_undistorted = 
    cv::Point2f(
      labelling_detail.hexagon_center_undistorted.x,
      labelling_detail.hexagon_center_undistorted.y
    );
  
  // from direction check
  detected_marker.tvec = direction_check_detail.tvec;
  detected_marker.rvec = direction_check_detail.rvec;
  detected_marker.labels = direction_check_detail.labels;
  return detected_marker;
}

const PaacmanParameters& Paacman::getParams(){
  return params;
}

} // namespace paacman
