#include "paacman.hpp"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <unistd.h>

namespace fs = std::filesystem;
bool check_computation_latency = false;
int fps = 40;

using std::vector;
using std::cout;
using std::endl;
using std::string;

int main(){
  paacman::Paacman paacman("../cfg/config_files.json");
  paacman.setDebugging(true).setTimeLogging(true);
  // paacman.setDebugging(false).setTimeLogging(false);
  bool show = false;

  vector<string> sample_image_pathes = {
    "/workspace/PACMAN/inputs",
  };

  for (auto sample_image: sample_image_pathes){
    cout << "path: " << sample_image << endl;
    std::vector<cv::Mat> images;
    std::vector<std::string> image_names;
    // check if image extention endswith .png
    std::string ext = sample_image.substr(sample_image.find_last_of(".") + 1);
    if (ext == "png" || ext == "jpg" || ext == "jpeg"){ 
      cv::Mat img = cv::imread(sample_image, cv::IMREAD_GRAYSCALE); // for best computation speed, use grayscale image
      if (img.empty()) {
        std::cout << "image is empty" << std::endl;
        return 0;
      }
      int H = paacman.getParams().getCameraInfo().height;
      int W = paacman.getParams().getCameraInfo().width;
      if (img.cols != W || img.rows != H)
        cv::resize(img, img, cv::Size(W, H));
      images.push_back(img.clone());
      image_names.push_back(sample_image);
    } 
    else if (fs::is_directory(sample_image)){
      // load all images in the directory
      static auto imageOrderCmp = [](const std::string& a, const std::string& b){
        return std::stoi(a.substr(a.find_last_of("_") + 1, a.find_last_of(".") - a.find_last_of("_") - 1))
          < std::stoi(b.substr(b.find_last_of("_") + 1, b.find_last_of(".") - b.find_last_of("_") - 1));
      };
      cv::glob(sample_image, image_names, false);
      for (const auto& image_name : image_names){
        cv::Mat img = cv::imread(image_name, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
          std::cout << "image is empty" << std::endl;
          return 0;
        }
        int H = paacman.getParams().getCameraInfo().height;
        int W = paacman.getParams().getCameraInfo().width;
        if (img.cols != W || img.rows != H)
          cv::resize(img, img, cv::Size(W, H));
        images.push_back(img.clone());
        if (images.size() > 100) break; //twtw
      }
    }

    // detect Markers
    std::map<int, paacman::DetectedMarker> id_to_detected_markers;
    std::cout << "H, W: " << images.back().rows << ' ' << images.back().cols << endl; 
    std::cout << images.size() << std::endl;
    for (size_t i = 0 ; i < images.size(); ++i){
      cout << "image name: " << image_names[i] << endl;
      std::cout << "lets start detecting markers..." << std::endl;
      std::cout << images[i].rows << ' ' << images[i].cols << endl;
      bool success = paacman.detectMarkers(images[i], id_to_detected_markers);

      if (success){
        // detected marker info
        for (auto [id, detected_marker]: id_to_detected_markers){
          cout << "detected marker info..." << endl;
          cout << "  id: " << id << endl;
          cout << "  center_2d: " << detected_marker.center_2d_undistorted << endl;
          cout << "  yaw_degree: " << detected_marker.getYawDeg() << endl;
        }

        if (fs::is_directory(sample_image)){
          cout << "name: " << image_names[i] << endl;
        }
      }
      // else{
        if (show){
          static int n = 0;
          cv::Mat out;
          paacman.visualizeDetectedMarkers(images[i], out, id_to_detected_markers, false);

          cv::imshow("out", out);
          cv::waitKey(0);
          cv::destroyAllWindows();

          std::string fname = string() + "/workspace/estimate/debug/"
            + std::to_string(n++) + ".png";
          cv::imwrite(fname, out);
        }
      // }
    }
  }
  return 0;
}
