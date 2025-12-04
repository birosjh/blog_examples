#include <iostream>
#include <opencv2/opencv.hpp>


int main() {

  std::string imagePath = "image.png"; 
  cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);

  if (image.empty()) {
      std::cerr << "Error: Could not open or find the image at " << imagePath << std::endl;
      return -1; // Indicate an error
  }

  std::cout << "Height: " << image.rows << std::endl;
  std::cout << "Width: " << image.cols << std::endl;

  

  return 0;

}