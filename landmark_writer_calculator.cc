#include <fstream>
#include <cstdlib>

#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/formats/landmark.pb.h"

//string
#include <string>
#include <tuple>

//For matrix operation
#include </usr/include/eigen3/Eigen/Core>
// #include </usr/include/eigen3/unsupported/Eigen/MatrixFunctions>
// using std::sqrt; 
// using namespace Eigen;

//anomaly detect 하는 class
class PoseAnomalyDetection{
  private:
    //몇 이상의 visbility를 가진 pose keypoints만을 oks 계산할 때 사용할 건지 
    int confidenceThreshold;
    //stand 판단하는 threshold
    int standThresholdMax;
    //shaking 판단하는 threshold
    int shakingSittingThresholdMax;
    //각 keypoint별 가중치
    Eigen::MatrixXd sigmas;

    //State variable
    bool isLastFrameFirst;
    int frameCount;
    int oksCheckInterval;

    //
    int possibleKeypointsNum;

    //Realtime plot variable
    bool isRealtimePlotActivated;
    int realtimePlotNum;
    // realtime_plot_comparedposevalues=deque([])


    std::list<double> comparedPoseValues;

    //lastkeypoint
    Eigen::MatrixXd lastFrameLandmarksXSet;
    Eigen::MatrixXd lastFrameLandmarksYSet;
    Eigen::MatrixXd lastFrameLandmarksVisSet;

    Eigen::MatrixXd curFrameLandmarksXSet;
    Eigen::MatrixXd curFrameLandmarksYSet;
    Eigen::MatrixXd curFrameLandmarksVisSet;

  public:
    PoseAnomalyDetection(){

      confidenceThreshold=0.5;
      standThresholdMax=0.3;
      shakingSittingThresholdMax=0.8;

      isLastFrameFirst=true;
      frameCount=0;
      oksCheckInterval=10;

      possibleKeypointsNum=33;

      isRealtimePlotActivated=false;
      realtimePlotNum=10;

      lastFrameLandmarksXSet=Eigen::MatrixXd(possibleKeypointsNum,1);
      lastFrameLandmarksYSet=Eigen::MatrixXd(possibleKeypointsNum,1);
      lastFrameLandmarksVisSet=Eigen::MatrixXd(possibleKeypointsNum,1);

      curFrameLandmarksXSet=Eigen::MatrixXd(possibleKeypointsNum,1);
      curFrameLandmarksYSet=Eigen::MatrixXd(possibleKeypointsNum,1);
      curFrameLandmarksVisSet=Eigen::MatrixXd(possibleKeypointsNum,1);

      for(int i=0; i<possibleKeypointsNum; ++i)
      {
        lastFrameLandmarksXSet(i,0)=0.0;
        lastFrameLandmarksYSet(i,0)=0.0;
        lastFrameLandmarksVisSet(i,0)=0.0;

        curFrameLandmarksXSet(i,0)=0.0;
        curFrameLandmarksYSet(i,0)=0.0;
        curFrameLandmarksVisSet(i,0)=0.0;
      }

      sigmas=Eigen::MatrixXd(possibleKeypointsNum,1);
      sigmas(0,0)=0.26;
      sigmas(1,0)=0.25;
      sigmas(2,0)=0.25;
      sigmas(3,0)=0.25;
      sigmas(4,0)=0.25;
      sigmas(5,0)=0.25;
      sigmas(6,0)=0.25;
      
      sigmas(7,0)=0.35;
      sigmas(8,0)=0.35;
      sigmas(9,0)=0.35;
      sigmas(10,0)=0.35;

      sigmas(11,0)=0.79;
      sigmas(12,0)=0.79;
      sigmas(13,0)=0.72;
      sigmas(14,0)=0.72;

      sigmas(15,0)=0.62;
      sigmas(16,0)=0.62;

      sigmas(17,0)=0.62;
      sigmas(18,0)=0.62;
      sigmas(19,0)=0.62;
      sigmas(20,0)=0.62;
      sigmas(21,0)=0.62;
      sigmas(22,0)=0.62;

      sigmas(23,0)=1.07;
      sigmas(24,0)=1.07;

      sigmas(25,0)=0.87;
      sigmas(26,0)=0.87;

      sigmas(27,0)=0.89;
      sigmas(28,0)=0.89;

      sigmas(29,0)=0.89;
      sigmas(30,0)=0.89;

      sigmas(31,0)=0.89;
      sigmas(32,0)=0.89;

      sigmas=sigmas/10.0;

    }
  
    std::string detectCurrentState(const mediapipe::NormalizedLandmarkList& landmarks,int image_width,int image_height);
    double detectAnomaly(std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd>&,int image_width,int image_height);
    std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> convertLandmarks(const mediapipe::NormalizedLandmarkList& landmarks);
    double computeOks(Eigen::MatrixXd lastFrameLandmarksXSet,Eigen::MatrixXd lastFrameLandmarksYSet,Eigen::MatrixXd lastFrameLandmarksVisSet,Eigen::MatrixXd curFrameLandmarksXSet, Eigen::MatrixXd curFrameLandmarksYSet,int image_width, int image_height);

};


std::string PoseAnomalyDetection::detectCurrentState(const mediapipe::NormalizedLandmarkList& landmarks,int image_width,int image_height){
    
    frameCount++;
    std::cout<<"frame count: "<<frameCount<<std::endl;
    
    std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> convertedLandmarks=convertLandmarks(landmarks);
    

    detectAnomaly(convertedLandmarks,image_width,image_height);

    std::string curState="Normal";

    return curState;

}

double PoseAnomalyDetection::detectAnomaly(std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd>& landmarks,int image_width,int image_height){


  Eigen::MatrixXd curFrameLandmarksX=std::get<0>(landmarks);
  Eigen::MatrixXd curFrameLandmarksY=std::get<1>(landmarks);
  Eigen::MatrixXd curFrameLandmarksVis=std::get<2>(landmarks);
  double comparedValue=0;

  std::cout<<image_width<<","<<image_height<<std::endl;

  if (isLastFrameFirst==true){
    lastFrameLandmarksXSet=lastFrameLandmarksXSet+curFrameLandmarksX;
    lastFrameLandmarksYSet=lastFrameLandmarksYSet+curFrameLandmarksY;
    lastFrameLandmarksVisSet=lastFrameLandmarksVisSet+curFrameLandmarksVis;

    // std::cout<<"before"<<std::endl;
    // std::cout<<curFrameLandmarksX<<std::endl;

    if (frameCount%oksCheckInterval==0){
      lastFrameLandmarksXSet=lastFrameLandmarksXSet/oksCheckInterval;
      lastFrameLandmarksYSet=lastFrameLandmarksYSet/oksCheckInterval;
      lastFrameLandmarksVisSet=lastFrameLandmarksVisSet/oksCheckInterval;

      isLastFrameFirst=false;
      // std::cout<<"yes"<<std::endl;
      // std::cout<<lastFrameLandmarksXSet<<std::endl;
    }
  }
  else{
    curFrameLandmarksXSet=curFrameLandmarksXSet+curFrameLandmarksX;
    curFrameLandmarksYSet=curFrameLandmarksXSet+curFrameLandmarksY;
    curFrameLandmarksVisSet=curFrameLandmarksXSet+curFrameLandmarksVis;

    if (frameCount%oksCheckInterval==0){
      curFrameLandmarksXSet=curFrameLandmarksXSet/oksCheckInterval;
      curFrameLandmarksYSet=curFrameLandmarksYSet/oksCheckInterval;
      curFrameLandmarksVisSet=curFrameLandmarksVisSet/oksCheckInterval;

      comparedValue=computeOks(lastFrameLandmarksXSet,lastFrameLandmarksYSet,lastFrameLandmarksVisSet,curFrameLandmarksXSet,curFrameLandmarksYSet,image_width,image_height);

      comparedPoseValues.push_back(comparedValue);
      lastFrameLandmarksXSet=curFrameLandmarksXSet;
      lastFrameLandmarksYSet=curFrameLandmarksYSet;
      lastFrameLandmarksVisSet=curFrameLandmarksVisSet;

      //초기화
      curFrameLandmarksXSet=Eigen::MatrixXd(possibleKeypointsNum,1);
      curFrameLandmarksYSet=Eigen::MatrixXd(possibleKeypointsNum,1);
      curFrameLandmarksVisSet=Eigen::MatrixXd(possibleKeypointsNum,1);

      for(int i=0; i<possibleKeypointsNum; ++i)
      {
        lastFrameLandmarksXSet(i,0)=0.0;
        lastFrameLandmarksYSet(i,0)=0.0;
        lastFrameLandmarksVisSet(i,0)=0.0;

        curFrameLandmarksXSet(i,0)=0.0;
        curFrameLandmarksYSet(i,0)=0.0;
        curFrameLandmarksVisSet(i,0)=0.0;
      }

      return 3.0;
    }
    
  }
    
  
  
}

std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> PoseAnomalyDetection::convertLandmarks(const mediapipe::NormalizedLandmarkList& landmarks){

  Eigen::MatrixXd landmarksX(possibleKeypointsNum,1);
  Eigen::MatrixXd landmarksY(possibleKeypointsNum,1);
  Eigen::MatrixXd landmarksVis(possibleKeypointsNum,1);

  for (int i = 0; i < landmarks.landmark_size(); ++i) {
      const mediapipe::NormalizedLandmark& landmark = landmarks.landmark(i);
      landmarksX(i,0)=landmark.x();
      landmarksY(i,0)=landmark.y();
      landmarksVis(i,0)=landmark.visibility();    
  }

  return std::make_tuple(landmarksX,landmarksY,landmarksVis);

}

double PoseAnomalyDetection::computeOks(Eigen::MatrixXd lastFrameLandmarksXSet,Eigen::MatrixXd lastFrameLandmarksYSet,Eigen::MatrixXd lastFrameLandmarksVisSet,Eigen::MatrixXd curFrameLandmarksXSet, Eigen::MatrixXd curFrameLandmarksYSet,int image_width, int image_height){
  
  double ious=0;


  
  return 1.0;
}


//==============================================================================================


//MediaPipe code
namespace mediapipe {

namespace api2 {
class LandmarkWriterCalculator : public Node {
 public:
  //Landmark
  static constexpr Input<mediapipe::NormalizedLandmarkList> kInLandmarks{"NORM_LANDMARKS"};
  static constexpr Input<std::pair<int, int>> kImageSize{"IMAGE_SIZE"};
  static constexpr Output<int>::Optional kOutUnused{"INT"};

  PoseAnomalyDetection poseAnomalyDetection;

  MEDIAPIPE_NODE_INTERFACE(LandmarkWriterCalculator, kInLandmarks, kImageSize,kOutUnused);

  static mediapipe::Status UpdateContract(CalculatorContract* cc) {
    
    return mediapipe::OkStatus();
  }

  // mediapipe::Status Open(CalculatorContext* cc) final {
  //   // std::string out_path = getenv("HOME");
  //   // out_path += "/landmarks.csv";
    
  //   // file.open(out_path);
  //   // RET_CHECK(file);
    
  //   return mediapipe::OkStatus();
  // }
  
  mediapipe::Status Process(CalculatorContext* cc) final {
    const mediapipe::NormalizedLandmarkList& landmarks = *kInLandmarks(cc);
    const std::pair<int, int>& image_size= *kImageSize(cc);

    // const auto& image = cc->Inputs().Tag(kImageFrameTag).Get<ImageFrame>();
    int image_width;
    int image_height;
    std::tie(image_width, image_height) = image_size;

    // std::cout<<image_width<<","<<image_height<<std::endl;

    // for (int i = 0; i < landmarks.landmark_size(); ++i) {
    //   const mediapipe::NormalizedLandmark& landmark = landmarks.landmark(i);
    //   std::cout << landmark.x() << ',' << landmark.y() << ',' << landmark.visibility()<<std::endl;
    //   if (i < landmarks.landmark_size() - 1) {
    //     file << ',';
    //   }
    // }
    // file << std::endl;

    //detect anomaly 
    // PoseAnomalyDetection ad=PoseAnomalyDetection();
    std::string curState=poseAnomalyDetection.detectCurrentState(landmarks,image_width,image_height);
    std::cout<<curState<<std::endl;

    
    return mediapipe::OkStatus();
  }
  
 private:
  std::ofstream file;
};
MEDIAPIPE_REGISTER_NODE(LandmarkWriterCalculator);






}

}


