#include <fstream>
#include <cstdlib>
#include "gnuplot.h"

#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/formats/landmark.pb.h"

//string
#include <string>
#include <tuple>
#include <vector>

//For matrix operation
#include </usr/include/eigen3/Eigen/Core>
#include <cmath>

//GNUPlot
// #include <gnuplot-iostream.h>

// #include </usr/include/eigen3/unsupported/Eigen/MatrixFunctions>
// using std::sqrt; 
// using namespace Eigen;

//anomaly detect 하는 class
class PoseAnomalyDetection{
  private:
    //몇 이상의 visbility를 가진 pose keypoints만을 oks 계산할 때 사용할 건지 
    double confidenceThreshold;
    //stand 판단하는 threshold
    double standThresholdMax;
    //shaking 판단하는 threshold
    double shakingSittingThresholdMax;
    //각 keypoint별 가중치
    Eigen::MatrixXd sigmas;

    //State variable
    bool isLastFrameFirst;
    int frameCount;
    

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
    int oksCheckInterval;

  public:
    PoseAnomalyDetection(){

      confidenceThreshold=0.5;
      standThresholdMax=0.5;
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
  
    std::tuple<int,double,std::string> detectCurrentState(const mediapipe::NormalizedLandmarkList& landmarks,int image_width,int image_height);
    std::tuple<int,double> detectAnomaly(std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd>&,int image_width,int image_height);
    std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> convertLandmarks(const mediapipe::NormalizedLandmarkList& landmarks);
    double computeOks(Eigen::MatrixXd lastFrameLandmarksXSet,Eigen::MatrixXd lastFrameLandmarksYSet,Eigen::MatrixXd lastFrameLandmarksVisSet,Eigen::MatrixXd curFrameLandmarksXSet, Eigen::MatrixXd curFrameLandmarksYSet,int image_width, int image_height);

};


std::tuple<int,double,std::string> PoseAnomalyDetection::detectCurrentState(const mediapipe::NormalizedLandmarkList& landmarks,int image_width,int image_height){
    
    frameCount++;
    // std::cout<<"frame count: "<<frameCount<<std::endl;
    
    std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> convertedLandmarks=convertLandmarks(landmarks);
    
    std::tuple<int,double> info=detectAnomaly(convertedLandmarks,image_width,image_height);
    // int frameCountInfo=frameCount;
    double comparedValueInfo=std::get<1>(info);

    std::string curState="Normal";

    if (comparedValueInfo<standThresholdMax){
      curState="Anomaly";
    }

    return std::make_tuple(frameCount,comparedValueInfo,curState);

}

std::tuple<int,double> PoseAnomalyDetection::detectAnomaly(std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd>& landmarks,int image_width,int image_height){


  Eigen::MatrixXd curFrameLandmarksX=std::get<0>(landmarks);
  Eigen::MatrixXd curFrameLandmarksY=std::get<1>(landmarks);
  Eigen::MatrixXd curFrameLandmarksVis=std::get<2>(landmarks);
  double comparedValue=0;

  // std::cout<<image_width<<","<<image_height<<std::endl;

  if (isLastFrameFirst==true){
    std::cout<<"first!"<<std::endl;
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

    std::cout<<"no!"<<std::endl;

    curFrameLandmarksXSet=curFrameLandmarksXSet+curFrameLandmarksX;
    curFrameLandmarksYSet=curFrameLandmarksYSet+curFrameLandmarksY;
    curFrameLandmarksVisSet=curFrameLandmarksVisSet+curFrameLandmarksVis;

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
        curFrameLandmarksXSet(i,0)=0.0;
        curFrameLandmarksYSet(i,0)=0.0;
        curFrameLandmarksVisSet(i,0)=0.0;
      }

      return std::make_tuple(frameCount,comparedValue);
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
  //2로 곱하니까 0.2곱해서 20 곱해줌
  Eigen::MatrixXd vars=sigmas*2;
  // std::cout<<vars<<std::endl;
  // std::cout<<"============"<<std::endl;
  for(int i=0; i<possibleKeypointsNum; ++i)
  {
    vars(i,0)=std::pow(vars(i,0),2);
  }
  
  int k=possibleKeypointsNum;

  std::vector<int> visEnabledIdx;

  //visibility flag True인 애들의 개수 가져오기
  for(int i=0; i<possibleKeypointsNum; ++i)
  {
    if(lastFrameLandmarksVisSet(i,0)>confidenceThreshold){
      visEnabledIdx.push_back(i);
    }
  }
  // std::cout<<"==========="<<std::endl;
  // std::cout<<lastFrameLandmarksVisSet<<std::endl;

  Eigen::MatrixXd dx;
  Eigen::MatrixXd dy;

  if (visEnabledIdx.size()>0){
    // std::cout<<"real!"<<std::endl;
    dx=curFrameLandmarksXSet-lastFrameLandmarksXSet;
    dy=curFrameLandmarksYSet-lastFrameLandmarksYSet;
  }
  else{
    // std::cout<<"nothing"<<std::endl;
  }
 
  int lastFrameArea=image_width*image_height;

  //calculate distance
  for(int i=0; i<possibleKeypointsNum; ++i)
  {
    dx(i,0)=std::pow(dx(i,0),2);
    dy(i,0)=std::pow(dy(i,0),2);
  }
  
  Eigen::MatrixXd e=dx+dy;

  // std::cout<<"============="<<std::endl;
  // std::cout<<e<<std::endl;

  for(int i=0; i<possibleKeypointsNum; ++i)
  {
    e(i,0)=e(i,0)/vars(i,0);
  }

  // std::cout<<"VisEnabledIdxsize:"<<visEnabledIdx.size()<<std::endl;

  if(visEnabledIdx.size()>0){
    for(int i=0; i<visEnabledIdx.size(); ++i)
    {
      ious+=std::exp(e(visEnabledIdx[i],0)*-1);
    }
  }

  // std::cout<<"========================================="<<std::endl;
  ious/=visEnabledIdx.size();
  // std::cout<<"ious:"<<ious<<std::endl;
  
  // std::cout<<"========================================="<<std::endl;
  
  return ious;
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
  //   // std::string file_path = getenv("HOME");
  //   // file_path += "/landmarks.csv";
    
  //   // file.open(file_path);
  //   // RET_CHECK(file);
    
  //   return mediapipe::OkStatus();
  // }

  mediapipe::Status Open(CalculatorContext* cc) final {
    // output_path = getenv("HOME");
    // output_path+="youngwan/youngjun/mp_cplus/mediapipe/output/pose";
    

    file_path = getenv("HOME");
    file_path +="/pose_anomaly.csv";
    // std::cout<<file_path<<std::endl;
    
    file.open(file_path);
    RET_CHECK(file);
    file<<"frame,keypointssimilarity,state"<<std::endl;
    
    return mediapipe::OkStatus();
  }

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
    std::tuple<int,double,std::string> info=poseAnomalyDetection.detectCurrentState(landmarks,image_width,image_height);
    int frameCount=std::get<0>(info);
    double comparedValue=std::get<1>(info);
    std::string curState=std::get<2>(info);

    std::cout<<"Framecount:"<<frameCount<<std::endl;

    if (frameCount%10==0 && frameCount>poseAnomalyDetection.oksCheckInterval){
      std::cout<<comparedValue<<","<<curState<<std::endl;
      file << frameCount << ',' << comparedValue << ',' << curState;
      file << ',';
      file << std::endl;

      std::string home = getenv("HOME");
      GnuplotPipe gp;
      gp.sendLine("set datafile separator ','");
      gp.sendLine("set terminal png");
      gp.sendLine("set output '"+home+"/pose_anomaly_plot.png'");
      gp.sendLine("set xlabel 'frames'");
      gp.sendLine("set yrange [0:1]");
      gp.sendLine("set ytics 0.1");
      gp.sendLine("plot '"+file_path+"' using 'keypointssimilarity' with linespoint ls 1 pt 5");

    }
    

    // frame,keypointsimilarity,state

    


    // std::cout<<curState<<std::endl;

    return mediapipe::OkStatus();
  }
  
 private:
  std::ofstream file;
  std::string output_path;
  std::string file_path;
};
MEDIAPIPE_REGISTER_NODE(LandmarkWriterCalculator);






}

}


