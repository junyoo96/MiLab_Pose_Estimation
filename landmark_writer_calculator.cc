#include <fstream>
#include <cstdlib>
#include "gnuplot.h"

#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/formats/landmark.pb.h"
// #include "mediapipe/framework/formats/detection.pb.h"

//string
#include <string>
#include <tuple>
#include <vector>

//For matrix operation
#include </usr/include/eigen3/Eigen/Core>
#include <cmath>

#include <time.h> 
#include <ctime>

#include <typeinfo>

//###############


//################

//vector angle
#define  PI 3.141592
#define  RADIAN ( PI / 180.0 )
#define  DEGREE ( 180.0 / PI )
#define  RAD2DEG(Rad)  ( Rad * DEGREE )

//GNUPlot
// #include <gnuplot-iostream.h>

// #include </usr/include/eigen3/unsupported/Eigen/MatrixFunctions>
// using std::sqrt; 
// using namespace Eigen;

//Blazepose Keypoint Enum
enum BlazePoseKeypointsIdx
{
  //FACE
  NOSE, //0
  LEFT_EYE_INNER,
  LEFT_EYE,
  LEFT_EYE_OUTER,
  RIGHT_EYE_INNER,
  RIGHT_EYE,
  RIGHT_EYE_OUTER,
  LEFT_EAR,
  RIGHT_EAR,
  MOUTH_LEFT,
  MOUTH_RIGHT, //10
  //UPPER BODY
  LEFT_SHOULDER, //11
  RIGHT_SHOULDER,
  LEFT_ELBOW,
  RIGHT_ELBOW,
  LEFT_WRIST,
  RIGHT_WRIST,
  LEFT_PINKY,
  RIGHT_PINKY,
  LEFT_INDEX,
  RIGHT_INDEX,
  LEFT_THUMB,
  RIGHT_THUMB, //22
  //LOWER BODY
  LEFT_HIP, //23
  RIGHT_HIP,
  LEFT_KNEE,
  RIGHT_KNEE,
  LEFT_ANKLE,
  RIGHT_ANKLE,//28
  LEFT_HEEL,
  RIGHT_HEEL, 
  LEFT_FOOT_INDEX,
  RIGHT_FOOT_INDEX //32
};

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
      
      //standThresholdMax :0.75 / detect : standing, shaking  
      standThresholdMax=0.85;
      // shakingSittingThresholdMax=0.8;
      
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
      
      // //################
      // //#Normal Version#
      // //################
      // //upper body
      // sigmas(0,0)=0.26; //nose

      // sigmas(1,0)=0.25; //left_eye_inner
      // sigmas(2,0)=0.25; //left_eye
      // sigmas(3,0)=0.25; //left_eye_outer
      // sigmas(4,0)=0.25; //right_eye_inner
      // sigmas(5,0)=0.25; //right_eye
      // sigmas(6,0)=0.25; //right_eye_outer
      
      // sigmas(7,0)=0.35; //left_ear
      // sigmas(8,0)=0.35; //right_ear

      // sigmas(9,0)=0.35; //mouth_left
      // sigmas(10,0)=0.35; //mouth_right

      // sigmas(11,0)=0.79; //left_shoulder
      // sigmas(12,0)=0.79; //right_shoulder

      // sigmas(13,0)=0.72; //left_elbow
      // sigmas(14,0)=0.72; //right_elbow

      // sigmas(15,0)=0.62; //left_wrist
      // sigmas(16,0)=0.62; //right_wrist

      // sigmas(17,0)=0.62; //left_pinky
      // sigmas(18,0)=0.62; //right_pinky
      // sigmas(19,0)=0.62; //left_index
      // sigmas(20,0)=0.62; //right_inddex
      // sigmas(21,0)=0.62; //left_thumb
      // sigmas(22,0)=0.62; //right_thumb
 
      // //lower body
      // sigmas(23,0)=1.07; //left_hip
      // sigmas(24,0)=1.07; //right_hip

      // sigmas(25,0)=0.87; //left_knee
      // sigmas(26,0)=0.87; //right_knee

      // sigmas(27,0)=0.89; //left_ankle
      // sigmas(28,0)=0.89; //right_ankle

      // sigmas(29,0)=0.89; //left_heel
      // sigmas(30,0)=0.89; //right_heel

      // sigmas(31,0)=0.89; //left_foot_index
      // sigmas(32,0)=0.89; //right_foot_index

      //################
      //#Custom Version#
      //################
      //Sensitivity weight 
      //0.1 :10  //x10 sensitivity
      //0.5 : 2 //x2 sensitivity
      //1 : 1 //standard
      //2 : 0.5 
      //10 :0.1
      //upper body

      // sigmas(0,0)=0.26; //nose
      // sigmas(1,0)=0.25; //left_eye_inner
      // sigmas(2,0)=0.25; //left_eye
      // sigmas(3,0)=0.25; //left_eye_outer
      // sigmas(4,0)=0.25; //right_eye_inner
      // sigmas(5,0)=0.25; //right_eye
      // sigmas(6,0)=0.25; //right_eye_outer
      sigmas(0,0)=10; //nose
      sigmas(1,0)=10; //left_eye_inner
      sigmas(2,0)=10; //left_eye
      sigmas(3,0)=10; //left_eye_outer
      sigmas(4,0)=10; //right_eye_inner
      sigmas(5,0)=10; //right_eye
      sigmas(6,0)=10; //right_eye_outer
      
      // sigmas(7,0)=0.35; //left_ear
      // sigmas(8,0)=0.35; //right_ear
      sigmas(7,0)=10; //left_ear
      sigmas(8,0)=10; //right_ear

      // sigmas(9,0)=0.35; //mouth_left
      // sigmas(10,0)=0.35; //mouth_right
      sigmas(9,0)=10; //mouth_left
      sigmas(10,0)=10; //mouth_right

      sigmas(11,0)=0.79; //left_shoulder
      sigmas(12,0)=0.79; //right_shoulder
      // sigmas(11,0)=0.79; //left_shoulder
      // sigmas(12,0)=0.79; //right_shoulder

      sigmas(13,0)=0.72; //left_elbow
      sigmas(14,0)=0.72; //right_elbow
      
      // sigmas(15,0)=0.62; //left_wrist
      // sigmas(16,0)=0.62; //right_wrist
      sigmas(15,0)=0.3; //left_wrist
      sigmas(16,0)=0.3; //right_wrist
      
      // sigmas(17,0)=0.62; //left_pinky
      // sigmas(18,0)=0.62; //right_pinky
      // sigmas(19,0)=0.62; //left_index
      // sigmas(20,0)=0.62; //right_inddex
      // sigmas(21,0)=0.62; //left_thumb
      // sigmas(22,0)=0.62; //right_thumb
      sigmas(17,0)=0.3; //left_pinky
      sigmas(18,0)=0.3; //right_pinky
      sigmas(19,0)=0.3; //left_index
      sigmas(20,0)=0.3; //right_inddex
      sigmas(21,0)=0.3; //left_thumb
      sigmas(22,0)=0.3; //right_thumb
 
      //lower body
      sigmas(23,0)=1.07; //left_hip
      sigmas(24,0)=1.07; //right_hip

      sigmas(25,0)=0.87; //left_knee
      sigmas(26,0)=0.87; //right_knee

      sigmas(27,0)=0.89; //left_ankle
      sigmas(28,0)=0.89; //right_ankle

      sigmas(29,0)=0.89; //left_heel
      sigmas(30,0)=0.89; //right_heel

      sigmas(31,0)=0.89; //left_foot_index
      sigmas(32,0)=0.89; //right_foot_index

      sigmas=sigmas/10.0;

    }
  
    std::tuple<int,double,std::string,std::string> detectCurrentState(const mediapipe::NormalizedLandmarkList& landmarks,int image_width,int image_height);
    std::tuple<int,double> detectAnomaly(std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd>&,int image_width,int image_height);
    std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> convertLandmarks(const mediapipe::NormalizedLandmarkList& landmarks);
    double computeOks(Eigen::MatrixXd lastFrameLandmarksXSet,Eigen::MatrixXd lastFrameLandmarksYSet,Eigen::MatrixXd lastFrameLandmarksVisSet,Eigen::MatrixXd curFrameLandmarksXSet, Eigen::MatrixXd curFrameLandmarksYSet,int image_width, int image_height);
    std::string judgeAnomalyActionType(std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd>& convertedLandmarks);
};


std::tuple<int,double,std::string,std::string> PoseAnomalyDetection::detectCurrentState(const mediapipe::NormalizedLandmarkList& landmarks,int image_width,int image_height){
    
    frameCount++;
    // std::cout<<"frame count: "<<frameCount<<std::endl;
    
    std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> convertedLandmarks=convertLandmarks(landmarks);
    
    std::tuple<int,double> info=detectAnomaly(convertedLandmarks,image_width,image_height);
    // int frameCountInfo=frameCount;
    double comparedValueInfo=std::get<1>(info);

    std::string curState="Normal";
    std::string curActionType="Sit";

    if (comparedValueInfo<standThresholdMax){
      curState="Anomaly";

      curActionType=judgeAnomalyActionType(convertedLandmarks);
    }

    //check whether sit and stand
    
    return std::make_tuple(frameCount,comparedValueInfo,curState,curActionType);

}

std::tuple<int,double> PoseAnomalyDetection::detectAnomaly(std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd>& landmarks,int image_width,int image_height){


  Eigen::MatrixXd curFrameLandmarksX=std::get<0>(landmarks);
  Eigen::MatrixXd curFrameLandmarksY=std::get<1>(landmarks);
  Eigen::MatrixXd curFrameLandmarksVis=std::get<2>(landmarks);
  double comparedValue=0;

  // std::cout<<image_width<<","<<image_height<<std::endl;

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
  
  Eigen::MatrixXd vars=sigmas*2;

  for(int i=0; i<possibleKeypointsNum; ++i)
  {
    vars(i,0)=std::pow(vars(i,0),2);
  }
  
  // int k=possibleKeypointsNum;

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
 
  // int lastFrameArea=image_width*image_height;

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
    //here
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

std::string PoseAnomalyDetection::judgeAnomalyActionType(std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd>& landmarks){
  
  std::string actionType="";

  Eigen::MatrixXd curFrameLandmarksX=std::get<0>(landmarks);
  Eigen::MatrixXd curFrameLandmarksY=std::get<1>(landmarks);
  Eigen::MatrixXd curFrameLandmarksVis=std::get<2>(landmarks);


  //check shoulder-hip-knee angle 
  double left_knee_x=curFrameLandmarksX(LEFT_HIP,0);
  double left_knee_y=curFrameLandmarksY(LEFT_HIP,0);
  double left_hip_x=curFrameLandmarksX(LEFT_SHOULDER,0);
  double left_hip_y=curFrameLandmarksY(LEFT_SHOULDER,0);
  double left_ankle_x=curFrameLandmarksX(LEFT_KNEE,0);
  double left_ankle_y=curFrameLandmarksY(LEFT_KNEE,0);
  
  // //check hip-knee-ankle angle 
  // double left_knee_x=curFrameLandmarksX(LEFT_KNEE,0);
  // double left_knee_y=curFrameLandmarksY(LEFT_KNEE,0);
  // double left_hip_x=curFrameLandmarksX(LEFT_HIP,0);
  // double left_hip_y=curFrameLandmarksY(LEFT_HIP,0);
  // double left_ankle_x=curFrameLandmarksX(LEFT_ANKLE,0);
  // double left_ankle_y=curFrameLandmarksY(LEFT_ANKLE,0);

  double left_hip_left_knee_vector_x=left_knee_x-left_hip_x;
  double left_hip_left_knee_vector_y=left_knee_y-left_hip_y;
  double left_knee_left_ankle_vector_x=left_ankle_x-left_knee_x;
  double left_knee_left_ankle_vector_y=left_ankle_y-left_knee_y;
  
  double dot_product= left_hip_left_knee_vector_x*left_knee_left_ankle_vector_x + left_hip_left_knee_vector_y*left_knee_left_ankle_vector_y;
  double left_hip_left_knee_vector_x_norm=std::sqrt(std::pow(left_hip_left_knee_vector_x,2)+std::pow(left_hip_left_knee_vector_y,2));
  double left_knee_left_ankle_vector_y_norm=std::sqrt(std::pow(left_knee_left_ankle_vector_x,2)+std::pow(left_knee_left_ankle_vector_y,2));
  double rad=std::acos(dot_product/(left_hip_left_knee_vector_x_norm*left_knee_left_ankle_vector_y_norm));
  double angle=180-RAD2DEG(rad);
  
  // std::cout<<angle<<std::endl;
  if (angle<130){
    actionType="Sit";
  }
  else{
    actionType="Stand";
  }
    
  return actionType;
}


//==============================================================================================


//MediaPipe code
namespace mediapipe {

namespace{
constexpr char kOutPoseStateTag[] = "POSE_STATE";
}

namespace api2 {

// constexpr char kDetectionTag[] = "DETECTION";

class LandmarkWriterCalculator : public Node {
 public:
  //Landmark
  static constexpr Input<mediapipe::NormalizedLandmarkList> kInLandmarks{"NORM_LANDMARKS"};
  static constexpr Input<std::pair<int, int>> kImageSize{"IMAGE_SIZE"};
  static constexpr Input<std::tuple<int,int>> kDetectionCounts{"DETECTIONS_COUNTS"};
  
  static constexpr Output<std::tuple<std::string,std::string>> kOutPoseState{kOutPoseStateTag};
  // static constexpr Output<std::tuple<std::string,std::string>>::Optional kOut{kOutPoseStateTag};

  PoseAnomalyDetection poseAnomalyDetection;

  MEDIAPIPE_NODE_INTERFACE(LandmarkWriterCalculator, kInLandmarks, kImageSize, kDetectionCounts,kOutPoseState);

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
    
    std::cout<<"gogo!"<<std::endl;

    file_path = getenv("HOME");
    file_path +="/pose_similarity.csv";
    // std::cout<<file_path<<std::endl;

    file_path_anomaly=getenv("HOME");
    file_path_anomaly+="/pose_anomaly.csv";
    
    file.open(file_path);
    file_anomaly.open(file_path_anomaly);
    RET_CHECK(file);
    RET_CHECK(file_anomaly);
    file<<"frame,keypointssimilarity,state"<<std::endl;
    file_anomaly<<"intervals,state"<<std::endl;

    return mediapipe::OkStatus();
  }

  mediapipe::Status Process(CalculatorContext* cc) final {

    mediapipe::Timestamp curTime2=cc->InputTimestamp();
    // std::cout<<"pose time:"<<curTime2<<std::endl;

    // std::cout<<"pose!!"<<std::endl;
    frameCount++;

    std::string curState="";
    std::string curActionType="";

    //detectioncounts
    const std::tuple<int, int>& detectionCounts= *kDetectionCounts(cc);
    int personCount=std::get<0>(detectionCounts);
    int objectCount=std::get<1>(detectionCounts);
    // std::cout<<"pose:"<<personCount<<","<<objectCount<<std::endl;

    bool isFaceExist=true;

    //얼굴없으면 keypoints 추출 안되기 때문에 그동안 모듈의 작업 수행 안되게 하기
    if(kInLandmarks(cc).IsEmpty()){
      isFaceExist=false;   
    }
    
    //얼굴이 존재할 때만 anomaly detection 수행 
    if(isFaceExist==true)
    {
      // std::cout<<"Face exists"<<std::endl;
      const mediapipe::NormalizedLandmarkList& landmarks = *kInLandmarks(cc);
  
      const std::pair<int, int>& image_size= *kImageSize(cc);

      // std::cout<<<<std::endl;
      // std::cout<<std::get<1>(detectionCounts)<<std::endl;

      int image_width;
      int image_height;
      std::tie(image_width, image_height) = image_size;

      if(isWebcam){
        if(frameCount==1){
          startTime=clock();
        }
        curTime=clock();
      }
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
      std::tuple<int,double,std::string,std::string> info=poseAnomalyDetection.detectCurrentState(landmarks,image_width,image_height);
      double comparedValue=std::get<1>(info);
      curState=std::get<2>(info);
      curActionType=std::get<3>(info);

      //10프레임마다 유사도 csv에 저장하고 plot하는 코드 
      if (frameCount%poseAnomalyDetection.oksCheckInterval==0 && frameCount>poseAnomalyDetection.oksCheckInterval){
        file << frameCount << ',' << comparedValue << ',' << curState;
        file << ',';
        file << std::endl;

        std::string home = getenv("HOME");
        // GnuplotPipe gp;
        // gp.sendLine("set datafile separator ','");
        // gp.sendLine("set terminal png");
        // gp.sendLine("set output '"+home+"/pose_similarity_plot.png'");
        // gp.sendLine("set xlabel 'frames'");
        // gp.sendLine("set yrange [0:1]");
        // gp.sendLine("set ytics 0.1");
        // gp.sendLine("plot '"+file_path+"' using 'keypointssimilarity' with linespoint ls 1 pt 5");
        
        frameSetCurState=curState;

      }
      
      if (frameSetCurState=="Anomaly"){
        isAnomaly=true;
      }

      //
      if(frameCount>poseAnomalyDetection.oksCheckInterval){
        
        if (isWebcam){
          double intervalTime=(double)((curTime-startTime)/CLOCKS_PER_SEC);
        // std::cout<<startTime<<"//"<<curTime<<"//"<<intervalTime<<"//"<<CLOCKS_PER_SEC<<std::endl;
        // std::cout<<intervalTime<<curTime<<poseAnomalyDetection.startTime;
          if(intervalTime>=1){
            secondCount++;
            std::string curSecondState="Normal";
            int curSecondStateIdx=0;
            startTime=curTime;
            if(isAnomaly==true){
              curSecondState="Anomaly";
              curSecondStateIdx=1;
              isAnomaly=false;
            }

            //최종판단 :실시간 영상에서 actiontype과 anomaly 최종 판단하는 곳 
            // std::cout<<secondCount<<","<<curSecondState<<","<<curActionType<<std::endl;
            // std::cout<<secondCount<<","<<curActionType<<std::endl;

            file_anomaly << secondCount << ',' << curSecondStateIdx;
            file_anomaly << ',';
            file_anomaly << std::endl;

            std::string home = getenv("HOME");
            // GnuplotPipe gp;
            // gp.sendLine("set datafile separator ','");
            // gp.sendLine("set terminal png");
            // gp.sendLine("set output '"+home+"/pose_anomaly_plot.png'");
            // gp.sendLine("set xlabel 'intervals'");
            
            
            // gp.sendLine("set yrange [0:1]");
            // gp.sendLine("set ytics 0.1");
            // gp.sendLine("plot '"+file_path_anomaly+"' using 'state' with linespoint ls 1 pt 5");
          }
        }
        else{
          
          if(frameCount%inputVideoFrame==0){
            std::string curFrameState="Normal";
            int curFrameStateIdx=0;
            if(isAnomaly==true){
              curFrameState="Anomaly";
              curFrameStateIdx=1;
              isAnomaly=false;
            }
            
            file_anomaly << frameCount << ',' << curFrameStateIdx;
            file_anomaly << ',';
            file_anomaly << std::endl;

            std::string home = getenv("HOME");
            // GnuplotPipe gp;
            // gp.sendLine("set datafile separator ','");
            // gp.sendLine("set terminal png");
            // gp.sendLine("set output '"+home+"/pose_anomaly_plot.png'");
            // gp.sendLine("set xlabel 'intervals'");
            
            // gp.sendLine("set yrange [0:1]");
            // gp.sendLine("set ytics 0.1");
            // gp.sendLine("plot '"+file_path_anomaly+"' using 'state' with linespoint ls 1 pt 5");
          }
        }
        
      }
    }
    else
    {
      curState="No Face/No person";
      curActionType="No Face/No person";
    }


    auto curPoseState=absl::make_unique<std::tuple<std::string,std::string>>(std::make_tuple(curState,curActionType));

    //send output
    cc->Outputs()
      .Tag(kOutPoseStateTag)
      .Add(curPoseState.release(), cc->InputTimestamp());
    
    return mediapipe::OkStatus();
  }
  
 private:
  std::ofstream file;
  std::ofstream file_anomaly;
  // std::string output_path;
  std::string file_path;
  std::string file_path_anomaly;
  clock_t startTime;
  clock_t curTime;
  // double secCount;
  int frameCount=0;
  int secondCount=0;
  std::string frameSetCurState;
  //check video from webcam for 1second decision
  bool isWebcam=true;
  int inputVideoFrame=15;
  bool isAnomaly=false;

};
MEDIAPIPE_REGISTER_NODE(LandmarkWriterCalculator);






}

}


