import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#jun
import numpy as np
import sys
import matplotlib.pyplot as plt
import time 
from collections import deque
#jun finish

# # For static images:
# pose = mp_pose.Pose(
#     static_image_mode=True, min_detection_confidence=0.5)
# for idx, file in enumerate(file_list):
#   image = cv2.imread(file)
#   image_hight, image_width, _ = image.shape
#   # Convert the BGR image to RGB before processing.
#   results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#   if not results.pose_landmarks:
#     continue
#   print(
#       f'Nose coordinates: ('
#       f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
#       f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_hight})'
#   )
#   # Draw pose landmarks on the image.
#   annotated_image = image.copy()
#   mp_drawing.draw_landmarks(
#       annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#   cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
# pose.close()

#jun

isLastFrameFirst=True


#고정(0.7이 제일 좋음)
confidence_threshold=0.5
#자세 변화 감지 민감도
#stand
stand_threshold_max=0.3
#shaking while sitting
shaking_sitting_threshold_max=0.8

#자세별 threshold


compared_pose_values=[]

#realtime plot 몇개 단위로 출력할지 
isRealtimePlotActivated=True
realtime_plot_num=10
realtime_plot_comparedposevalues=deque([])

#oks 비교하는 frame 간격 몇 으로 할지
oks_check_interval=10
#frame 몇 번 돌았는지 확인 
frame_check_count=0
#detect 가능한 keyponits 개수
possible_num_keypoints=33

#last frame keypoints 저장 변수
last_frame_keypoints_set=np.zeros((possible_num_keypoints,2))
current_frame_keypoints_set=np.zeros((possible_num_keypoints,2))
last_pose_score_set=np.zeros((possible_num_keypoints,1))
current_pose_score_set=np.zeros((possible_num_keypoints,1))

#Video로 만들 frame 저장 변수
video_frames=[]
video_save_path="video_output/test.mp4"
big_difference_video_save_path="video_output/test_big_difference.mp4"

def detectAnomaly(current_pose_landmarks,bbox,isRealtimePlotActivated,realtime_plot_ax):

    global isLastFrameFirst
    #고정(0.7이 제일 좋음)
    global confidence_threshold
    #자세 변화 감지 민감도
    global stand_threshold_max
    global shaking_sitting_threshold_max

    global compared_pose_values
    #human 몇명 감지됐는지 확인
    # num_human_list=[]
    #oks 비교하는 frame 간격 몇 으로 할지
    global oks_check_interval
    #frame 몇 번 돌았는지 확인 
    global frame_check_count
    #detect 가능한 keyponits 개수
    global possible_num_keypoints

    #last frame keypoints 저장 변수
    global last_frame_keypoints_set
    global current_frame_keypoints_set
    global last_pose_score_set
    global current_pose_score_set

    converted_current_pose_landmarks, converted_current_pose_landmarks_coordinates, converted_current_pose_landmarks_visibility=convert(current_pose_landmarks)
    
    pose_coord = np.array(converted_current_pose_landmarks_coordinates).reshape(possible_num_keypoints,2)
    pose_score = np.array(converted_current_pose_landmarks_visibility).reshape(possible_num_keypoints,1)
    
    #jun : 이전 frame과 현재 frame간의 값 비교 
    if isLastFrameFirst:
        last_frame_keypoints_set+=pose_coord
        last_pose_score_set+=pose_score

        if frame_check_count%oks_check_interval==0:
            last_frame_keypoints_set/=oks_check_interval
            last_pose_score_set/=oks_check_interval
            isLastFrameFirst=False
 
    else:
        current_frame_keypoints_set+=pose_coord
        current_pose_score_set+=pose_score
        
        if frame_check_count%oks_check_interval==0:
            current_frame_keypoints_set/=oks_check_interval
            current_pose_score_set/=oks_check_interval

            #OKS로 frame간의 값 비교 
            compared_value=computeOks(last_frame_keypoints_set, current_frame_keypoints_set,last_pose_score_set,bbox,confidence_threshold=0.5)
            print("compared_value:",compared_value)

            #실시간 plot 
            if isRealtimePlotActivated==True:
                if len(realtime_plot_comparedposevalues)==5:
                    realtime_plot_comparedposevalues.popleft()
                    realtime_plot_comparedposevalues.append(compared_value)
                else:
                    realtime_plot_comparedposevalues.append(compared_value)
                showRealtimePlot(realtime_plot_num,realtime_plot_comparedposevalues,frame_check_count,realtime_plot_ax)

            compared_pose_values.append(compared_value)
            last_frame_keypoints_set=current_frame_keypoints_set
            last_pose_score_set=current_pose_score_set

            #초기화
            current_frame_keypoints_set=np.zeros((possible_num_keypoints,2))
            current_pose_score_set=np.zeros((possible_num_keypoints,1))

            #현재 상태 return 
            if compared_value<=stand_threshold_max:
                return "Stand"
            # elif stand_threshold_max<=compared_value and compared_value<=shaking_sitting_threshold_max:
            #     return "Shaking"
            else:
                return "Normal"

def convert(pose_landmarks):

    converted_pose_landmarks=[]
    converted_current_pose_landmarks_coordinates=[]
    converted_current_pose_landmarks_visibility=[]
    for pose_landmark in pose_landmarks:
        converted_pose_landmarks.append(pose_landmark.x)
        converted_pose_landmarks.append(pose_landmark.y)
        converted_pose_landmarks.append(pose_landmark.visibility)

        converted_current_pose_landmarks_coordinates.append(pose_landmark.x)
        converted_current_pose_landmarks_coordinates.append(pose_landmark.y)
        converted_current_pose_landmarks_visibility.append(pose_landmark.visibility)
    
    return converted_pose_landmarks,converted_current_pose_landmarks_coordinates,converted_current_pose_landmarks_visibility


#computeOks
def computeOks(last_frame_keypoints_np, current_frame_keypoints,last_pose_score_np,bbox,confidence_threshold=0.5):
    # print("last_frame_keypoints",last_frame_keypoints,last_frame_keypoints.shape)
    # print("current_frame_keypoints",current_frame_keypoints)
    # print("last_pose_score",last_pose_score)
    # print("bbox",bbox)
    # print("이전",last_frame_keypoints)
    # last_frame_keypoints_np=np.array(last_frame_keypoints)
    # print("이후",last_frame_keypoints_np)
    # last_pose_score_np=np.array(last_pose_score)
    # print("last_frame_keypoints_np",last_frame_keypoints_np)
    # print("last_pose_score_np",last_pose_score_np)
    last_frame_keypoints_np=np.insert(last_frame_keypoints_np,2,last_pose_score_np.flatten(),axis=1)
    # print("last_frame_keypoints_np",last_frame_keypoints_np)
    last_frame_keypoints=list(last_frame_keypoints_np.flatten())
    # print("last_frame_keypoints",last_frame_keypoints)
    # print("current_frame_keypoints",current_frame_keypoints)
    current_frame_keypoints_np=current_frame_keypoints.flatten()
    # print("current_frame_keypoints_np",current_frame_keypoints_np)

    ious=0
    #sigmas가 뭔지 coco값 그대로 들어가는게 맞는지 확인
    # sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
    # sigmas = np.array([1 for i in range(33)])/10.0
    sigmas = np.array([0.26,0.25,0.25,0.25,0.25,0.25,0.25,
                    0.35,0.35,
                    0.35,0.35,
                    0.79,0.79,
                    0.72,0.72,
                    0.62,0.62,
                    0.62,0.62,0.62,0.62,0.62,0.62,
                    1.07,1.07,
                    0.87,0.87,
                    0.89,0.89,
                    0.89,0.89,
                    0.89,0.89])/10.0


    vars = (sigmas * 2)**2
    k = len(sigmas)
    # compute oks between each detection and ground truth object
        #각 ground truth 돌면서 oks계산
    
    # create bounds for ignore regions(double the gt bbox)
    #g : last frame keypoints
    g = last_frame_keypoints
    #xg : ground truth keypoints array에서 0번째 부터 시작해서 3칸 간격으로 각 keypoint의 x좌표 가져오기
    #yg : ground truth keypoints array에서 1번째 부터 시작해서 3칸 간격으로 각 keypoint의 y좌표 가져오기
    #vg(visibility flag) : ground truth keypoints array에서 2번째 부터 시작해서 3칸 간격으로 각 keypoint의 z좌표 가져오기
    xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
    # print("x",xg)
    # print("y",yg)
    # print("v",vg)
    vg=np.array(vg)
    #k1 : visibility flag True인 애들의 개수 가져오기
    k1 = np.count_nonzero(vg > confidence_threshold)
    

    #bounding box 처리
    bb = bbox
    # print(bb)
    #bb[0] : top left x,bb[1]:top left y,bb[2]:width,bb[3]:height
    #x0 : down left x
    #x1 : ?
    #y0 : down left y
    #y1 : ?
    # x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
    # y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
    
    #current frame keypoints
    d = current_frame_keypoints_np
    #predicted keypoints array에서 0번째 부터 시작해서 3칸 간격으로 각 keypoint의 x좌표 가져오기
    #predicted keypoints array에서 1번째 부터 시작해서 3칸 간격으로 각 keypoint의 y좌표 가져오기
    xd = d[0::2]; yd = d[1::2]
    #visibility flag True인 애들만
    #threshold중 keypoint가 confidence score 보다 높은 것들이 1개라도 있다면 
    if k1>0:
        # measure the per-keypoint distance if keypoints visible
        #각 keypoint간의 거리를 계산 
        dx = xd - xg
        dy = yd - yg
        # print("x거리",dx)
        # print("y거리",dy)
    #threshold중 keypoint가 confidence score 보다 높은 것들이 아예 없을 경우 
    else:
        # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
        print("nothing!")
        sys.exit()
        # z = np.zeros((k))
        # dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
        # dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
    #밑에서부터 OKS 수식 구현한 것 
    #vars = (sigmas * 2)**2
    # e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
    last_frame_area=bb[2]*bb[3]
    # print("last_frame_area",last_frame_area)
    # print("check",(dx**2 + dy**2),len((dx**2 + dy**2)))

    # e = (dx**2 + dy**2) / vars / (last_frame_area+np.spacing(1)) / 2
    e = (dx**2 + dy**2) / vars 
    # e = (dx**2 + dy**2)

    #나중에 confidence score높은 애들만 계산하는걸로 고치면 될듯?
    if k1 > 0:
        # print("사용된 keypoint 수 ",k1)
        # print("사용된 keypoint index",vg > confidence_threshold)
        e=e[vg > confidence_threshold]
        # print("e",e)
        # print("e",e,len(e))
    #np.sum(np.exp(-e)) / e.shape[0]값 그대로 출력하면 될듯
    ious = np.sum(np.exp(-e)) / e.shape[0]

    return ious


def showRealtimePlot(realtime_plot_num,realtime_plot_comparedposevalues,frame_check_count,realtime_plot_ax):

    x_max_val=frame_check_count+(realtime_plot_num-1)*oks_check_interval

    realtime_plot_ax.set_xlim([frame_check_count,x_max_val])
    
    xticks_val=[i for i in range(frame_check_count,x_max_val+1,oks_check_interval)]
    realtime_plot_ax.set_xticks(xticks_val)
    realtime_plot_ax.set_ylim([0,1])
    
    x_val=[i for i in range(frame_check_count,frame_check_count+(len(realtime_plot_comparedposevalues)-1)*oks_check_interval+1,oks_check_interval)]
    print("x_val",x_val)
    print("realtime_plot_comparedposevalues",len(realtime_plot_comparedposevalues))
    realtime_plot_ax.plot(x_val,realtime_plot_comparedposevalues,color='blue')
    
    figure.canvas.draw()
    
    figure.canvas.flush_events()
    # time.sleep(0.5)
    


def savePlot():

    print("compared_pose_values_num",len(compared_pose_values))
    print("compared_pose_values",compared_pose_values)
    plt.figure(figsize=(15,15))
    plt.ylim(0,1)
    plt.plot([(i+2)*oks_check_interval for i in range(len(compared_pose_values))],compared_pose_values)
    different_frame_indices=[idx for idx,val in enumerate(compared_pose_values) if val<=stand_threshold_max]
    different_frame_string=""
    for frame_index in different_frame_indices:
        different_frame_string+=str(frame_index-1)+"->"+str(frame_index)+"/"
        
    max_difference_frame=compared_pose_values.index(min(compared_pose_values)) 
    plt.title("Total Similarity: "+"#f:"+str(len(compared_pose_values))+"/st:"+str(stand_threshold_max)+"/ct:"+str(confidence_threshold))
    plt.xlabel("Frame Interval : "+str(oks_check_interval))
    plt.ylabel("Similarity")
    plt.show()
    plt.savefig("video_output/compared_value_fig_"+"ct("+str(confidence_threshold)+")_"+"st("+str(stand_threshold_max)+")"+".png",dpi=200)
    print("Save compared value plot between frame")
    # jun


def saveBigDifferenceVideo(frame_array,save_path,fps,size):
    

    big_difference_frame_idx=[(frame_idx+2)*oks_check_interval for frame_idx,val in enumerate(compared_pose_values) if val<stand_threshold_max]
    # big_difference_frame_idx=(big_difference_frame_idx+2)*oks_check_interval
    # print(big_difference_frame_idx)

    if not big_difference_frame_idx:
        return

    out = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    target_frame=big_difference_frame_idx[0]
    # print("target_frame",target_frame)
    target_frame_start=0
    target_frame_end=0

    frame_coverage=50

    if target_frame-frame_coverage>0:
        target_frame_start=target_frame-frame_coverage
    else:
        target_frame_start=0
    
    if target_frame+frame_coverage+1>len(frame_array):
        target_frame_end=len(frame_array)-1
    else:
        target_frame_end=target_frame+frame_coverage+1

    for frame in frame_array[target_frame_start:target_frame_end]:
        # writing to a image array
        out.write(frame)
    out.release()

def saveVideo(frame_array,save_path,fps,size):
    out = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


#=================================================================
#main loop

prevTime=0
frame_avg=[]
alert_count=0
#current state
cur_state="Normal"

realtime_plot_ax=None
#realtime plot 
if isRealtimePlotActivated:
    plt.ion()
    figure, realtime_plot_ax = plt.subplots(figsize=(8,6))
    

    plt.title("Realtime plot",fontsize=25)

    plt.xlabel("Frames",fontsize=18)
    plt.ylabel("Similarity",fontsize=18)

# For webcam input:
pose = mp_pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

while cap.isOpened():
  success, image = cap.read()


  if not success:
    print("Ignoring empty camera frame.")
    # If loading a video, use 'break' instead of 'continue'.
    continue

  frame_check_count+=1
  print("frame_count:",frame_check_count)

  # Flip the image horizontally for a later selfie-view display, and convert
  # the BGR image to RGB.
  image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
  # To improve performance, optionally mark the image as not writeable to
  # pass by reference.
  image.flags.writeable = False
  results = pose.process(image)
#   print(type(results.pose_landmarks.landmark))
#   print(len(results.pose_landmarks.landmark))
#   print(results.pose_landmarks.landmark[-1].x)
#   print(results.pose_landmarks.landmark[-1].y)
#   print(results.pose_landmarks.landmark[-1].z)

  image_height,image_width,channel= image.shape
  bbox=[]
  bbox.append(0)
  bbox.append(0)
  bbox.append(image_width)
  bbox.append(image_height)

  

  if results.pose_landmarks is None:
    cur_state="No person"
  else:
    cur_state=detectAnomaly(results.pose_landmarks.landmark,bbox,isRealtimePlotActivated,realtime_plot_ax)
        
  #jun_finish

  # Draw the pose annotation on the image.
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  mp_drawing.draw_landmarks(
      image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

  #Show state
  cv2.putText(image, cur_state, (0, 120), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0),thickness=2)

  #Show frame 
  curTime=time.time()
  sec=curTime-prevTime
  prevTime=curTime
  fps=1/(sec)
  frame_avg.append(fps)
  fpsStr="FPS : %0.1f" % fps
  cv2.putText(image, fpsStr, (0, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255),thickness=2)


  cv2.imshow('MediaPipe Pose', image)

  video_frames.append(image)

  
  if frame_check_count==300:
      break
  if cv2.waitKey(5) & 0xFF == 27:
    break
pose.close()
cap.release()

#Plot results
plt.close()
savePlot()

#Save big diffrence momment in video
saveBigDifferenceVideo(video_frames,big_difference_video_save_path,
    fps=round(sum(frame_avg)/len(frame_avg),1),
    size=(image_width,image_height))

#Save entire video
saveVideo(video_frames,video_save_path,
    fps=round(sum(frame_avg)/len(frame_avg),1),
    size=(image_width,image_height))

