import os
import time
from threading import Thread
from queue import Queue

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp

from alphapose.utils.transforms import get_func_heatmap_to_coord
from alphapose.utils.pPose_nms import pose_nms, write_json

#jun
import matplotlib.pyplot as plt

DEFAULT_VIDEO_SAVE_OPT = {
    'savepath': 'examples/res/1.mp4',
    'fourcc': cv2.VideoWriter_fourcc(*'mp4v'),
    'fps': 25,
    'frameSize': (640, 480)
}

EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


class DataWriter():
    def __init__(self, cfg, opt, save_video=False,
                 video_save_opt=DEFAULT_VIDEO_SAVE_OPT,
                 queueSize=1024):
        self.cfg = cfg
        self.opt = opt
        self.video_save_opt = video_save_opt

        self.eval_joints = EVAL_JOINTS
        self.save_video = save_video
        self.heatmap_to_coord = get_func_heatmap_to_coord(cfg)
        # initialize the queue used to store frames read from
        # the video file
        if opt.sp:
            self.result_queue = Queue(maxsize=queueSize)
        else:
            self.result_queue = mp.Queue(maxsize=queueSize)

        if opt.save_img:
            if not os.path.exists(opt.outputpath + '/vis'):
                os.mkdir(opt.outputpath + '/vis')

        if opt.pose_flow:
            from trackers.PoseFlow.poseflow_infer import PoseFlowWrapper
            self.pose_flow_wrapper = PoseFlowWrapper(save_path=os.path.join(opt.outputpath, 'poseflow'))

    def start_worker(self, target):
        if self.opt.sp:
            p = Thread(target=target, args=())
        else:
            p = mp.Process(target=target, args=())
        # p.daemon = True
        p.start()
        return p

    def start(self):
        # start a thread to read pose estimation results per frame
        self.result_worker = self.start_worker(self.update)
        return self

    def update(self):
        final_result = []
        norm_type = self.cfg.LOSS.get('NORM_TYPE', None)
        hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        if self.save_video:
            # initialize the file video stream, adapt ouput video resolution to original video
            stream = cv2.VideoWriter(*[self.video_save_opt[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
            if not stream.isOpened():
                print("Try to use other video encoders...")
                ext = self.video_save_opt['savepath'].split('.')[-1]
                fourcc, _ext = self.recognize_video_ext(ext)
                self.video_save_opt['fourcc'] = fourcc
                self.video_save_opt['savepath'] = self.video_save_opt['savepath'][:-4] + _ext
                stream = cv2.VideoWriter(*[self.video_save_opt[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
            assert stream.isOpened(), 'Cannot open video for writing'
        # keep looping infinitelyd

        
        #jun
        isLastFrameFirst=True
        compared_pose_values=[]
        print("start check!")
        while True:
            

            # ensure the queue is not empty and get item
            (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name) = self.wait_and_get(self.result_queue)
            if orig_img is None:
                # if the thread indicator variable is set (img is None), stop the thread
                if self.save_video:
                    stream.release()
                write_json(final_result, self.opt.outputpath, form=self.opt.format, for_eval=self.opt.eval)
                print("Results have been written to json.")

                # jun
                print(compared_pose_values)
                plt.plot([i for i in range(len(compared_pose_values))],compared_pose_values)
                max_difference_frame=compared_pose_values.index(min(compared_pose_values)) 
                plt.title("Most diffrent frame: "+str(max_difference_frame)+"->"+str(max_difference_frame+1)+"/"+str(len(compared_pose_values)))
                plt.xlabel("Frame")
                plt.ylabel("Similarity")
                plt.savefig("video_output/compared_value_fig_1.png",dpi=300)
                print("Save compared value plot between frame")
                # jun

                return
            # image channel RGB->BGR
            orig_img = np.array(orig_img, dtype=np.uint8)[:, :, ::-1]
            if boxes is None or len(boxes) == 0:
                if self.opt.save_img or self.save_video or self.opt.vis:
                    self.write_image(orig_img, im_name, stream=stream if self.save_video else None)
            else:
                # location prediction (n, kp, 2) | score prediction (n, kp, 1)
                assert hm_data.dim() == 4
                #pred = hm_data.cpu().data.numpy()

                if hm_data.size()[1] == 136:
                    self.eval_joints = [*range(0,136)]
                elif hm_data.size()[1] == 26:
                    self.eval_joints = [*range(0,26)]
                pose_coords = []
                pose_scores = []
                print("human number",hm_data.shape[0])
                for i in range(hm_data.shape[0]):
                    bbox = cropped_boxes[i].tolist()
                    pose_coord, pose_score = self.heatmap_to_coord(hm_data[i][self.eval_joints], bbox, hm_shape=hm_size, norm_type=norm_type)
                    # print("pose_coord",len(pose_coord),pose_coord)
                    
                    #jun : 이전 frame과 현재 frame간의 값 비교 
                    if isLastFrameFirst:
                        last_frame_keypoints=pose_coord
                        last_pose_score=pose_score
                        isLastFrameFirst=False
                    else:
                        # print("check_last",last_frame_keypoints)
                        # print("check_current",pose_coord)
                        #방법 1. 단순히 각 x,y좌표끼리 거리 구해서 더한 것으로 frame간의 값 비교
                        # save_compared_value=abs(np.array(last_frame_keypoints)-np.array(pose_coord))
                        # last_frame_keypoints=pose_coord
                        # compared_pose_values.append(np.sum(save_compared_value))

                        #방법 2. OKS로 frame간의 값 비교 
                        compared_value=self.computeOks(last_frame_keypoints, pose_coord,last_pose_score,bbox,confidence_threshold=0.5)
                        print("compared_value",compared_value)
                        compared_pose_values.append(compared_value)
                        last_frame_keypoints=pose_coord
                        last_pose_score=pose_score


                        # print("compared",np.sum(save_compared_value))

                    #jun : 이전 frame과 현재 frame간의 값 비교


                    pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                    # print("pose_coord shape",torch.from_numpy(pose_coord).unsqueeze(0).shape)
                    pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
                # print("pose_coords",pose_coords,len(pose_coords))
                preds_img = torch.cat(pose_coords)
                # print("preds_img",preds_img.shape)
                preds_scores = torch.cat(pose_scores)
                if not self.opt.pose_track:
                    boxes, scores, ids, preds_img, preds_scores, pick_ids = \
                        pose_nms(boxes, scores, ids, preds_img, preds_scores, self.opt.min_box_area)

                _result = []
                for k in range(len(scores)):
                    _result.append(
                        {
                            'keypoints':preds_img[k],
                            'kp_score':preds_scores[k],
                            'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                            'idx':ids[k],
                            'box':[boxes[k][0], boxes[k][1], boxes[k][2]-boxes[k][0],boxes[k][3]-boxes[k][1]] 
                        }
                    )

                result = {
                    'imgname': im_name,
                    'result': _result
                }


                if self.opt.pose_flow:
                    poseflow_result = self.pose_flow_wrapper.step(orig_img, result)
                    for i in range(len(poseflow_result)):
                        result['result'][i]['idx'] = poseflow_result[i]['idx']

                final_result.append(result)
                if self.opt.save_img or self.save_video or self.opt.vis:
                    if hm_data.size()[1] == 49:
                        from alphapose.utils.vis import vis_frame_dense as vis_frame
                    elif self.opt.vis_fast:
                        from alphapose.utils.vis import vis_frame_fast as vis_frame
                    else:
                        from alphapose.utils.vis import vis_frame
                    img = vis_frame(orig_img, result, self.opt)
                    self.write_image(img, im_name, stream=stream if self.save_video else None)

    def write_image(self, img, im_name, stream=None):
        if self.opt.vis:
            cv2.imshow("AlphaPose Demo", img)
            cv2.waitKey(30)
        if self.opt.save_img:
            cv2.imwrite(os.path.join(self.opt.outputpath, 'vis', im_name), img)
        if self.save_video:
            stream.write(img)

    def wait_and_put(self, queue, item):
        queue.put(item)

    def wait_and_get(self, queue):
        return queue.get()

    def save(self, boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name):
        # save next frame in the queue
        self.wait_and_put(self.result_queue, (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name))

    def running(self):
        # indicate that the thread is still running
        return not self.result_queue.empty()

    def count(self):
        # indicate the remaining images
        return self.result_queue.qsize()

    def stop(self):
        # indicate that the thread should be stopped
        self.save(None, None, None, None, None, None, None)
        self.result_worker.join()

    def terminate(self):
        # directly terminate
        self.result_worker.terminate()

    def clear_queues(self):
        self.clear(self.result_queue)
        
    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def results(self):
        # return final result
        print(self.final_result)
        return self.final_result

    def recognize_video_ext(self, ext=''):
        if ext == 'mp4':
            return cv2.VideoWriter_fourcc(*'mp4v'), '.' + ext
        elif ext == 'avi':
            return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
        elif ext == 'mov':
            return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
        else:
            print("Unknow video format {}, will use .mp4 instead of it".format(ext))
            return cv2.VideoWriter_fourcc(*'mp4v'), '.mp4'


    #computeOks
    def computeOks(self, last_frame_keypoints, current_frame_keypoints,last_pose_score,bbox,confidence_threshold=0.5):
        # print("last_frame_keypoints",last_frame_keypoints,last_frame_keypoints.shape)
        # print("current_frame_keypoints",current_frame_keypoints)
        # print("last_pose_score",last_pose_score)
        # print("bbox",bbox)
        
        last_frame_keypoints_np=np.array(last_frame_keypoints)
        last_pose_score_np=np.array(last_pose_score)
        last_frame_keypoints_np=np.insert(last_frame_keypoints_np,2,last_pose_score_np.flatten(),axis=1)
        last_frame_keypoints=list(last_frame_keypoints_np.flatten())
        current_frame_keypoints_np=np.array(current_frame_keypoints).flatten()

        ious=0
        #sigmas가 뭔지 coco값 그대로 들어가는게 맞는지 확인
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
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
        vg=np.array(vg)
        #k1 : visibility flag True인 애들의 개수 가져오기
        k1 = np.count_nonzero(vg > confidence_threshold)

        #bounding box 처리
        bb = bbox
        #bb[0] : top left x,bb[1]:top left y,bb[2]:width,bb[3]:height
        #x0 : down left x
        #x1 : ?
        #y0 : down left y
        #y1 : ?
        x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
        y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
        
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
        #threshold중 keypoint가 confidence score 보다 높은 것들이 아예 없을 경우 
        else:
            # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
            print("nothing!")
            z = np.zeros((k))
            dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
            dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
        #밑에서부터 OKS 수식 구현한 것 
        #vars = (sigmas * 2)**2
        # e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
        last_frame_area=bb[2]*bb[3]
        e = (dx**2 + dy**2) / vars / (last_frame_area+np.spacing(1)) / 2
        #나중에 confidence score높은 애들만 계산하는걸로 고치면 될듯?
        if k1 > 0:
            e=e[vg > confidence_threshold]
        #np.sum(np.exp(-e)) / e.shape[0]값 그대로 출력하면 될듯
        ious = np.sum(np.exp(-e)) / e.shape[0]

        return ious

