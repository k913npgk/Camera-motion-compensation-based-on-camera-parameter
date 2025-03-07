import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

from tracker import matching
from tracker.gmc import GMC
from tracker.basetrack import BaseTrack, TrackState
from tracker.kalman_filter import KalmanFilter

from fast_reid.fast_reid_interfece import FastReIDInterface

from tracker.image_similarity import image_similarity_model

import copy

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, feat=None, feat_history=50):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        # eiou
        self.last_tlwh = self._tlwh
        
        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            # IM = [[1, 0], [0, 1]]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            # R8x8[:2, :2] = R
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                # print("BM-----------------")
                # print(mean)
                mean = R8x8.dot(mean)
                mean[:2] += t
                # print("AM-----------------")
                # print(mean)
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        
        # eiou
        self.last_tlwh = new_track.tlwh
        
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        # eiou
        self.last_tlwh = new_tlwh
        
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def last_tlbr(self):
        ret = self.last_tlwh.copy()
        ret[2:] += ret[:2]
        return ret
    
    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BoTSORT(object):
    def __init__(self, args, frame_rate=30):

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.frame_id = 0
        self.args = args

        self.track_high_thresh = args.track_high_thresh
        self.track_low_thresh = args.track_low_thresh
        self.new_track_thresh = args.new_track_thresh

        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        if args.with_reid:
            self.encoder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, args.device)

        self.gmc = GMC(method=args.cmc_method, verbose=[args.name, args.ablation])
        
        # self.image_similarity = image_similarity_model()
        self.error_list = []

    # error frame detection (compare Appearance of bounding box)
    # input : predict box, img(this frame), last frame detection, last frame img
    def errorParameterDetect(self, strack_pool, img, last_strack_pool, prev_img):
        mm = []
        lmm = []
        for i, st in enumerate(last_strack_pool):
            if st.state == TrackState.Tracked:
                lmm.append(st.mean.copy())
                mm.append(strack_pool[i].mean.copy())
        score_list = []
        mm = np.array(mm)
        lmm = np.array(lmm)
        print(mm.shape)
        if mm.shape[0] == 0:
            return True
        for i, m in enumerate(mm):
            m_int = m.astype(int)
            lmm_int = lmm.astype(int)
            t = img[m_int[1]:m_int[1]+m_int[3], m_int[0]:m_int[0]+m_int[2]]
            lt = prev_img[lmm_int[i][1]:lmm_int[i][1]+lmm_int[i][3], lmm_int[i][0]:lmm_int[i][0]+lmm_int[i][2]]
            if t.size == 0 or lt.size == 0:
                print('over')
            else:
                score = self.image_similarity.predict(t, lt)
                score_list.append(score[0])
                print(score)
        score_list = np.array(score_list)
        if len(score_list[score_list >= 0.5]) < 1:
            return True
        else:
            return False

    def update(self, output_results, img, img_info, camera_parameter, prev_camera_parameter, next_camera_parameter):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        if len(output_results):
            if output_results.shape[1] > 6:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
                classes = output_results[:, 5]
                truncated = output_results[:, 6]
                occluded = output_results[:, 7]
                alpha = output_results[:, 8]
                dimensions = output_results[:, 9:12]
                location = output_results[:, 12:15]
                rotation_y = output_results[:, 15]
            elif output_results.shape[1] == 6:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
                classes = output_results[:, -1]
            else:
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2
                classes = output_results[:, -1]
            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]
            if output_results.shape[1] > 6:
                truncated = truncated[lowest_inds]
                occluded = occluded[lowest_inds]
                alpha = alpha[lowest_inds]
                dimensions = dimensions[lowest_inds]
                location = location[lowest_inds]
                rotation_y = rotation_y[lowest_inds]
            # Find high threshold detections
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]
            if output_results.shape[1] > 6:
                truncated_keep = truncated[remain_inds]
                occluded_keep = occluded[remain_inds]
                alpha_keep = alpha[remain_inds]
                dimensions_keeps = dimensions[remain_inds]
                location_keep = location[remain_inds]
                rotation_y_keep = rotation_y[remain_inds]

        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []

        '''Extract embeddings '''
        if self.args.with_reid:
            features_keep = self.encoder.inference(img, dets)

        if len(dets) > 0:
            '''Detections'''
            if self.args.with_reid:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f) for
                              (tlbr, s, f) in zip(dets, scores_keep, features_keep)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                              (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        # add information for KITTI
        for i, _ in enumerate(detections):
            detections[i].classes = int(classes_keep[i])
        #     detections[i].truncated = truncated_keep[i]
        #     detections[i].occluded = occluded_keep[i]
        #     detections[i].alpha = alpha_keep[i]
        #     detections[i].dimensions = dimensions_keeps[i]
        #     detections[i].location = location_keep[i]
        #     detections[i].rotation_y = rotation_y_keep[i]


        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
    
        # Fix camera motion
        curr = [0,0]
        warp, rotation, skip_check = self.gmc.apply(img, camera_parameter, prev_camera_parameter, next_camera_parameter, dets)
        # print(warp)
        if skip_check:
            return [], [], [], skip_check
        # last_strack_pool = strack_pool.copy()
        # if len(strack_pool) > 0:
        #     print(strack_pool[0].tlbr)
        # Predict the current location with KF
        # last_pos = []
        # for i, _ in enumerate(strack_pool):
        #     last_pos.append(strack_pool[i].tlbr)
        STrack.multi_predict(strack_pool)
        # mean_nc = []
        # cov_nc = []
        # for i, _ in enumerate(strack_pool):
        #     mean_nc.append(strack_pool[i].mean)
        #     cov_nc.append(strack_pool[i].covariance)
        # if len(strack_pool) > 0:
        #     print(strack_pool[0].tlbr)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)
        
        # opticalflow expanding
        # frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3,
        #                                useHarrisDetector=False, k=0.04)
        # keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **feature_params)
        # if self.frame_id != 1:
        #     matchedKeypoints, status, err = cv2.calcOpticalFlowPyrLK(self.prevFrame, frame, self.prevKeyPoints, None)
        #     optical_point = [self.prevKeyPoints, matchedKeypoints]
        # else:
        #     optical_point = []
        # self.prevFrame = frame.copy()
        # self.prevKeyPoints = copy.copy(keypoints)
        
        # opticalflow expanding
        
        # if len(strack_pool) > 0:
        #     print(strack_pool[0].tlbr)
        #     n= input()
        # error_check = self.errorParameterDetect(strack_pool, img, last_strack_pool, prev_img)
        # if error_check == True:
        #     self.error_list.append(self.frame_id)
        #test
        # t_focal_matrix, t_camera_parameter = camera_parameter[0], camera_parameter[1]
        # t_prev_focal_matrix, t_prev_camera_parameter = prev_camera_parameter[0], prev_camera_parameter[1]
        # for b in strack_pool:
        #     tlwh = b.tlbr.copy()
        #     print(tlwh)
        #     t_r = [[i[0], i[1], i[2]] for i in t_camera_parameter]
        #     t_pr = [[i[0], i[1], i[2]] for i in t_prev_camera_parameter] 
        #     t_t = [i[3] for i in t_camera_parameter] 
        #     t_pt = [i[3] for i in t_prev_camera_parameter] 
        #     r = np.dot(np.linalg.pinv(t_r), t_pr)
        #     t = np.array(t_pt) - np.array(t_t)
        #     r = [[r[0][0], r[0][1]], [r[1][0], r[1][1]]]
        #     t = [t[0], t[1]]
        #     temp_1 = np.dot(r, [tlwh[0], tlwh[1]]) + t
        #     temp_2 = np.dot(r, [tlwh[2], tlwh[3]]) + t
        #     b.tlbr[0] = temp_1[0]
        #     b.tlbr[1] = temp_1[1]
        #     b.tlbr[2] = temp_2[0]
        #     b.tlbr[3] = temp_2[1]
        #     print(b.tlbr[0])
        #     print(b.tlbr[1])
        #     print(b.tlbr[2])
        #     print(b.tlbr[3])
        #test    
            
        # DEPTH ESTIMATE
        # def depth_get(pool, k):
        #     depth_list = []
        #     for boxes in pool:
        #         ret = boxes.tlwh.copy()
        #         cx = ret[0] + 0.5 * ret[2]
        #         y2 = ret[1] +  ret[3]
        #         lendth = 2000 - y2
        #         depth_list.append(lendth)
            
        #     if len(depth_list) == 0:
        #         all_list = []
        #         for i in range(k):
        #             all_list.append([])
        #         return all_list
            
        #     max_len, mix_len = 2000, 0
            
        #     depth_step = (max_len - mix_len + 1) / k
            
        #     now = mix_len
        #     all_list = []
        #     while now <= max_len:
        #         now_list = []
        #         for boxes in pool:
        #             ret = boxes.tlwh.copy()
        #             cx = ret[0] + 0.5 * ret[2]
        #             y2 = ret[1] +  ret[3]
        #             lendth = 2000 - y2
        #             if now <= lendth < now + depth_step:
        #                 now_list.append(boxes)
        #         all_list.append(now_list)
        #         now += depth_step
        #     return all_list
        
        
        # strack_pool_s = depth_get(strack_pool, 6)
        # detections_s = depth_get(detections, 6)
        # for idx, pool in enumerate(strack_pool_s):
        #     ious_dists, one_box = matching.iou_distance(strack_pool_s[idx], detections_s[idx])
        #     ious_dists_mask = (ious_dists > self.proximity_thresh) 

        #     if not self.args.mot20:
        #         ious_dists = matching.fuse_score(ious_dists, detections_s[idx])

        #     if self.args.with_reid:
        #         emb_dists = matching.embedding_distance(strack_pool_s[idx], detections_s[idx]) / 2.0
        #         raw_emb_dists = emb_dists.copy()
        #         emb_dists[emb_dists > self.appearance_thresh] = 1.0
        #         emb_dists[ious_dists_mask] = 1.0
        #         dists = np.minimum(ious_dists, emb_dists)
                
        #     else:
        #         dists = ious_dists

        #     matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
            
        #     for itracked, idet in matches:
        #         track = strack_pool_s[idx][itracked]
        #         det = detections_s[idx][idet]
        #         if track.state == TrackState.Tracked:
        #             track.update(detections_s[idx][idet], self.frame_id)
        #             activated_starcks.append(track)
        #         else:
        #             track.re_activate(det, self.frame_id, new_id=False)
        #             refind_stracks.append(track)
        #     # if idx < 10:
        #     #     strack_pool_s[idx+1] += u_track
        #     #     detections_s[idx+1] += u_detection
        #     print(strack_pool_s)
        
        ious_dists, one_box = matching.iou_distance(strack_pool, detections)
        # eiou
        # init_expand_scale = 0.7
        # expand_scale_step = 0.1
        # cur_expand_scale = init_expand_scale + expand_scale_step * self.frame_id
        # ious_dists = matching.eiou_distance(strack_pool, detections, cur_expand_scale)
        # eiou
        ious_dists_mask = (ious_dists > self.proximity_thresh)
        # print(ious_dists)
        
        # if self.frame_id == 1:
        #     ious_dists, one_box = matching.iou_distance(strack_pool, detections)
        #     ious_dists_mask = (ious_dists > self.proximity_thresh) 
        # else:
        #     ious_dists = []
        #     one_box = []
        #     for idx, pool in enumerate(strack_pool_s):
        #         s_ious_dists, s_one_box = matching.iou_distance(strack_pool_s[idx], detections_s[idx])
        #         print(s_ious_dists)
        #         if len(ious_dists) == 0:
        #             ious_dists = s_ious_dists
        #         else:
        #             if len(s_ious_dists) != 0:
        #                 ious_dists += s_ious_dists
        #         for y in s_one_box:
        #             one_box.append(y)
        #     if len(strack_pool_s) > 0:
        #         ious_dists = ious_dists/len(strack_pool_s)
        # ious_dists_mask = (ious_dists > self.proximity_thresh)

        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            # hm
            # dists = HM(ious_dists, emb_dists)
            # hm
            dists = np.minimum(ious_dists, emb_dists)
            # Popular ReID method (JDE / FairMOT)
            # raw_emb_dists = matching.embedding_distance(strack_pool, detections)
            # dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections)
            # emb_dists = dists

            # IoU making ReID
            # dists = matching.embedding_distance(strack_pool, detections)
            # dists[ious_dists_mask] = 1.0
        else:
            dists = ious_dists

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=1.1) # self.args.match_thresh
        
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            # fuse_bbox = detections[idet]
            # fuse_bbox._tlwh = np.asarray([fuse_bbox.tlwh[0]*0.5 + track.tlwh[0]*0.5, fuse_bbox.tlwh[1]*0.5 + track.tlwh[1]*0.5, fuse_bbox.tlwh[2]*0.5 + track.tlwh[2]*0.5, fuse_bbox.tlwh[3]*0.5 + track.tlwh[3]*0.5], dtype=np.float)
            # change Iterative compensation
            # print("before1")
            # print([detections[idet].tlwh[0], detections[idet].tlwh[1]])
            # alt_R = [[warp[0][0], warp[1][0]], [warp[0][1], warp[1][1]]]
            # alt_loc = np.dot(alt_R, [detections[idet].tlwh[0], detections[idet].tlwh[1]]) - [warp[0][2], warp[1][2]]
            # detections[idet]._tlwh[0], detections[idet]._tlwh[1] = alt_loc[0], alt_loc[1]
            # print("after1")
            # print([detections[idet].tlwh[0], detections[idet].tlwh[1]])
            # b =input()
            # change
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
                # change
                # warp_t = [detections[idet].tlbr[0] - last_pos[itracked][0], detections[idet].tlbr[1] - last_pos[itracked][1]]
                # mean_nc[itracked][:2] += warp_t
                
                # track.mean = mean_nc[itracked]
                # track.covariance = cov_nc[itracked]
                
                # track.update(detections[idet], self.frame_id)
                # activated_starcks.append(track)
                # change
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            classes_second = classes[inds_second]
        else:
            dets_second = []
            scores_second = []
            classes_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists, two_box = matching.iou_distance(r_tracked_stracks, detections_second, H=warp, rotation=rotation)
        # eiou
        # dists = matching.eiou_distance(r_tracked_stracks, detections_second, expand=0.5)
        # eiou
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            # fuse_bbox = detections[idet]
            # fuse_bbox._tlwh = np.asarray([fuse_bbox.tlwh[0]*0.5 + track.tlwh[0]*0.5, fuse_bbox.tlwh[1]*0.5 + track.tlwh[1]*0.5, fuse_bbox.tlwh[2]*0.5 + track.tlwh[2]*0.5, fuse_bbox.tlwh[3]*0.5 + track.tlwh[3]*0.5], dtype=np.float)
            # change Iterative compensation
            # print("before2")
            # print([detections[idet].tlwh[0], detections[idet].tlwh[1]])
            # alt_R = [[warp[0][0], warp[1][0]], [warp[0][1], warp[1][1]]]
            # alt_loc = np.dot(alt_R, [detections[idet].tlwh[0], detections[idet].tlwh[1]]) - [warp[0][2], warp[1][2]]
            # detections[idet]._tlwh[0], detections[idet]._tlwh[1] = alt_loc[0], alt_loc[1]
            # print("after2")
            # print([detections[idet].tlwh[0], detections[idet].tlwh[1]])
            # b =input()
            # change
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
                # change
                # warp_t = [detections[idet].tlbr[0] - last_pos[itracked][0], detections[idet].tlbr[1] - last_pos[itracked][1]]
                # mean_nc[itracked][:2] += warp_t
                
                # track.mean = mean_nc[itracked]
                # track.covariance = cov_nc[itracked]
                
                # track.update(detections[idet], self.frame_id)
                # activated_starcks.append(track)
                # change
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        ious_dists, three_box = matching.iou_distance(unconfirmed, detections, H=warp, rotation=rotation)
        # eiou
        # ious_dists = matching.eiou_distance(unconfirmed, detections, 0.5)
        # eiou
        ious_dists_mask = (ious_dists > self.proximity_thresh)
        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            # hm
            # dists = HM(ious_dists, emb_dists)
            # hm
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            # change
            loc = track.tlbr
            # change
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
            # change
            elif loc[0] > 0 or loc[1] < 0 or loc[2] > img_info['height'] or loc[3] > img_info['width']:
                track.mark_removed()
                removed_stracks.append(track)
            # change

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]
        # np.concatenate((one_box, two_box), axis=0)
        # np.concatenate((one_box, three_box), axis=0)
        return output_stracks, one_box, curr, skip_check


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist, unuse_box = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

def HM(d1, d2):
    newCostMatrix = np.empty_like(d1)
    for i, _n in enumerate(d1):
        for j, n in enumerate(_n):
            newCostMatrix[i][j] = 2/(1/d1[i][j] + 1/d2[i][j])
    return newCostMatrix
