import math
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from tracker import kalman_filter


def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    # print(cost_matrix)
    # change
    for i in range(len(cost_matrix[0])):
        mat = cost_matrix[:, i]
        if min(mat) == 1:
            cost_matrix[:, i] += 0.2
    # change
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0: 
            if cost_matrix[ix][mx] < 1:
                matches.append([ix, mx])
            else:
                x[ix] = -1
                y[mx] = -1
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    predict_box = np.ascontiguousarray(atlbrs, dtype=np.float)
    
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious, predict_box

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious, predict_box


def tlbr_expand(tlbr, scale=1.2):
    w = tlbr[2] - tlbr[0]
    h = tlbr[3] - tlbr[1]

    half_scale = 0.5 * scale

    tlbr[0] -= half_scale * w
    tlbr[1] -= half_scale * h
    tlbr[2] += half_scale * w
    tlbr[3] += half_scale * h

    return tlbr


def iou_distance(atracks, btracks, H=[], rotation=[], optical=[]):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    e = 2
    lamda = 1
    i_w = 1242
    i_h = 375
    max_e = 3
    min_e = 1.5
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    #change Iterative compensation
    
    # if H != []:
    #     for idx, x in enumerate(atlbrs):
    #         # print("OLD")
    #         # print(atlbrs[idx])
    #         # self compesation test
    #         H = np.array(H)
    #         R = H[:2,:2]
    #         # print("R")
    #         # print(R)
    #         T = H[:2, 2]
    #         T = [[T[0]], [T[1]]]
    #         # print("T")
    #         # print(T)
    #         loc = [[atlbrs[idx][0]], [atlbrs[idx][1]]]
    #         w = atlbrs[idx][2] - atlbrs[idx][0] 
    #         h = atlbrs[idx][3] - atlbrs[idx][1]
    #         new_loc = R.dot(loc) + T
    #         # print(new_loc)
    #         atlbrs[idx][0] = new_loc[0]
    #         atlbrs[idx][1] = new_loc[1]
    #         atlbrs[idx][2] = new_loc[0] + w
    #         atlbrs[idx][3] = new_loc[1] + h
    #         # print("NEW")
    #         # print(atlbrs[idx])
    #change
    
    #change  EXPANSION
    gap_b_x = []
    gap_b_y = []
    gap_b_s = []
    e = []
    for idx, x in enumerate(btlbrs):
        gap_b_x.append([])
        gap_b_y.append([])
        gap_b_s.append([])
        w = abs(btlbrs[idx][2] - btlbrs[idx][0]) 
        h = abs(btlbrs[idx][3] - btlbrs[idx][1])
        for idx2, x2 in enumerate(btlbrs):
            if idx2 != idx:
                gap_b_x[idx].append(abs(btlbrs[idx][0] - btlbrs[idx2][0]))
                gap_b_y[idx].append(abs(btlbrs[idx][1] - btlbrs[idx2][1]))
                gap_b_s[idx].append(abs(math.sqrt(math.pow(btlbrs[idx][0] - btlbrs[idx2][0], 2)+math.pow(btlbrs[idx][1] - btlbrs[idx2][1], 2))))
        if gap_b_s[idx] != []:
            value_ew = gap_b_x[idx][gap_b_s[idx].index(min(gap_b_s[idx]))]/w
            value_eh = gap_b_y[idx][gap_b_s[idx].index(min(gap_b_s[idx]))]/h
            if value_ew > max_e:
                value_ew = max_e
            if value_ew < min_e:
                value_ew = min_e
            if value_eh > max_e:
                value_eh = max_e
            if value_eh < min_e:
                value_eh = min_e
            e.append([value_ew, value_eh])
        else:
            e.append([max_e, max_e])
    for idx, x in enumerate(btlbrs):
        w = btlbrs[idx][2] - btlbrs[idx][0] 
        h = btlbrs[idx][3] - btlbrs[idx][1]
        # print([e[idx][0], e[idx][1]])
        ew = ((w*e[idx][0]) - w)/2
        eh = ((h*e[idx][1]) - h)/2
        # ew = ((w*1.5) - w)/2
        # eh = ((h*1.5) - h)/2
        btlbrs[idx][0] -= ew
        btlbrs[idx][1] -= eh
        btlbrs[idx][2] += ew
        btlbrs[idx][3] += eh
    for idx, x in enumerate(atlbrs):
        gap_a = []
        for idx2, x2 in enumerate(btlbrs):
            gap_a.append(math.sqrt(math.pow(atlbrs[idx][0] - btlbrs[idx2][0], 2)+math.pow(atlbrs[idx][1] - btlbrs[idx2][1], 2)))
        if gap_a != []:
            e_a_x = e[gap_a.index(min(gap_a))][0]
            e_a_y = e[gap_a.index(min(gap_a))][1]
        else:
            e_a_x = 1
            e_a_y = 1
        w = atlbrs[idx][2] - atlbrs[idx][0] 
        h = atlbrs[idx][3] - atlbrs[idx][1]
        ew = ((w*e_a_x) - w)/2
        eh = ((h*e_a_y) - h)/2
        # ew = ((w*1.5) - w)/2
        # eh = ((h*1.5) - h)/2
        atlbrs[idx][0] -= ew
        atlbrs[idx][1] -= eh
        atlbrs[idx][2] += ew
        atlbrs[idx][3] += eh
    #change
    # if len(atlbrs) > 0 and len(btlbrs) > 0:
    #     print("matching")
    #     print([atlbrs, btlbrs])
    #     p = input()
    
    # opticalflow expanding
    # if optical != []:
    #     _ious = []
    #     for i, a in enumerate(btlbrs):
    #         loc = [btlbrs[i][0], btlbrs[i][1], btlbrs[i][2], btlbrs[i][3]]
    #         w = btlbrs[i][2] - btlbrs[i][0] 
    #         h = btlbrs[i][3] - btlbrs[i][1]
    #         x_bias = []
    #         y_bias = []
    #         for j, _p in enumerate(optical[1]):
    #             p = _p[0]
    #             if p[0] >= loc[0] and p[0] <= loc[0]+w and p[1] >= loc[1] and p[1] <= loc[1]+h:
    #                 x_bias.append(abs(p[0] - optical[0][j][0][0]))
    #                 y_bias.append(abs(p[1] - optical[0][j][0][1]))
    #         x_bias = np.array(x_bias)
    #         y_bias = np.array(y_bias)
    #         if len(x_bias) != 0 and len(y_bias) != 0:
    #             x_mid = np.median(x_bias)
    #             y_mid = np.median(y_bias)
    #         else:
    #             x_mid = 0
    #             y_mid = 0
    #         # print([x_mid, y_mid])
    #         # e_x = 1 + (x_mid*2/w)
    #         # e_y = 1 + (y_mid*2/h)
    #         # if e_x < 1.1:
    #         #     e_x = 1.1
    #         # elif e_x > 2:
    #         #     e_x = 2
    #         # if e_y < 1.1:
    #         #     e_y = 1.1
    #         # elif e_y > 2:
    #         #     e_y = 2
    #         # ew = ((w*e_x) - w)/2
    #         # eh = ((h*e_y) - h)/2
    #         ew = x_mid
    #         eh = y_mid
    #         # print([e_x, e_y])
    #         e_btlbrs = [[loc[0] - ew, loc[1] - eh, loc[2] + ew, loc[3] + eh]]
    #         e_atlbrs = []
    #         for idx, x in enumerate(atlbrs):
    #             w = atlbrs[idx][2] - atlbrs[idx][0] 
    #             h = atlbrs[idx][3] - atlbrs[idx][1]
    #             # ew = ((w*e_x) - w)/2
    #             # eh = ((h*e_y) - h)/2
    #             ew = x_mid
    #             eh = y_mid
    #             e_atlbrs.append([atlbrs[idx][0] - ew, atlbrs[idx][1] - eh, atlbrs[idx][2] + ew, atlbrs[idx][3] + eh])
    #         s_ious, predict_box = ious(e_atlbrs, e_btlbrs)
    #         if len(_ious) != 0:
    #             _ious = np.append(_ious, s_ious,axis=1)
    #         else:
    #             _ious = s_ious
                    
    # else:        
    # opticalflow expanding
    
    _ious, predict_box = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix, predict_box


def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)

    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # / 2.0  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    # measurements = np.asarray([det.to_xyah() for det in detections])
    measurements = np.asarray([det.to_xywh() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    # measurements = np.asarray([det.to_xyah() for det in detections])
    measurements = np.asarray([det.to_xywh() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost

def gate(cost_matrix, emb_cost):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    if cost_matrix.size == 0:
        return cost_matrix
    
    index = emb_cost > 0.3
    cost_matrix[index] = 1
 
    return cost_matrix

def eiou_distance(atracks, btracks, expand):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.last_tlbr for track in atracks]
        btlbrs = [track.last_tlbr for track in btracks]
    _ious = eious(atlbrs, btlbrs, expand)
    cost_matrix = 1 - _ious

    return cost_matrix

def eious(atlbrs, btlbrs, e):
    """
    Compute cost based on EIoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    eious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if eious.size == 0:
        return eious

    atlbrs = np.array([expand(tlbr, e) for tlbr in atlbrs])
    btlbrs = np.array([expand(tlbr, e) for tlbr in btlbrs])

    eious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return eious

def expand(tlbr, e):
    
    t,l,b,r = tlbr
    w = r-l
    h = b-t
    expand_w = 2*w*e + w
    expand_h = 2*h*e + h

    new_tlbr = [t-expand_h//2,l-expand_w//2,b+expand_h//2,r+expand_w//2]

    return new_tlbr

def fistMot(atracks, btracks):
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
        
    _ious, predict_box = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix, predict_box