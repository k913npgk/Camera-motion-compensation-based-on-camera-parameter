import argparse
import os
import sys
import os.path as osp
import cv2
import numpy as np
import torch

sys.path.append('.')

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking

from tracker.tracking_utils.timer import Timer
from tracker.bot_sort import BoTSORT

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

# Global
trackerTimer = Timer()
timer = Timer()


def make_parser():
    parser = argparse.ArgumentParser("BoT-SORT Tracks For Evaluation!")

    parser.add_argument("path", help="path to dataset under evaluation, currently only support MOT17 and MOT20.")
    parser.add_argument("--benchmark", dest="benchmark", type=str, default='MOT17', help="benchmark to evaluate: MOT17 | MOT20")
    parser.add_argument("--eval", dest="split_to_eval", type=str, default='test', help="split to evaluate: train | val | test")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("--default-parameters", dest="default_parameters", default=False, action="store_true", help="use the default parameters as in the paper")
    parser.add_argument("--save-frames", dest="save_frames", default=False, action="store_true", help="save sequences with tracks.")

    # Detector
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold valid for tracks")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')

    # CMC
    parser.add_argument("--cmc-method", default="file", type=str, help="cmc method: files (Vidstab GMC) | sparseOptFlow | orb | ecc | none")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="use Re-ID flag.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')

    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            device=torch.device("cpu"),
            fp16=False
    ):
        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16

        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        if img is None:
            raise ValueError("Empty image: ", img_info["file_name"])

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)

        return outputs, img_info


def image_track(predictor, vis_folder, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()

    if args.ablation:
        files = files[len(files) // 2 + 1:]

    num_frames = len(files)

    # Tracker
    tracker = BoTSORT(args, frame_rate=args.fps)

    results = []
    
    camera_parameter = []
    # get camera parameter
    f = open('C:\\Users\\CSY\\SMILEtrack-main\\BoT-SORT\\datasets\\sports\\kitti_test\\re0013.txt', 'r')
    for line in f.readlines():
        camera_parameter.append(eval(line))
    f.close
    skip_check = False
    # get detections
    # regionlets = [[]]
    # frame_count = 0
    # classes = ['Cyclist', 'Car', 'Pedestrian', 'Person', 'Van', "Truck", "Tram"]
    # f = open('C:\\Users\\CSY\\SMILEtrack-main\\BoT-SORT\\datasets\\sports\\lsvm0013.txt', 'r')
    # for line in f.readlines():
    #     temp = line.split()
    #     while int(temp[0]) != frame_count:
    #         # print(frame_count)
    #         # for i in regionlets[-1]:
    #         #     if i[4] >= 0.6:
    #         #         print([i[0], i[1], i[2], i[3], i[4]])
    #         # i = input()
    #         frame_count += 1
    #         regionlets.append([])
    #     if temp[2] != 'DontCare' and temp[2] != 'Misc':
    #         # regionlets[-1].append(np.array([float(temp[6]), float(temp[7]), float(temp[8]), float(temp[9]), 1.0, classes.index(temp[2])]))
    #         if float(temp[17]) > -1:
    #             regionlets[-1].append(np.array([float(temp[6]), float(temp[7]), float(temp[8]), float(temp[9]), 1.0, classes.index(temp[2]), float(temp[3]), float(temp[4]), float(temp[5]), float(temp[10]), float(temp[11]), float(temp[12]), float(temp[13]), float(temp[14]), float(temp[15]), float(temp[16])]))
    #         # regionlets[-1].append(np.array([float(temp[6]), float(temp[7]), float(temp[8]), float(temp[9]), 1.0, classes.index(temp[2]), float(temp[3]), float(temp[4]), float(temp[5]), float(temp[10]), float(temp[11]), float(temp[12]), float(temp[13]), float(temp[14]), float(temp[15]), float(temp[16])]))
    # f.close
    # get detections
    # get detections rrc
    # regionlets = [[]]
    # frame_count = 0
    # classes = ['Cyclist', 'Car', 'Pedestrian', 'Person', 'Van', "Truck", "Tram"]
    # f = open('C:\\Users\\CSY\\SMILEtrack-main\\BoT-SORT\\datasets\\sports\\Car\\0000.txt', 'r')
    # for line in f.readlines():
    #     temp = line.split(',')
    #     while int(temp[0]) != frame_count:
    #         # print(frame_count)
    #         # for i in regionlets[-1]:
    #         #     if i[4] >= 0.6:
    #         #         print([i[0], i[1], i[2], i[3], i[4]])
    #         # i = input()
    #         frame_count += 1
    #         regionlets.append([])
    #     regionlets[-1].append(np.array([float(temp[1]), float(temp[2]), float(temp[3]), float(temp[4]), float(temp[5]), 1]))
    # f.close
    # f = open('C:\\Users\\CSY\\SMILEtrack-main\\BoT-SORT\\datasets\\sports\\Pedestrian\\0000.txt', 'r')
    # for line in f.readlines():
    #     temp = line.split(',')
    #     regionlets[int(temp[0])].append(np.array([float(temp[1]), float(temp[2]), float(temp[3]), float(temp[4]), float(temp[5]), 2]))
    # f.close
    # get detections rrc
    regionlets = np.array(regionlets)
    curr_standard = 0
    for frame_id, img_path in enumerate(files, 1):

        # Detect objects
        outputs, img_info = predictor.inference(img_path, timer)
        scale = min(exp.test_size[0] / float(img_info['height'], ), exp.test_size[1] / float(img_info['width']))

        if outputs[0] is not None:
            outputs = outputs[0].cpu().numpy()
            detections = outputs[:, :7]
            detections[:, :4] /= scale
            
            trackerTimer.tic()
            if frame_id >= (len(camera_parameter) - 1):
                next_frame_id = frame_id-1
            else:
                next_frame_id = frame_id
            if frame_id >= 2:
                prev_frame_id = frame_id-2
            else:
                last_img_info = img_info
                prev_frame_id = frame_id-1
            if skip_check:
                print('error')
                # online_targets, predict_box, curr, skip_check = tracker.update(detections, img_info["raw_img"], last_img_info["raw_img"], camera_parameter[frame_id-1], camera_parameter[prev_frame_id-1], camera_parameter[next_frame_id])
                online_targets, predict_box, curr, skip_check = tracker.update(np.array(regionlets[frame_id-1]), img_info["raw_img"], img_info, camera_parameter[frame_id-1], camera_parameter[prev_frame_id-1], camera_parameter[next_frame_id])
            else:
                # online_targets, predict_box, curr, skip_check = tracker.update(detections, img_info["raw_img"], last_img_info["raw_img"], camera_parameter[frame_id-1], camera_parameter[prev_frame_id], camera_parameter[next_frame_id])
                online_targets, predict_box, curr, skip_check = tracker.update(np.array(regionlets[frame_id-1]), img_info["raw_img"], img_info, camera_parameter[frame_id-1], camera_parameter[prev_frame_id], camera_parameter[next_frame_id])
                last_img_info = img_info
            if skip_check:
                continue
            trackerTimer.toc()
            
            if frame_id == 1:
                curr_standard = len(curr)
            
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)

                    # save results
                    # results.append(
                    #     f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    # )
                    
                    # save results (KITTI)
                    # results.append(
                    #     f"{frame_id-1},{tid},{classes[t.classes]},{t.truncated},{t.occluded},{t.alpha},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[0]+tlwh[2]:.2f},{tlwh[1]+tlwh[3]:.2f},{t.dimensions[0]},{t.dimensions[1]},{t.dimensions[2]},{t.location[0]},{t.location[1]},{t.location[2]},{t.rotation_y},{t.score:.2f}\n"
                    # )
                    
                    # save results (KITTI test)
                    results.append(
                        f"{frame_id-1},{tid},{classes[t.classes]},{1},{1},{1},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[0]+tlwh[2]:.2f},{tlwh[1]+tlwh[3]:.2f},{1},{1},{1},{1},{1},{1},{1},{t.score:.2f}\n"
                    )
            # for t in predict_box:
            #     results.append(
            #         f"{frame_id},{0},{t[0]:.2f},{t[1]:.2f},{t[2]-t[0]:.2f},{t[3]-t[1]:.2f},{0.00:.2f},-1,-1,-1\n"
            #     )
            # results.append(
            #             f"{frame_id},{999},{curr[0]-100:.2f},{curr[0]-100:.2f},{100:.2f},{100:.2f},{0.00:.2f},-1,-1,-1\n"
            #         )
            # curr_count = 0
            # for s in curr[0]:
            #     results.append(
            #             f"{frame_id},{1},{s[0]-100:.2f},{s[1]-100:.2f},{100:.2f},{100:.2f},{0.00:.2f},-1,-1,-1\n"
            #         )
            #     # if curr_count == 10:
            #     #     break
            #     curr_count+=1
            # curr_count = 0
            # for t in curr[1]:
            #     results.append(
            #             f"{frame_id},{4},{t[0]-100:.2f},{t[1]-100:.2f},{100:.2f},{100:.2f},{0.00:.2f},-1,-1,-1\n"
            #         )
            #     # if curr_count == 10:
            #     #     break
            #     curr_count+=1
            # curr_count = 0
            # for i in curr:
            #     if len(curr) < curr_standard*0.4:
            #         results.append(
            #             f"{frame_id},{2},{i[0][0]-5:.2f},{i[0][1]-5:.2f},{5:.2f},{5:.2f},{0.00:.2f},-1,-1,-1\n"
            #         )
            #     else: 
            #         results.append(
            #             f"{frame_id},{1},{i[0][0]-5:.2f},{i[0][1]-5:.2f},{5:.2f},{5:.2f},{0.00:.2f},-1,-1,-1\n"
            #         )
            # curr_standard = len(curr)
            #      curr_count += 1
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        if args.save_frames:
            save_folder = osp.join(vis_folder, args.name)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info('Processing frame {}/{} ({:.2f} fps)'.format(frame_id, num_frames, 1. / max(1e-5, timer.average_time)))

    res_file = osp.join(vis_folder, args.name + ".txt")
    # print(tracker.error_list)
    # print([12, 21, 29, 45, 46, 47, 54, 63, 66, 69, 70, 72, 79, 88, 95, 96, 97, 98])
    with open(res_file, 'w') as f:
        f.writelines(results)
    logger.info(f"save results to {res_file}")


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    vis_folder = osp.join(output_dir, "track_results")
    os.makedirs(vis_folder, exist_ok=True)

    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if args.ckpt is None:
        ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
    else:
        ckpt_file = args.ckpt
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")

    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    predictor = Predictor(model, exp, args.device, args.fp16)

    image_track(predictor, vis_folder, args)


if __name__ == "__main__":
    args = make_parser().parse_args()

    data_path = args.path
    fp16 = args.fp16
    device = args.device

    if args.benchmark == 'MOT20':
        train_seqs = [1, 2, 3, 5]
        test_seqs = [4, 6, 7, 8]
        seqs_ext = ['']
        MOT = 20
    elif args.benchmark == 'MOT17':
        train_seqs = [2, 4, 5, 9, 10, 11, 13]
        test_seqs = [1, 3, 6, 7, 8, 12, 14]
        seqs_ext = ['FRCNN', 'DPM', 'SDP']
        MOT = 17
    elif args.benchmark == 'sports':
        train_seqs = [1]
        test_seqs = [1]
        seqs_ext = ['']
        MOT = 17
    else:
        raise ValueError("Error: Unsupported benchmark:" + args.benchmark)

    ablation = False
    if args.split_to_eval == 'train':
        seqs = train_seqs
    elif args.split_to_eval == 'val':
        seqs = train_seqs
        ablation = True
    elif args.split_to_eval == 'test':
        seqs = test_seqs
    else:
        raise ValueError("Error: Unsupported split to evaluate:" + args.split_to_eval)

    mainTimer = Timer()
    mainTimer.tic()

    for ext in seqs_ext:
        for i in seqs:
            if i < 10:
                seq = 'MOT' + str(MOT) + '-0' + str(i)
            else:
                seq = 'MOT' + str(MOT) + '-' + str(i)

            if ext != '':
                seq += '-' + ext

            args.name = seq

            args.ablation = ablation
            args.mot20 = MOT == 20
            args.fps = 30
            args.device = device
            args.fp16 = fp16
            args.batch_size = 1
            args.trt = False

            split = 'train' if i in train_seqs else 'test'
            args.path = data_path + '/' + split + '/' + seq + '/' + 'img1'
            if args.benchmark == 'sports':
                args.path = 'datasets/sports/test/img1'
            if args.default_parameters:

                if MOT == 20:  # MOT20
                    args.exp_file = r'./yolox/exps/example/mot/yolox_x_mix_mot20_ch.py'
                    args.ckpt = r'./pretrained/bytetrack_x_mot20.tar'
                    args.match_thresh = 0.7
                else:  # MOT17
                    if ablation:
                        args.exp_file = r'./yolox/exps/example/mot/yolox_x_ablation.py'
                        args.ckpt = r'./pretrained/bytetrack_ablation.pth.tar'
                    else:
                        args.exp_file = r'./yolox/exps/example/mot/yolox_x_mix_det.py'
                        args.ckpt = r'./pretrained/bytetrack_x_mot17.pth.tar'

                exp = get_exp(args.exp_file, args.name)

                args.track_high_thresh = 0.6
                args.track_low_thresh = 0.1
                args.track_buffer = 30

                if seq == 'MOT17-05-FRCNN' or seq == 'MOT17-06-FRCNN':
                    args.track_buffer = 14
                elif seq == 'MOT17-13-FRCNN' or seq == 'MOT17-14-FRCNN':
                    args.track_buffer = 25
                else:
                    args.track_buffer = 30

                if seq == 'MOT17-01-FRCNN':
                    args.track_high_thresh = 0.65
                elif seq == 'MOT17-06-FRCNN':
                    args.track_high_thresh = 0.65
                elif seq == 'MOT17-12-FRCNN':
                    args.track_high_thresh = 0.7
                elif seq == 'MOT17-14-FRCNN':
                    args.track_high_thresh = 0.67
                elif seq in ['MOT20-06', 'MOT20-08']:
                    args.track_high_thresh = 0.3
                    exp.test_size = (736, 1920)

                args.new_track_thresh = args.track_high_thresh + 0.1
            else:
                exp = get_exp(args.exp_file, args.name)

            exp.test_conf = max(0.001, args.track_low_thresh - 0.01)
            main(exp, args)

    mainTimer.toc()
    print("TOTAL TIME END-to-END (with loading networks and images): ", mainTimer.total_time)
    print("TOTAL TIME (Detector + Tracker): " + str(timer.total_time) + ", FPS: " + str(1.0 /timer.average_time))
    print("TOTAL TIME (Tracker only): " + str(trackerTimer.total_time) + ", FPS: " + str(1.0 / trackerTimer.average_time))

