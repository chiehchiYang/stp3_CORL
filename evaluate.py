from argparse import ArgumentParser
from PIL import Image
import torch
import torch.utils.data
import numpy as np
import torchvision
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
import pathlib
import datetime

from stp3.trainer import TrainingModule
from stp3.metrics import IntersectionOverUnion, PanopticMetric, PlanningMetric, softIoU
from stp3.utils.network import preprocess_batch, NormalizeInverse
from stp3.utils.instance import predict_instance_segmentation_and_trajectories
from stp3.utils.visualisation import make_contour, heatmap_image, flow_to_image
from stp3.datas.CarlaData import CarlaDataset




class Val_goal_points_metric:
    
    def __init__(self) -> None:
        self.True_counter = 0
        self.False_counter = 0
        self.Total_counter = 0
        
        
    def val_points(self, pred, gt, n_present):
        pred =  pred[:, n_present - 1:][0][0][0].cpu().numpy()
        gt = gt[:, n_present - 1:][0][0].cpu().numpy()
        
        index = np.argwhere(gt != 0)
        
        for x, y in index:    
            if pred[x][y] > 0.5:
                
                self.True_counter+=1
            else:
                self.False_counter+=1
            self.Total_counter+=1

    def show_result(self):
        print("TP: ", self.True_counter)
        print("FN: ", self.False_counter)
        print("Total: ", self.Total_counter)
        recall = self.True_counter/self.Total_counter
        print("Recall: ",recall)
        
        
    ### 143915/ 155275
        
    
    

def mk_save_dir():
    now = datetime.datetime.now()
    string = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
    save_path = pathlib.Path('imgs') / string
    save_path.mkdir(parents=True, exist_ok=False)
    return save_path

def eval(checkpoint_path, dataroot):
    
    # train
    # test
    # val 
    
    
    val_goal_points = Val_goal_points_metric()
    
    save_path = mk_save_dir()

    trainer = TrainingModule.load_from_checkpoint(checkpoint_path, strict=False)#True)#False)# True)
    print(f'Loaded weights from \n {checkpoint_path}')
    trainer.eval()

    device = torch.device('cuda:0')
    trainer.to(device)
    model = trainer.model
    
    cfg = model.cfg
    cfg.GPUS = "[0]"
    cfg.BATCHSIZE = 1
    cfg.LIFT.GT_DEPTH = False
    cfg.DATASET.DATAROOT = dataroot
    cfg.DATASET.MAP_FOLDER = dataroot
    dataroot = cfg.DATASET.DATAROOT
    nworkers = cfg.N_WORKERS
    
    # load Carla dataset 
    valdata = CarlaDataset(dataroot, False, cfg) 
    # val set
    
    # valdata = CarlaDataset(dataroot, True, cfg) 
    # train set ( Town10HD )
    
    valloader = torch.utils.data.DataLoader(
            valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=nworkers, pin_memory=True, drop_last=False)
    
    n_classes = len(cfg.SEMANTIC_SEG.VEHICLE.WEIGHTS) # = 2 
    
    hdmap_class = cfg.SEMANTIC_SEG.HDMAP.ELEMENTS
    metric_vehicle_val = IntersectionOverUnion(n_classes).to(device)
    future_second = int(cfg.N_FUTURE_FRAMES / 2)
    
    if cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
        metric_pedestrian_val = IntersectionOverUnion(n_classes).to(device)

    if cfg.SEMANTIC_SEG.HDMAP.ENABLED:
        metric_hdmap_val = []
        for i in range(len(hdmap_class)):
            metric_hdmap_val.append(IntersectionOverUnion(2, absent_score=1).to(device))

    # if cfg.INSTANCE_SEG.ENABLED:
    #     metric_panoptic_val = PanopticMetric(n_classes=n_classes).to(device)
        
        
    metric_centerness_val = softIoU(2).to(device)

    if cfg.PLANNING.ENABLED:
        metric_planning_val = []
        for i in range(future_second):
            metric_planning_val.append(PlanningMetric(cfg, 2*(i+1)).to(device))
            
    # write video 
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(f'output.mp4', fourcc, 20.0, (1196,  279))
    
    #     plt.savefig(save_path / ('%04d.png' % frame)) 
    
    for index, batch in enumerate(tqdm(valloader)):
        preprocess_batch(batch, device)
        image = batch['image']
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        future_egomotion = batch['future_egomotion']
        command = batch['command']
        trajs = batch['sample_trajectory']
        target_points = batch['target_point']
        B = len(image)
        labels = trainer.prepare_future_labels(batch)
        
        with torch.no_grad():
            output = model(
                image, intrinsics, extrinsics, future_egomotion
            )

        n_present = model.receptive_field
                
        output = output[1]

        # semantic segmentation metric
        seg_prediction = output['segmentation'].detach()
        seg_prediction = torch.argmax(seg_prediction, dim=2, keepdim=True)
        metric_vehicle_val(seg_prediction[:, n_present - 1:], labels['segmentation'][:, n_present - 1:])

        if cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
            pedestrian_prediction = output['pedestrian'].detach()
            pedestrian_prediction = torch.argmax(pedestrian_prediction, dim=2, keepdim=True)
            metric_pedestrian_val(pedestrian_prediction[:, n_present - 1:],
                                       labels['pedestrian'][:, n_present - 1:])
        else:
            pedestrian_prediction = torch.zeros_like(seg_prediction)

        if cfg.SEMANTIC_SEG.HDMAP.ENABLED:
            for i in range(len(hdmap_class)):
                hdmap_prediction = output['hdmap'][:, 2 * i:2 * (i + 1)].detach()
                hdmap_prediction = torch.argmax(hdmap_prediction, dim=1, keepdim=True)
                metric_hdmap_val[i](hdmap_prediction, labels['hdmap'][:, i:i + 1])

        # if cfg.INSTANCE_SEG.ENABLED:
        #     pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
        #         output, compute_matched_centers=False, make_consistent=True
        #     )
        #     metric_panoptic_val(pred_consistent_instance_seg[:, n_present - 1:],
        #                              labels['instance'][:, n_present - 1:])
        
        # seg_prediction = output['heatmap'].detach()
        # metric_heatmap_val(seg_prediction[:,:], labels['heatmap'][:, :])
            

        
        pred_centerness = output['instance_center'].detach()
        
        

        
        val_goal_points.val_points(pred_centerness, batch['goal_points'], n_present)
        
        
        metric_centerness_val(pred_centerness[:, n_present - 1:], labels['centerness'][:, n_present - 1:])

        if cfg.PLANNING.ENABLED:
            occupancy = torch.logical_or(seg_prediction, pedestrian_prediction)
            _, final_traj = model.planning(
                cam_front=output['cam_front'].detach(),
                trajs=trajs[:, :, 1:],
                gt_trajs=labels['gt_trajectory'][:, 1:],
                cost_volume=output['costvolume'][:, n_present:].detach(),
                semantic_pred=occupancy[:, n_present:].squeeze(2),
                hd_map=output['hdmap'].detach(),
                commands=command,
                target_points=target_points
            )
            occupancy = torch.logical_or(labels['segmentation'][:, n_present:].squeeze(2),
                                         labels['pedestrian'][:, n_present:].squeeze(2))
            for i in range(future_second):
                cur_time = (i+1)*2
                metric_planning_val[i](final_traj[:,:cur_time].detach(), labels['gt_trajectory'][:,1:cur_time+1], occupancy[:,:cur_time])

        #if index % 100 == 0:
        
        
        # save(output, labels, batch, n_present, index, save_path)
    

    val_goal_points.show_result()
    print("----")
    
    
    results = {}
    
    scores, tp, fp, fn = metric_centerness_val.compute()
    print(scores, tp, fp, fn)
    results['centerness_iou'] = scores[1]
    print("tp: ", tp)
    print("fp: ", fp)
    print("fn: ", fn)
    print("recall : ", tp/(tp+fn))
    print("precision :", tp/(tp+fp))

    scores = metric_vehicle_val.compute()
    results['vehicle_iou'] = scores[1]

    if cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
        scores = metric_pedestrian_val.compute()
        results['pedestrian_iou'] = scores[1]

    if cfg.SEMANTIC_SEG.HDMAP.ENABLED:
        for i, name in enumerate(hdmap_class):
            scores = metric_hdmap_val[i].compute()
            results[name + '_iou'] = scores[1]

    # if cfg.INSTANCE_SEG.ENABLED:
    if False:
        scores = metric_panoptic_val.compute()
        for key, value in scores.items():
            results['vehicle_'+key] = value[1]

    #if cfg.PLANNING.ENABLED:
    if False:
        for i in range(future_second):
            scores = metric_planning_val[i].compute()
            for key, value in scores.items():
                results['plan_'+key+'_{}s'.format(i+1)]=value.mean()

    for key, value in results.items():
        print(f'{key} : {value.item()}')
        
        
        
    

def save(output, labels, batch, n_present, frame, save_path):
    
    # n_present = 3 

    
    b = 0
    t = n_present - 1
    

    

    
    hdmap = labels['hdmap'][0].detach().cpu().numpy()
    images = batch['image']

    denormalise_img = torchvision.transforms.Compose(
        (NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         torchvision.transforms.ToPILImage(),)
    )

    val_w = 2.99
    val_h = 2.99 * (224. / 480.)
    plt.figure(1, figsize=(4*val_w,2*val_h))
    width_ratios = (val_w,val_w,val_w,val_w, val_w)
        
    gs = matplotlib.gridspec.GridSpec(2, 5, width_ratios=width_ratios)
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)



    #  front, left, right and rear rgb image
    plt.subplot(gs[0, 2])
    plt.annotate('FRONT LEFT', (0.01, 0.87), color='white', xycoords='axes fraction', fontsize=14)
    plt.imshow(denormalise_img(images[0,n_present-1,2].cpu()))
    plt.axis('off')

    plt.subplot(gs[0, 1])
    plt.annotate('FRONT', (0.01, 0.87), color='white', xycoords='axes fraction', fontsize=14)
    plt.imshow(denormalise_img(images[0,n_present-1,0].cpu()))
    plt.axis('off')

    plt.subplot(gs[0, 0])
    plt.annotate('FRONT RIGHT', (0.01, 0.87), color='white', xycoords='axes fraction', fontsize=14)
    plt.imshow(denormalise_img(images[0,n_present-1,1].cpu()))
    plt.axis('off')

    plt.subplot(gs[1, 1])
    plt.annotate('BACK', (0.01, 0.87), color='white', xycoords='axes fraction', fontsize=14)
    showing = denormalise_img(images[0, n_present - 1, 3].cpu())
    showing = showing.transpose(Image.FLIP_LEFT_RIGHT)
    plt.imshow(showing)
    plt.axis('off')
    plt.subplot(gs[:, 3])
    
    
       
    centerness = labels['centerness'][b, t, 0].cpu().numpy()
    center_plot = heatmap_image(centerness)
    showing = make_contour(center_plot)    
    
    #showing = torch.zeros((200, 200, 3)).numpy()
    #showing[:, :] = np.array([219 / 255, 215 / 255, 215 / 255])

    # drivable area 
    area = hdmap[1]
    hdmap_index = ~(area > 0)
    showing[hdmap_index] = np.array([161 / 255, 158 / 255, 158 / 255])

    # lane
    area = hdmap[0]
    hdmap_index = area > 0
    showing[hdmap_index] = np.array([84 / 255, 70 / 255, 70 / 255])
    
    # vehicle 
    semantic_seg = labels['segmentation'][:, n_present - 1][0][0].cpu().numpy()
    semantic_index = semantic_seg > 0
    showing[semantic_index] = np.array([255 / 255, 128 / 255, 0 / 255])
    
    # ground truth goal area  
    

    # pedestrian_seg = labels['pedestrian'][:, n_present - 1][0][0].cpu().numpy()
    # pedestrian_index = pedestrian_seg > 0
    # showing[pedestrian_index] = np.array([28 / 255, 81 / 255, 227 / 255])
    plt.imshow(showing)
    plt.axis('off')
    
    bx = np.array([-50.0 + 0.5/2.0, -50.0 + 0.5/2.0])
    dx = np.array([0.5, 0.5])
    w, h = 1.85, 4.084
    pts = np.array([
        [-h / 2. + 0.5, w / 2.],
        [h / 2. + 0.5, w / 2.],
        [h / 2. + 0.5, -w / 2.],
        [-h / 2. + 0.5, -w / 2.],
    ])
    pts = (pts - bx) / dx
    pts[:, [0, 1]] = pts[:, [1, 0]]
    plt.fill(pts[:, 0], pts[:, 1], '#76b900')

    plt.xlim((200, 0))
    plt.ylim((0, 200))

    #########################################################
    plt.subplot(gs[:, 4])
    
    # showing = torch.zeros((200, 200, 3)).numpy()
    # showing[:, :] = np.array([219 / 255, 215 / 255, 215 / 255])
    
    
    
           
    centerness = output['instance_center'][b, t, 0].cpu().numpy()
    center_plot = heatmap_image(centerness)
    showing = make_contour(center_plot)    
    
    
    
    

    # drivable area 
    area = hdmap[1]
    hdmap_index = ~(area > 0)
    showing[hdmap_index] = np.array([161 / 255, 158 / 255, 158 / 255])

    # lane
    area = hdmap[0]
    hdmap_index = area > 0
    showing[hdmap_index] = np.array([84 / 255, 70 / 255, 70 / 255])
    
    # vehicle 
    semantic_seg = labels['segmentation'][:, n_present - 1][0][0].cpu().numpy()
    semantic_index = semantic_seg > 0
    showing[semantic_index] = np.array([255 / 255, 128 / 255, 0 / 255])

        
    
    # prediction heatmap 
    
    # pedestrian_seg = output['pedestrian'].argmax(dim=2).detach().cpu().numpy()
    # pedestrian_seg = pedestrian_seg[0, 0]# [::-1, ::-1]
    
    # # print(pedestrian_seg.shape)
    # index = np.nonzero(pedestrian_seg)
    # # print(index)
    # showing[index] = np.array([28 / 255, 81 / 255, 227 / 255])
    
    
    # heatmap = output['heatmap'].detach()# .cpu().numpy()[0][0]

    # heatmap = heatmap.cpu().numpy()[0][0]
    # heatmap_prediction_plot, index = draw_heatmap(heatmap)
    # showing[index] = heatmap_prediction_plot[index]
    
    # goal area prediction 
    # pedestrian_seg = torch.argmax(pedestrian[0], dim=0).cpu().numpy()
    # pedestrian_index = pedestrian_seg > 0
    # showing[pedestrian_index] = np.array([28 / 255, 81 / 255, 227 / 255])

    plt.imshow(showing)
    plt.axis('off')

    bx = np.array([-50.0 + 0.5/2.0, -50.0 + 0.5/2.0])
    dx = np.array([0.5, 0.5])
    w, h = 1.85, 4.084
    pts = np.array([
        [-h / 2. + 0.5, w / 2.],
        [h / 2. + 0.5, w / 2.],
        [h / 2. + 0.5, -w / 2.],
        [-h / 2. + 0.5, -w / 2.],
    ])
    pts = (pts - bx) / dx
    pts[:, [0, 1]] = pts[:, [1, 0]]
    plt.fill(pts[:, 0], pts[:, 1], '#76b900')

    plt.xlim((200, 0))
    plt.ylim((0, 200))
    
    plt.savefig(save_path / ('%04d.png' % frame))
    plt.close()
    

    
    
    
    # # test threshold 
    # centerness = labels['centerness'][b, t, 0].cpu().numpy()    
    # mask = centerness < 0.2
    # centerness[mask] = 0.0
    # center_plot = heatmap_image(centerness)
    # plt.imshow(make_contour(center_plot)  )  
    # plt.axis('off')
    # plt.savefig('test.png')
    # plt.close()
    
    

if __name__ == '__main__':
    parser = ArgumentParser(description='STP3 evaluation')
    parser.add_argument('--checkpoint', default='last.ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', default=None, type=str)

    args = parser.parse_args()

    eval(args.checkpoint, args.dataroot)
