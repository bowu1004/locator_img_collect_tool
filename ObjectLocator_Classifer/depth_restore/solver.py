import torch
import depth_restore.network as network
import numpy as np
import cv2
from PIL import Image
from os.path import exists, join
from os import makedirs
import torchvision.transforms as transforms


class Solver(object):
    def __init__(self):
        self.net = network.Net().cuda()
        self.transform = transforms.Compose([
            # transforms.Resize([224, 224]),  # 如果需要调整大小，可以取消注释此行
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def init_weights(self, ckpt_path):
        self.net.eval()
        ckpt = torch.load(ckpt_path)
        state_dict = ckpt['state_dict']
        temp = {}
        for key, value in state_dict.items():
            if str(key).startswith('get_pred_d') == True:
                continue
            elif str(key).startswith('get_pdepths.get_pdepth_6.') == True:
                continue
            elif str(key).startswith('pool_k') == True:
                key = 'cross_modal_feature_fusion.high_level_fusion.FAM_1' + key[len('pool_k'):]
            elif str(key).startswith('pool_r') == True:
                key = 'cross_modal_feature_fusion.high_level_fusion.FAM_2' + key[len('pool_r'):]
            elif str(key).startswith('fc.att') == True:
                key = 'get_fused_depth_features.get_fusion_weights.fc' + key[len('fc.att'):]
            elif str(key).startswith('fc.fc') == True:
                key = 'get_fused_depth_features.get_fusion_weights.fc' + key[len('fc.fc'):]
            elif str(key).startswith('get_cmprs_r.') == True:
                key = 'cross_modal_feature_fusion.compr_I.' + key[len('get_cmprs_r.'):]
            elif str(key).startswith('get_cmprs_d.') == True:
                key = 'get_fused_depth_features.compr_Do.' + key[len('get_cmprs_d.'):]
            elif str(key).startswith('get_cmprs_dp.') == True:
                key = 'get_fused_depth_features.compr_De.' + key[len('get_cmprs_dp.'):]
            elif str(key).startswith('get_pdepths.get_pd_6.') == True:
                key = 'estimate_depth.get_feat.' + key[len('get_pdepths.get_pd_6.'):]
            elif str(key).startswith('get_pdepths.get_cmprs_pd.') == True:
                key = 'estimate_depth.compr.' + key[len('get_pdepths.get_cmprs_pd.'):]
            elif str(key).startswith('get_pdepths.up_pd.') == True:
                key = 'estimate_depth.u_decoder.' + key[len('get_pdepths.up_pd.'):]
            elif str(key).startswith('get_att.conv') == True:
                key = 'cross_modal_feature_fusion.high_level_fusion.transform.' + key[len('get_att.conv.'):]
            elif str(key).startswith('up_kb.') == True:
                key = 'cross_modal_feature_fusion.low_level_fusion.u_decoder_boundary.' + key[len('up_kb.'):]
            elif str(key).find('up.') != -1:
                if str(key).find('bdry_conv.') != -1:
                    key = str(key).replace('up.', 'cross_modal_feature_fusion.low_level_fusion.boundary_enhance.')
                    key = str(key).replace('bdry_conv.', 'conv.')
                else:
                    key = str(key).replace('up.', 'cross_modal_feature_fusion.low_level_fusion.u_decoder_saliency.')
            temp[key] = value
        state_dict = temp
        self.net.load_state_dict(state_dict)

    def test(self, img):

        with torch.no_grad():
            # torch.save({'state_dict': self.net.state_dict()}, './new.pth')
            image_pil = Image.fromarray(img)

            transformed_image = torch.unsqueeze(self.transform(image_pil), 0)

            predictions = self.net(transformed_image.cuda())

            # start_time = time.time()
            batch_preds = np.squeeze(np.squeeze(np.array((predictions.permute(0, 2, 3, 1) * 255).cpu()), 3), 0)
            # batch_preds = np.squeeze(np.squeeze(np.array(predictions.permute(0, 2, 3, 1).cpu() * 255), 3), 0)
            # end_time = time.time()
            # print(f"预处理推理时长为{end_time - start_time}s")
        return batch_preds.astype(np.uint8)