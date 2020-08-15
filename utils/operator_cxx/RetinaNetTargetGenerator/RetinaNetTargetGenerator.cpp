/*
 * author: kohill
 */
#include "mobula_op.h"
#include <memory>
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <cmath>
namespace mobula {

template <typename T>
T box_iou(T x0, T y0, T x1, T y1, T hat_x0, T hat_y0, T hat_x1, T hat_y1){
    T i_w = std::min(hat_x1, x1) - std::max(x0, hat_x0);
    T i_h = std::min(hat_y1, y1) - std::max(y0, hat_y0);
    if(i_w >0 && i_h >0){
        return i_w * i_h / ((x1 - x0) * (y1 - y0) + (hat_x1 - hat_x0) * (hat_y1 - hat_y0) - i_w * i_h);
    }else{
        return 0;
    }
}
#define UNUSED(expr) do { (void)(expr); } while (0)

template <typename T>
MOBULA_FUNC void retinanet_target_gen(const int image_h, const int image_w,
const int feature_h, const int feature_w, const int feature_ch, const int stride,
const T* bboxes, const int number_of_bboxes, const T negative_iou_threshold, const T positive_iou_threshold,
T* anchors_base_wh, const int anchors_base_wh_size, T* output) {
    for(int f_w=0; f_w < feature_w; f_w++){
        for(int f_h=0; f_h < feature_h; f_h ++){
            T ori_x = f_w * stride + static_cast<T>(stride) / 2;
            T ori_y = f_h * stride + static_cast<T>(stride) / 2;
            for(int anchor_idx=0; anchor_idx<anchors_base_wh_size; ++anchor_idx){
                T *output_base = output + f_h * feature_w * anchors_base_wh_size * feature_ch;
                output_base += f_w * anchors_base_wh_size * feature_ch;
                output_base += anchor_idx * feature_ch;

                T anchor_w = anchors_base_wh[anchor_idx * 2 + 0];
                T anchor_h = anchors_base_wh[anchor_idx * 2 + 1];

                T anchor_x0 = ori_x - anchor_w / 2;
                T anchor_y0 = ori_y - anchor_h / 2;
                T anchor_x1 = ori_x + anchor_w / 2;
                T anchor_y1 = ori_y + anchor_h / 2;
                if(anchor_x0 < 0){
                    anchor_x0 = 0;
                    continue;
                }
                if(anchor_y0 < 0){
                    anchor_y0 = 0;
                    continue;
                }
                if(anchor_x1 > image_w){
                    anchor_x1 = image_w;
                    continue;
                }
                if(anchor_y1 > image_h){
                    anchor_y1 = image_h;
                    continue;
                }
                if(anchor_x0 >= anchor_x1 || anchor_y0 >= anchor_y1){
                    continue;
                }
                // If the maximum IoU between this anchor and the gt_boxes is greater than a threshold,
                // then it will be assigned as positive.
                // If the maximum IoU between this anchor and the gt_boxes is less than a threshold,
                // then it will be assigned as negative.
                T max_iou = 0;
                int gt_bbox_idx_with_max_iou = -1;
                for(int gt_bbox_idx=0; gt_bbox_idx < number_of_bboxes; gt_bbox_idx++){
                    T gt_x0 = bboxes[gt_bbox_idx * 5 + 0];
                    T gt_y0 = bboxes[gt_bbox_idx * 5 + 1];
                    T gt_x1 = bboxes[gt_bbox_idx * 5 + 2];
                    T gt_y1 = bboxes[gt_bbox_idx * 5 + 3];
                    T iou = box_iou(anchor_x0, anchor_y0, anchor_x1, anchor_y1, gt_x0, gt_y0, gt_x1, gt_y1);
                    if (iou > max_iou){
                        max_iou = iou;
                        gt_bbox_idx_with_max_iou = gt_bbox_idx;
                    }
                }
                if(gt_bbox_idx_with_max_iou >= 0 && max_iou >positive_iou_threshold){
                    // positive sample
                    T gt_x0 = bboxes[gt_bbox_idx_with_max_iou * 5 + 0];
                    T gt_y0 = bboxes[gt_bbox_idx_with_max_iou * 5 + 1];
                    T gt_x1 = bboxes[gt_bbox_idx_with_max_iou * 5 + 2];
                    T gt_y1 = bboxes[gt_bbox_idx_with_max_iou * 5 + 3];
                    int class_id = static_cast<int>(bboxes[gt_bbox_idx_with_max_iou * 5 + 4]);

                    output_base[0] = 1; // set cls mask to be 1.
                    output_base[1] = 1; // set regression mask to be 1.
                    output_base[2] = (gt_x0 - anchor_x0) / (anchor_x1 - anchor_x0 + 1);
                    output_base[3] = (gt_y0 - anchor_y0) / (anchor_y1 - anchor_y0 + 1);
                    output_base[4] = std::log(gt_x1 - gt_x0) - std::log(anchor_x1 - anchor_x0);
                    output_base[5] = std::log(gt_y1 - gt_y0) - std::log(anchor_y1 - anchor_y0);
                    output_base[6 + class_id - 1] = 1;
                } else if(gt_bbox_idx_with_max_iou >= 0 && max_iou >negative_iou_threshold){
                    // just ignore these sample whose ious are between negative_iou_threshold and positive_iou_threshold.
                    // we assume the default value is zero, so nothing need to do here.
                } else{
                    // negative sample.
                    // mask should be 1, label should be 0;
                    output_base[0] = 1; // set cls mask to be 1.
                    output_base[1] = 0; // set regression mask to be 0.
                }
            }
        }
    }

} // fcos_target_gen_kernel


}  // namespace mobula
