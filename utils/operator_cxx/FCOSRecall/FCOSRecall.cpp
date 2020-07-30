/*
 * author: kohill
 */
#include "mobula_op.h"
#include <algorithm>

namespace mobula {

#define UNUSED(expr) do { (void)(expr); } while (0)

template <typename T>
MOBULA_FUNC void fcos_recall(const T *prediction, int feature_n, int feature_h, int feature_w,  int feature_ch,
    int stride, int number_of_bboxes, T* gt_bboxes, T* output) {
    UNUSED(prediction);
    UNUSED(feature_h);
    UNUSED(feature_w);
    UNUSED(feature_ch);
    UNUSED(stride);
    UNUSED(output);
    const int ch_l=0, ch_t=1, ch_r=2, ch_b=3;
    int ch_center_ness = 4;
    // channel 5 is the probability of the background class.
    int ch_cls_start = 6;
    int ch_cls_end = feature_ch;

    int n_bbox = 0;

    
    for(int f_w=0; f_w < feature_w; f_w++){
        for(int f_h=0; f_h < feature_h; f_h ++){
            T ori_x = f_w * stride + static_cast<T>(stride) / 2;
            T ori_y = f_h * stride + static_cast<T>(stride) / 2; 
            T delta_l = *(prediction + ch_l * feature_h * feature_w + f_h * feature_w + f_w);
            T delta_t = *(prediction + ch_t * feature_h * feature_w + f_h * feature_w + f_w);
            T delta_r = *(prediction + ch_r * feature_h * feature_w + f_h * feature_w + f_w);
            T delta_b = *(prediction + ch_b * feature_h * feature_w + f_h * feature_w + f_w);
            T pred_x0 = ori_x - delta_l;
            T pred_y0 = ori_y - delta_t;
            T pred_x1 = ori_x + delta_r;
            T pred_y1 = ori_y + delta_b;
            for(int nb=0; nb < number_of_bboxes; nb++){
                T x0 = gt_bboxes[nb * 5 + 0];
                T y0 = gt_bboxes[nb * 5 + 1];
                T x1 = gt_bboxes[nb * 5 + 2];
                T y1 = gt_bboxes[nb * 5 + 3];
                // IoU
                T iou_delta_x = sdt::min(pred_x1, pred_x1) - std::max(x0, pred_x0);
                T iou_delta_y = sdt::min(pred_y1, pred_y1) - std::max(y0, pred_y0);
                T area_intersect = 0;
                if(iou_delta_x >= 0 && iou_delta_y >= 0){
                    area_intersect = iou_delta_x * iou_delta_y;
                }
                T area_union = (x1 - x0) * (y1 - y0) + (pred_x1 - pred_x0) * (pred_y1 - pred_y0) - area_intersect;
                T IoU = area_intersect / area_union;
                if
            }

        }
    }

} // fcos_target_gen_kernel


}  // namespace mobula
