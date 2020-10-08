/*
 * author: kohill
 */
#include "mobula_op.h"
#if USING_CUDA
#include <cuda.h>
#else
#include <algorithm>
#include <cmath>
using std::exp;
using std::log;
using std::max;
using std::min;
using std::pow;
#endif  // USING_CUDA

#include "../tensor.hpp"
namespace mobula {

#define UNUSED(expr) do { (void)(expr); } while (0)

template <typename T>
MOBULA_KERNEL retinanet_regression_kernel(
		const int size,
		const int image_h,
		const int image_w,
		const int n_batch,
		const int feature_h,
		const int feature_w,
		const int number_of_classes,
		const T* pointer_reg_preds,
		const T* pointer_cls_preds,
		const T* pointer_bbox_norm_coef,
		const int stride,
		const T* anchors_base_wh,
		const int anchors_base_wh_size,
		T* output) {

	parfor(n_batch*feature_h*feature_w, [&](int index) {
		int batch_idx = index / (feature_h*feature_w);
		index -= batch_idx * (feature_h*feature_w);
		int h_idx = index / feature_w;
		index -= h_idx * feature_w;
		int w_idx = index;

		Tensor4D<const T> tensor_cls_preds = Tensor4D<const T>(pointer_cls_preds, n_batch, anchors_base_wh_size * number_of_classes,  feature_h, feature_w);
		Tensor4D<const T> tensor_reg_preds = Tensor4D<const T>(pointer_reg_preds, n_batch, anchors_base_wh_size * 4, feature_h, feature_w);
		Tensor5D<T> tensor_output = Tensor5D<T>(output, n_batch, anchors_base_wh_size * number_of_classes, feature_h, feature_w, 6);

		T ori_x = w_idx * stride + static_cast<T>(stride) / 2;
		T ori_y = h_idx * stride + static_cast<T>(stride) / 2;

		for(int anchor_idx=0; anchor_idx<anchors_base_wh_size; ++anchor_idx){

			T anchor_w = anchors_base_wh[anchor_idx * 2 + 0];
			T anchor_h = anchors_base_wh[anchor_idx * 2 + 1];

			T anchor_x0 = ori_x - anchor_w / 2;
			T anchor_y0 = ori_y - anchor_h / 2;
			T anchor_x1 = ori_x + anchor_w / 2;
			T anchor_y1 = ori_y + anchor_h / 2;
//			if(anchor_x0 < 0 || anchor_y0 < 0 || anchor_x1 > image_w || anchor_y1 > image_h){
//				continue;
//			}
			if(anchor_x0 >= anchor_x1 || anchor_y0 >= anchor_y1){
				continue;
			}
			T net_pred_0 = tensor_reg_preds(batch_idx, anchor_idx * 4 + 0, h_idx, w_idx) * pointer_bbox_norm_coef[0];
			T net_pred_1 = tensor_reg_preds(batch_idx, anchor_idx * 4 + 1, h_idx, w_idx) * pointer_bbox_norm_coef[1];
			T net_pred_2 = tensor_reg_preds(batch_idx, anchor_idx * 4 + 2, h_idx, w_idx) * pointer_bbox_norm_coef[2];
			T net_pred_3 = tensor_reg_preds(batch_idx, anchor_idx * 4 + 3, h_idx, w_idx) * pointer_bbox_norm_coef[3];

			T pred_x0 = net_pred_0 * (anchor_x1 - anchor_x0 + 1) + anchor_x0;
			T pred_y0 = net_pred_1 * (anchor_y1 - anchor_y0 + 1) + anchor_y0;
			T pred_w = exp(net_pred_2 + log(anchor_x1 - anchor_x0));
			T pred_h = exp(net_pred_3+ log(anchor_y1 - anchor_y0));
			for(int nc=0; nc< number_of_classes; nc++){
				tensor_output(batch_idx, anchor_idx * number_of_classes + nc, h_idx, w_idx, 0) = pred_x0;
				tensor_output(batch_idx, anchor_idx * number_of_classes + nc, h_idx, w_idx, 1)= pred_y0;
				tensor_output(batch_idx, anchor_idx * number_of_classes + nc, h_idx, w_idx, 2)= pred_x0 + pred_w - 1;
				tensor_output(batch_idx, anchor_idx * number_of_classes + nc, h_idx, w_idx, 3) = pred_y0 + pred_h - 1;
				tensor_output(batch_idx, anchor_idx * number_of_classes + nc, h_idx, w_idx, 5) = nc;
				T score = tensor_cls_preds(batch_idx, anchor_idx * number_of_classes + nc, h_idx, w_idx);
				tensor_output(batch_idx, anchor_idx * number_of_classes + nc, h_idx, w_idx, 4) = score;
			}
		}
	});


} // retinanet_regression_kernel


}  // namespace mobula
