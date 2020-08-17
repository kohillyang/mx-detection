/*
 * author: kohill
 */
#include "bilinear.h"
#include "mobula_op.h"
#include <memory>
#include <cmath>
#include <algorithm>
namespace mobula {

#define UNUSED(expr) do { (void)(expr); } while (0)
template<typename Dtype>
#include <cstdio>
void putVecMaps(Dtype* entryX, Dtype* entryY, Dtype* count, float centerA_x0, float centerA_y0,
                float centerB_x1, float centerB_y1, int stride, int grid_x, int grid_y, int thre){
    centerA_x0 = centerA_x0 / stride;
    centerA_y0 = centerA_y0 / stride;
    centerB_x1 = centerB_x1 / stride;
    centerB_y1 = centerB_y1 / stride;

    float bc_x = centerB_x1 - centerA_x0;
    float bc_y = centerB_y1 - centerA_y0;

    int min_x = std::max( int(round(std::min(centerA_x0, centerB_x1)-thre)), 0);
    int max_x = std::min( int(round(std::max(centerA_x0, centerB_x1)+thre)), grid_x);

    int min_y = std::max( int(round(std::min(centerA_y0, centerB_y1)-thre)), 0);
    int max_y = std::min( int(round(std::max(centerA_y0, centerB_y1)+thre)), grid_y);

    float norm_bc = std::sqrt(bc_x*bc_x + bc_y*bc_y);
    if(norm_bc < 1e-3){
        std::printf("norm bc is too small, plussing 1 to avoid NAN");
        norm_bc += 1.0f;
    }
    bc_x = bc_x / norm_bc;
    bc_y = bc_y / norm_bc;

    // float x_p = (centerA.x + centerB.x) / 2;
    // float y_p = (centerA.y + centerB.y) / 2;
    // float angle = atan2f(centerB.y - centerA.y, centerB.x - centerA.x);
    // float sine = sinf(angle);
    // float cosine = cosf(angle);
    // float a_sqrt = (centerA.x - x_p) * (centerA.x - x_p) + (centerA.y - y_p) * (centerA.y - y_p);
    // float b_sqrt = 10; //fixed

    for (int g_y = min_y; g_y < max_y; g_y++){
        for (int g_x = min_x; g_x < max_x; g_x++){
            float ba_x = g_x - centerA_x0;
            float ba_y = g_y - centerA_y0;
            float dist = std::abs(ba_x*bc_y -ba_y*bc_x);

            // float A = cosine * (g_x - x_p) + sine * (g_y - y_p);
            // float B = sine * (g_x - x_p) - cosine * (g_y - y_p);
            // float judge = A * A / a_sqrt + B * B / b_sqrt;

            if(dist <= thre){
            //if(judge <= 1){
            int cnt = count[g_y*grid_x + g_x];
            //LOG(INFO) << "putVecMaps here we start for " << g_x << " " << g_y;
            if (cnt == 0){
                entryX[g_y*grid_x + g_x] = bc_x;
                entryY[g_y*grid_x + g_x] = bc_y;
            }
            else{
                entryX[g_y*grid_x + g_x] = (entryX[g_y*grid_x + g_x]*cnt + bc_x) / (cnt + 1);
                entryY[g_y*grid_x + g_x] = (entryY[g_y*grid_x + g_x]*cnt + bc_y) / (cnt + 1);
                count[g_y*grid_x + g_x] = cnt + 1;
            }
          }

        }
    }
}

template <typename T, typename T_index>
MOBULA_KERNEL paf_gen_kernel(const T_index* limb_sequence, const T* keypoints, const int h, const int w, const int stride,
                              const int nperson, const int nparts, const int nlimb, int threshold, T* output) {
    const int grid_x = w / stride;
    const int grid_y = h / stride;
    const int channelOffset = grid_y * grid_x;
    for(int nl=0; nl < nlimb; nl ++){
        T* count = new_array<T>(sizeof(T) * grid_y * grid_x);
        memset(count, 0, grid_x * grid_y * sizeof(T));
        for(int i=0; i < nperson; i++){
            int index0 = limb_sequence[nl * 2 + 0];
            int index1 = limb_sequence[nl * 2 + 1];
            float x0 = keypoints[i * nparts * 3 + index0 * 3 + 0];
            float y0 = keypoints[i * nparts * 3 + index0 * 3 + 1];
            int available_0 = keypoints[i * nparts * 3 + index0 * 3 + 2];

            float x1 = keypoints[i * nparts * 3 + index1 * 3 + 0];
            float y1 = keypoints[i * nparts * 3 + index1 * 3 + 1];
            int available_1 = keypoints[i * nparts * 3 + index1 * 3 + 2];

            if(available_0 && available_1 && x0 >= 0 && y0 >=0 && x1 >=0 && y1 >=0){
                putVecMaps(output + (nl *2 + 0) * channelOffset, output + (nl *2 + 1) * channelOffset,
                count, x0, y0, x1, y1, stride, grid_x, grid_y, threshold);
            }
        }
        del_array<T>(count);
    }
} // paf_gen_kernel


}  // namespace mobula
