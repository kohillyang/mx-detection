/*
 * author: kohill
 */
#include "bilinear.h"
#include "mobula_op.h"
#include <memory>
#include <cmath>
#include <algorithm>
#include <vector>
#include <tuple>
#include <cstdint>
#include <iostream>
namespace mobula {

#define UNUSED(expr) do { (void)(expr); } while (0)
#include <cstdio>
template <typename T>
class HeatPeak{
    public:
    int x;
    int y;
    T score;
    int peak_id;
    HeatPeak(int x_, int y_, T score_, int peak_id_){
        this->x = x_;
        this->y = y_;
        this->score = score_;
        this->peak_id = peak_id_;
    };
};
template <typename T>
class ConnectionCandidate{
public:
    int nA;
    int nB;
    T score0;
    T score1;
    ConnectionCandidate(int nA_, int nB_, T score0_, T score1_){
        this->nA = nA_;
        this->nB = nB_;
        this->score0 = score0_;
        this->score1 = score1_;
    }
};
template <typename T>
bool f_compare_connection_candidates(const ConnectionCandidate<T> &c0, const ConnectionCandidate<T> &c1){
    return c0.score0 > c1.score0;
}
template <typename T>
class SubSet{
public:
    std::vector<int> parts;
    T score = 0;
    SubSet(size_t number_of_parts){
        this->parts.resize(number_of_parts + 2, -1);
    }
    int& operator [](int idx){
        if(idx >= 0){
            return this->parts[idx];
        }else{
            return this->parts[static_cast<int>(parts.size()) + idx];
        }
    }
    int size(){
        return static_cast<int>(this->parts.size()) - 1;
    }

};


template <typename T, typename T_index>
MOBULA_KERNEL heat_paf_parser_kernel(const T* p_heat, const T* p_paf, const T_index *limbs, const T_index number_of_parts,
                                     const T_index number_of_limbs, const T_index image_width, const T_index image_height,
                                     const T_index max_person_number, T* p_keypoints_out, T* p_scores_out) {
    const T threshold1 = .1;
    const T threshold2 = 0.05;
    const T mid_num = 10;
    // find sub-max positions
    std::vector<std::vector<HeatPeak<T>>> heatPeaks;
    for(int i=0, peak_counter=0; i< number_of_parts; i++){
        heatPeaks.push_back(std::vector<HeatPeak<T>>());
        const size_t channel_offset = i * (image_height * image_width);
        for(int m=0; m < image_height; m ++){
            for(int n=0; n < image_width; n++){
                auto currentValue = p_heat[ channel_offset + m * image_width + n ];
                if(currentValue > threshold1){
                    auto upValue = m==0 ? 0 : p_heat[ channel_offset + (m-1) * image_width + n ];
                    auto bottomValue = m==(image_height-1) ? 0 : p_heat[ channel_offset + (m+1) * image_width + n ];
                    auto rightValue = n==(image_width - 1) ? 0 : p_heat[ channel_offset + m * image_width + (n + 1) ];
                    auto leftValue = n==0? 0 : p_heat[ channel_offset + m * image_width + (n - 1) ];
                    if(currentValue >= upValue && currentValue >= bottomValue
                        && currentValue >= leftValue && currentValue >= rightValue){
                        heatPeaks[i].emplace_back(n, m, currentValue, peak_counter);
                        peak_counter += 1;
                    }
                }

            }
        }
    }
    std::vector<std::vector<std::tuple<int, int, T, int, int>>> connection_all;
    for(int i=0; i< number_of_limbs; i++){
        size_t indexA = limbs[i * 2 + 0];
        size_t indexB = limbs[i * 2 + 1];
        const T *p_score_mid_x = p_paf + (i * 2 + 0) * image_height * image_width;
        const T *p_score_mid_y = p_paf + (i * 2 + 1) * image_height * image_width;
        auto connection_candidates = std::vector<ConnectionCandidate<T>>();
        for(int nA = 0; nA < static_cast<int>(heatPeaks[indexA].size()); nA ++){
            for(int nB=0; nB < static_cast<int>(heatPeaks[indexB].size()); nB ++){
                auto& p0 = heatPeaks[indexA][nA];
                auto& p1 = heatPeaks[indexB][nB];
                T vec_x = p1.x - p0.x;
                T vec_y = p1.y - p0.y;
                T norm = std::sqrt(vec_x * vec_x + vec_y *vec_y);
                if(norm < 0.1){
                    norm += 1;
//                    std::puts("norm is too small, adding one to avoid NAN.");
                }
                vec_x /= norm;
                vec_y /= norm;
                T score_with_dist_prior = 0.0;
                int count_satisfy_thre2 = 0;
                for(T t=0; t< mid_num; t++){
                    int integral_x = static_cast<int>(std::round(p0.x + (p1.x - p0.x) / (mid_num-1) * t));
                    int integral_y = static_cast<int>(std::round(p0.y + (p1.y - p0.y) / (mid_num-1) * t));
//                    std::cerr << integral_x << " " << image_width << std::endl;
//                    std::cerr << integral_y << " " << image_height << std::endl;

                    auto paf_predict_x = p_score_mid_x[integral_y * image_width + integral_x];
                    auto paf_predict_y = p_score_mid_y[integral_y * image_width + integral_x];
                    auto score = vec_x * paf_predict_x + vec_y * paf_predict_y;
                    score_with_dist_prior += score;
                    if(score > threshold2){
                        count_satisfy_thre2 += 1;
                    }
                }
                score_with_dist_prior /= mid_num;
                score_with_dist_prior += std::min(.5 * image_height / norm - 1, 0.0);
                if(count_satisfy_thre2 > 0.8 * mid_num && score_with_dist_prior >0){
                    connection_candidates.emplace_back(nA, nB, score_with_dist_prior, score_with_dist_prior + p0.score + p1.score);
                }
            }
        } // End calculate scores of all possible connections;
        // Remove redundant connections
        // Sort all candidates according to score0
        std::vector<std::tuple<int, int, T, int, int>> connections;
        std::sort(connection_candidates.begin(), connection_candidates.end(), f_compare_connection_candidates<T>);
        for(int nc=0; nc < static_cast<int>(connection_candidates.size()); nc ++){
            auto &nA = connection_candidates[nc].nA;
            auto &nB = connection_candidates[nc].nB;
            bool exist_flag = false;
            for(const auto &c:connections){
                if(nA == std::get<3>(c) || nB == std::get<4>(c)){
                    exist_flag = true;
                    break;
                }
            }
            // if one parts connected to more than one other parts, only keep the one with the largest score.
            if(!exist_flag){
                connections.emplace_back(heatPeaks[indexA][nA].peak_id, heatPeaks[indexB][nB].peak_id, connection_candidates[nc].score0, nA, nB);
            }
            if(connections.size() > std::min(heatPeaks[indexA].size(), heatPeaks[indexB].size())){
                break;
            }
        }
        connection_all.push_back(connections);
    }

    std::vector<HeatPeak<T>> heatPeaks_flatten;
    for(auto &ca:heatPeaks){
        for(auto &c: ca){
        	heatPeaks_flatten.push_back(c);
        }
    }
    // parts connected with each other should be merged.
    std::vector<SubSet<T>> subsets;
    for(int k=0; k< number_of_limbs; k++){
        size_t indexA = limbs[k * 2 + 0];
        size_t indexB = limbs[k * 2 + 1];

        for(size_t i=0; i< connection_all[k].size(); i++){
            int found = 0;
            int subset_idx[2] = {-1, -1};
            for(size_t j=0; j < subsets.size(); j++){
                if(subsets[j][indexA] == std::get<0>(connection_all[k][i]) || subsets[j][indexB] == std::get<1>(connection_all[k][i])){
                    if(found >= 2){
                        puts("[Warning]: found >= 2, This should not happen.");
                        found = 1; // should never reach here.
                    }
                    subset_idx[found] = j;
                    found += 1;
                }
            }
            if(found == 1){
                int j = subset_idx[0];
                int partB = std::get<1>(connection_all[k][i]);
                if(subsets[j][indexB] != partB){
                    subsets[j][indexB] = partB;
                    subsets[j][-1] += 1;
                    subsets[j].score += heatPeaks_flatten[partB].score + std::get<2>(connection_all[k][i]);
                }
            }
            else if(found == 2){
                int j1 = subset_idx[0];
                int j2 = subset_idx[1];
                int membership_count = 0;
                for(int x=0; x < subsets[j1].size(); x++){
                    if(subsets[j1][x] >= 0 and subsets[j2][x] >= 0){
                       membership_count +=1;
                    }
                }
                if(membership_count == 0){
                   // merge them
                   // ignore score
                   for(int x=0; x < subsets[j2].size(); x ++){
                        if(subsets[j2][x] > 0){
                            // here subsets[j1][x] should be -1
                            subsets[j1][x] = subsets[j2][x];
                            subsets[j1][-1] += 1;
                        }
                   }
                   // delete the other one
                   subsets.erase(subsets.begin() + j2);
                }else{
                    // same as found == 1
                    int partB = std::get<1>(connection_all[k][i]);
                    subsets[j1][indexB] = partB;
                    subsets[j1][-1] += 1;
                    subsets[j1].score += heatPeaks_flatten[partB].score + std::get<2>(connection_all[k][i]);
                }
            }else if(found ==0){
                // if find no partA in the subset, create a new subset
            	auto row = SubSet<T>(number_of_parts);
            	int partA = std::get<0>(connection_all[k][i]);
            	int partB = std::get<1>(connection_all[k][i]);
            	row[indexA] = partA;
            	row[indexB] = partB;
            	row[-1] = 2;
                row.score = heatPeaks_flatten[partA].score + heatPeaks_flatten[partB].score + std::get<2>(connection_all[k][i]);
                subsets.push_back(row);
            }
        }
    }
    // copy
    for(int i=0; i< std::min(static_cast<int>(subsets.size()), static_cast<int>(max_person_number)); i++){
        if(subsets[i][-1] >= 3 && subsets[i].score / subsets[i][-1] >= .2){ //   // -1 is the number of available parts.
            for(int j=0; j < number_of_parts; j ++){
                int part = subsets[i][j];
                if(part >= 0){
                    T x = heatPeaks_flatten[part].x;
                    T y = heatPeaks_flatten[part].y;
                    p_keypoints_out[i * number_of_parts * 3 + j * 3 + 0] = x;
                    p_keypoints_out[i * number_of_parts * 3 + j * 3 + 1] = y;
                    p_keypoints_out[i * number_of_parts * 3 + j * 3 + 2] = 1;
                }else{
                    p_keypoints_out[i * number_of_parts * 3 + j * 3 + 2] = 0;
                }
            }
            p_scores_out[i] = subsets[i].score;
        }
    } // copy
} // paf_gen_kernel


}  // namespace mobula
