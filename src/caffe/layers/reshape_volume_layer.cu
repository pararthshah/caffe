#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ReshapeVolumeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int n = 0; n < top[0]->num(); ++n) {
    for (int c = 0; c < out_channels_; ++c) {
      caffe_copy(slice_size_, bottom[0]->gpu_data() + bottom[0]->offset(n,0,0,0), 
        top[0]->mutable_gpu_data() + top[0]->offset(n,c,0,0));
    }
  }
}

template <typename Dtype>
void ReshapeVolumeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  if (out_channels_ == 1) {
    caffe_copy(slice_size_, top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
    return;
  }
  for (int n = 0; n < top[0]->num(); ++n) {
    caffe_gpu_add(slice_size_, top[0]->gpu_diff() + top[0]->offset(n,0,0,0),
      top[0]->gpu_diff() + top[0]->offset(n,1,0,0),
      bottom[0]->mutable_gpu_diff() + bottom[0]->offset(n,0,0,0));

    // Add remaining top blob diffs.
    for (int c = 2; c < out_channels_; ++c) {
      const Dtype* top_diff = top[0]->gpu_diff() + top[0]->offset(n,c,0,0);
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff() + bottom[0]->offset(n,0,0,0);
      caffe_axpy(slice_size_, Dtype(1.), top_diff, bottom_diff);
    }
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(ReshapeVolumeLayer);

}  // namespace caffe
