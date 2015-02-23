#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ReshapeVolumeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";

  ReshapeVolumeParameter reshape_volume_param = this->layer_param_.reshape_volume_param();
  out_height_ = reshape_volume_param.out_height();
  out_width_ = reshape_volume_param.out_width();
  out_channels_ = reshape_volume_param.out_channels();
  slice_size_ = out_height_ * out_width_;

  CHECK_EQ(slice_size_, bottom[0]->channels()) << this->type() << 
    "Layer expects out_h * out_w == in_c";
  CHECK_EQ(bottom[0]->height(), 1) << this->type() << 
    "Layer expects in_h == 1";
  CHECK_EQ(bottom[0]->width(), 1) << this->type() << 
    "Layer expects in_W == 1";
  top[0]->Reshape(bottom[0]->num(), out_channels_, out_height_, out_width_);
}

template <typename Dtype>
void ReshapeVolumeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int n = 0; n < top[0]->num(); ++n) {
    for (int c = 0; c < out_channels_; ++c) {
      caffe_copy(slice_size_, bottom[0]->cpu_data() + bottom[0]->offset(n,0,0,0), 
        top[0]->mutable_cpu_data() + top[0]->offset(n,c,0,0));
    }
  }
}

template <typename Dtype>
void ReshapeVolumeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  if (out_channels_.size() == 1) {
    caffe_copy(slice_size_, top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());
    return;
  }
  for (int n = 0; n < top[0]->num(); ++n) {
    caffe_add(slice_size_, top[0]->cpu_diff() + top[0]->offset(n,0,0,0),
      top[0]->cpu_diff() + top[0]->offset(n,1,0,0),
      bottom[0]->mutable_cpu_diff() + bottom[0]->offset(n,0,0,0));

    // Add remaining top blob diffs.
    for (int c = 2; c < out_channels_; ++c) {
      const Dtype* top_diff = top[0]->cpu_diff() + top[0]->offset(n,c,0,0);
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff() + bottom[0]->offset(n,0,0,0);
      caffe_axpy(count_, Dtype(1.), top_diff, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReshapeVolumeLayer);
#endif

INSTANTIATE_CLASS(ReshapeVolumeLayer);
REGISTER_LAYER_CLASS(ReshapeVolume);

}  // namespace caffe
