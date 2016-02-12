#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ImageDataLayer<Dtype>::~ImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const int image_length=this->layer_param_.image_data_param().image_length();
  const int label_length=this->layer_param_.image_data_param().label_length();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  while ( 1 ) {
      bool flag = false;
      vector<string> filenames;
      filenames.clear();
      for ( int img_index = 0; img_index < image_length; ++img_index ) {
          string filename;
          if ( infile>>filename ) {
              filenames.push_back(filename);
              flag = true;
          } else {
              flag = false;
              break;
          }
      }
      vector<float> labels;
      labels.clear();
      for ( int label_index = 0; label_index < label_length; ++label_index ) {
          float label;
          if ( infile>>label ) {
              labels.push_back(label);
              flag = true;
          } else {
              flag = false;
              break;
          }
      }
      if ( !flag ) 
          break;
      lines_.push_back(make_pair(filenames, labels));
  }
  infile.close();
//  string filename;
//  int label;
//  while (infile >> filename >> label) {
//    lines_.push_back(std::make_pair(filename, label));
//  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first[0],
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first[0];
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  top_shape[1] *= image_length;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  //vector<int> label_shape(1, batch_size);
  //vector<int> label_shape(batch_size,label_length);
  //top[1]->Reshape(label_shape);
  top[1]->Reshape(batch_size, label_length, 1, 1);

  LOG(INFO)<<top[1]->num()<<","<<top[1]->channels()<<","<<top[1]->width()<<","<<top[1]->height();
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    //this->prefetch_[i].label_.Reshape(label_shape);
    this->prefetch_[i].label_.Reshape(batch_size, label_length, 1, 1);
  }
}

template <typename Dtype>
void ImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void ImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  const int image_length = image_data_param.image_length();
  const int label_length = image_data_param.label_length();
  string root_folder = image_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first[0],
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first[0];
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  top_shape[1]*= image_length;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
      int offset_h = -1, offset_w = -1;
      int img_offset;
      if ( this->layer_param_.transform_param().crop_size() )
          img_offset = cv_img.channels() * this->layer_param_.transform_param().crop_size() * this->layer_param_.transform_param().crop_size();
      else
          img_offset = cv_img.channels() * cv_img.rows * cv_img.cols;
      for ( int img_id = 0; img_id < image_length; ++img_id ) {
        // get a blob
        timer.Start();
        CHECK_GT(lines_size, lines_id_);
        cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first[img_id],
            new_height, new_width, is_color);
        CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first[img_id];
        read_time += timer.MicroSeconds();
        timer.Start();
        // Apply transformations (mirror, crop...) to the image
        int offset = batch->data_.offset(item_id);
        //this->transformed_data_.set_cpu_data(prefetch_data + offset + img_offset * img_id);
        this->transformed_data_.set_cpu_data(prefetch_data + offset + img_offset * 0);
        if ( item_id == 0 )
            this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
        else
            this->data_transformer_->Transform(cv_img, &(this->transformed_data_), true);
        trans_time += timer.MicroSeconds();
      }

      for ( int label_id = 0; label_id < label_length; ++label_id ) {
          prefetch_label[item_id*label_length + label_id] = lines_[lines_id_].second[label_id];
      }
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageDataLayer);
REGISTER_LAYER_CLASS(ImageData);

}  // namespace caffe
#endif  // USE_OPENCV
