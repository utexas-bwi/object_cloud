#include "villa_yolocloud/YoloModel.h"
#include <fstream>

using namespace cv;
using namespace std;

void YoloModel::load(string labels_file, string model_config_file, string weights_file) {
#ifdef GPU
    cuda_set_device(0);
#endif

    char mf[model_config_file.size()+1];
    strcpy(mf, model_config_file.c_str());
    char wf[model_config_file.size()+1];
    strcpy(wf, weights_file.c_str());

    net_ = load_network(mf,
                        wf,
                        0);
    set_batch_network(net_, 1);

    // Read in labels
    std::ifstream file(labels_file);
    std::string str;
    while (std::getline(file, str)) {
        labels_.push_back(str);
    }

    int classes = net_->layers[net_->n-1].classes;
    if (classes != labels_.size()) {
        throw runtime_error("Mismatch in classes and labels!");
    }
}

vector<Detection> YoloModel::detect(cv::Mat *image) {
    if (net_ == nullptr) {
        throw runtime_error("Network not loaded.");
    }
    int classes = net_->layers[net_->n-1].classes;

    // Darknet image
    im_[0] = mat_to_image(*image);
    // Padding / Resizing
    im_[1] = letterbox_image(im_[0], net_->w, net_->h);

    float *p = network_predict(net_, im_[1].data);

    // Get detections and do non-maximum suppression
    int nboxes = 0;
    detection *dets = get_network_boxes(net_, image->cols, image->rows, kThresh, kHierThresh, 0, 1, &nboxes);
    do_nms_sort(dets, nboxes, classes, kNMS);

    // Detections to return
    vector<Detection> bboxes;
    for (int i = 0; i < nboxes; i++) {
        int class_id = max_element(dets[i].prob, dets[i].prob+classes) - dets[i].prob;

        if (dets[i].prob[class_id] > kThresh) {
            box b = dets[i].bbox;
            int left  = (b.x-b.w/2.)*image->cols;
            int right = (b.x+b.w/2.)*image->cols;
            int top   = (b.y-b.h/2.)*image->rows;
            int bot   = (b.y+b.h/2.)*image->rows;

            Detection d;
            d.label = labels_[class_id];
            d.bbox = Rect(left, top, right-left, bot-top);
            bboxes.push_back(d);
        }
    }

    // Free all the things
    free_image(im_[0]);
    free_image(im_[1]);
    free_detections(dets, nboxes);

    return bboxes;
}

void YoloModel::unload() {
    if (net_ != nullptr) {
        free_network(net_);
        labels_.clear();
    }
}

YoloModel::~YoloModel() {
    unload();
}
