#label img in data_raw 

# clean tmp_data and data_out
# copy the new labeled image to tmp_data

# copy config
cp ../py-faster-rcnn.ruqiang826/models/pascal_voc/VGG_CNN_M_1024/faster_rcnn_end2end/test.prototxt ./model/

# run auto label, get labeled file in data_out
python script/auto_label.py --model model/vgg_cnn_m_1024_faster_rcnn_iter_9980.caffemodel --input tmp_data --output data_out/
