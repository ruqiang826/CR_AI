#label img in data_raw 

# clean tmp_data and data_out
# copy the new labeled image to tmp_data

# update the CLASS defination in script/auto_label.py

# copy the lastest config and model
cp ../py-faster-rcnn.ruqiang826/models/pascal_voc/VGG_CNN_M_1024/faster_rcnn_end2end/test.prototxt ./model/
cp ../py-faster-rcnn.ruqiang826/output/faster_rcnn_end2end/voc_2007_trainval/vgg_cnn_m_1024_faster_rcnn_iter_13200.caffemodel ./model/


# run auto label, get labeled file in data_out
python script/auto_label.py --model model/vgg_cnn_m_1024_faster_rcnn_iter_13200.caffemodel --input tmp_data --output data_out/
