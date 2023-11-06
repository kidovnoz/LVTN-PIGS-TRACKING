import datetime
import os
import time
from hashlib import scrypt
from re import L, sub
from turtle import circle, clear
from typing import Counter, List
import pyglet
from deep_sort import detection
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
from PIL import Image
from tensorflow._api.v2.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.python.saved_model import tag_constants
import core.utils as utils
from collision_check import CollisionCheck
from core.config import cfg
from core.yolov4 import YOLO, filter_boxes
# deep sort imports
from deep_sort import nn_matching, preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 512, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/main.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.5, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
flags.DEFINE_boolean('drink', False, 'count drink object')


def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = 100
    nms_max_overlap = 1
    centroid_dict = defaultdict(list)
   
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=13)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video
    
    #toa do ve
    ve = [(230,530),(230,720),(450,720),(450,650),(270,650),(270,530),(230,530)]
    ve1 = [(230,540),(230,730),(540,730),(540,730),(540,730),(230,530),(230,530)]
    ve2 = [(160, 984),(1818,931), (1230, 3), (622, 7)]
    ve3 = [(1383,263),(1605,850),(1503,950),(1340,410)]
    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    # video_path = "rtsp://admin:admin@192.168.0.108:554/cam/realmonitor?channel=1&subtype=0"
    
    try:
        vid = cv2.VideoCapture(int(video_path))
        
    except:
        vid = cv2.VideoCapture(video_path)
       
    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    last_in_box = []
    last_in_box2 = []
    pig_consume_time = dict()
    pig_consume_time2 = dict()
    frame_num = 0
    
    # while video is running
    while True:
       
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()
        
        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov4' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=100,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=100,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]
        
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score        
        )
        
        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        # allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        
        if FLAGS.count:

            cv2.putText(frame, "Pig Number : {}".format(count), (5, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (254, 255, 0), 1)
            print("Pig Number: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
       
        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       
        
        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # print(indices)
        in_box = []
        in_box2 = []
        
        
        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update >= 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            
            #ve tam object
            cx = int(((bbox[0])+(bbox[2])) // 2)
            cy = int(((bbox[1])+(bbox[3])) // 2)
           
            #draw line tracker by kido
            centroid_dict[int(track.track_id)].append((cx,cy))
            # if track.track_id in last_in_box:
                
            start_pt = (cx,cy)
            end_pt = (cx,cy)
            # cv2.line(frame,start_pt,end_pt,color,2)
                # img =  cv2.imshow('line', img)
            
            l = len(centroid_dict[track.track_id])
                # print(centroid_dict[track.track_id])

                # Caculate Activity Index  
            s = set(centroid_dict[track.track_id])
                # print(s)

                # Chuyển lại s về list
            unique_l = list(s)
            t_len = len(unique_l) # --> Activity Index
                # print("ID:",track.track_id)
                # print("Activity Index:",t_len)
            missing_number = str(None)
            
            for i in range(1, (track.track_id) + 2):
                if i == (track.track_id):
                    missing_number = i
                    break 
            if missing_number is not None:
                    track.track_id += str(missing_number)
            print("Updated list:", track.track_id)
            print(track.track_id)
                            
            for pt in range(len(centroid_dict[int(track.track_id)])):
                    if not pt + 1 == l:
                        start_pt = (centroid_dict[int(track.track_id)][pt][0],centroid_dict[int(track.track_id)][pt][1])
                        end_pt = (centroid_dict[int(track.track_id)][pt + 1][0],centroid_dict[int(track.track_id)][pt + 1][1])
                        if track.track_id != 404:
                            cv2.line(frame,start_pt,end_pt,color,2)
                        
                        # img = cv2.imshow('line', img)
                # print((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
            # Index heights
            index_r = abs(int((bbox)[0] - (bbox)[2]))
            index_d = abs(int((bbox)[1] - (bbox)[3]))
            # print ("index_ r: ",index_r)
            # print ("index_ d: ",index_d)
            kc = 3  #1 la khoang cach camera den vat the
            index_sensor_r = (( 4.8 * index_r) / 1280) # 4.8 kich thuoc cam bien don vi mm
            index_height_r = ((kc*index_sensor_r) / 2.3 ) # 2.3 là tieu cu camera
            # print (index_height_r)
            index_sensor_d = (( 3.6 * index_d) / 720) # 3.6 kich thuoc cam bien don vi mm
            index_height_d = ((kc*index_sensor_d) / 2.3 ) 
            # print (index_height_d)

            # config --------------
            fix_r_d = 80/100
            fix_cv = 90/100
            fix_weight_max = 130
            fix_weight_min = 60
            # ----------------------
            if index_r > index_d and index_r // index_d >= 1.5 :
                global w
                chu_vi = index_height_d / 2 * 3.14 * fix_cv  # chu vi vong nguc cua heo
                weights = int(chu_vi * chu_vi  * (index_height_r *fix_r_d) * 87.5  )
                w = ("Weight: " + str(weights)) # ---fix_r_d-------------------------------> Chưa tính sai số của trọng lượng của heo
                if weights <= fix_weight_max:
                    if weights >= fix_weight_min:
                        cv2.rectangle(frame, ((cx-35), (cy-15)),((cx)+(len(w))+30, (cy)+10), color, 2)
                        cv2.putText(frame,w,(cx-30 , cy),0, 0.35, (255,255,255) ,1)

            if index_d > index_r and index_d // index_r >= 1.5 :
                chu_vi = index_height_r / 2 * 3.14 * fix_cv
                weights = int(chu_vi * chu_vi * (index_height_d* fix_r_d) * 87.5 )
                w = ("Weight: " + str(weights))
                if weights <= fix_weight_max:
                    if weights >= fix_weight_min:
                        cv2.rectangle(frame, ((cx-35), (cy-15)),((cx)+(len(w))+30, (cy)+10), color, 2)
                        cv2.putText(frame,w,(cx-30 , cy),0, 0.35, (255,255,255) ,1)  # ----------------------------------> Chưa tính sai số của trọng lượng của he

                # dieu kien nhan dang trong 1 khu vc
             #kido ve
            
            inside_ve3 = cv2.polylines(frame, [np.array(ve3, np.int32)], True, (0, 0, 255),2)
            inside_ve3 = cv2.pointPolygonTest(np.array(ve3, np.int32),(int(bbox[2]),int(bbox[3])), True) 
            inside_ve4 = cv2.pointPolygonTest(np.array(ve3, np.int32),(int(bbox[2]),int(bbox[1])), True)
            # cv2.circle(frame,(int(bbox[0]),int(bbox[3])),5 ,color,-1) 
            #do chinh xac CNN
            sc = set(scores)
            scr =list(sc)
            if track.track_id < len(scores):
                # print("Percision ID {} ".format(track.track_id),scr[track.track_id])
                kq = scr[track.track_id]

                # Them so lan an cua heo: str(pig_consume_time[track.track_id] if track.track_id in pig_consume_time else 0)
            
            text = class_name + ":" + str(track.track_id) + "-Eaten:" + str(pig_consume_time[track.track_id] if track.track_id in pig_consume_time else 0 )
            text2 ="Drink: " + str(pig_consume_time2[track.track_id] if track.track_id in pig_consume_time2 else 0 )
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(frame,"Accuracy:"+"{:.2f}".format(kq),(int(bbox[2])-80,int(bbox[3])-5), 0, 0.35, color, 1)
            cv2.putText(frame,"Index:"+str(t_len),(int(bbox[2])-55,int(bbox[1])+15), 0, 0.35, color, 1)
            cropped_frame = frame[int(bbox[1])-30:int(bbox[3])+30,int(bbox[0])-30:int(bbox[2])+30]
            path = 'D:\File Code LVTN\yolov4-deepsort\data_crop'
            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-25)), (int(bbox[0])+(len(text))*8, int(bbox[1])), color, -1)
            if track.track_id != 404:
                cv2.putText(frame, text, (int(bbox[0]), int(bbox[1]-10)),0, 0.5, color,2)
                cv2.putText(frame, text2, (int(bbox[0]), int(bbox[1]+15)),0, 0.5, color,2)
                # cv2.imwrite(path,"id_"+str(track.track_id)+".png",cropped_frame)
                cv2.imwrite(os.path.join(path ,"id_"+ str(track.track_id)+"_"+ str(time_result.tm_year)+ '_' +str(time_result.tm_mon) +'_' + str(time_result.tm_mday)+'_' +str(time_result.tm_hour)  +'.png'), cropped_frame)
            # cv2.circle(frame,(cx , cy),5 ,color,-1)
            else:
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,255,255), 2)

                          
            cv2.putText(frame,localtime,(0,15), 0, 0.5, (255,0,127), 1)
                # Kiem tra xem bbox cua heo - nam trong vung duoc kiem tra hay khong, neu co thi them id cua heo vao in_box
            if CollisionCheck(ve3, (bbox)):
                in_box.append(track.track_id)
            if FLAGS.drink:
                inside_ve = cv2.polylines(frame, [np.array(ve, np.int32)], True, (0, 0, 255),2)
                cv2.putText(frame,"Drinking Area",(250,700),0,0.5,(0,255,255),1)   
                if CollisionCheck(ve, (bbox)):
                    in_box2.append(track.track_id)
                inside_ve1 = cv2.pointPolygonTest(np.array(ve, np.int32),(int(bbox[2]),int(bbox[3])),True)
                inside_ve5 = cv2.pointPolygonTest(np.array(ve, np.int32),(int(bbox[0]),int(bbox[3])),True)
                if inside_ve1 >= 0  :
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(0,0,255), 2)
                if inside_ve5 >= 0  :
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(0,0,255), 2)
                
            if inside_ve3 >= 0  :
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(0,0,255), 2)
            if inside_ve4 >= 0  :
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(0,0,255), 2)
            #crop image
            
        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
        print("Currently inside box: " + str(in_box))       
        print(track.track_id)
        # so sanh last_in_box va in_box de biet so lan an
        # neu co id trong in_box khong xuat hien trong last_in_box -> con heo id vao an
          
        for id in in_box:
            if id not in last_in_box:
                if id not in pig_consume_time:
                    pig_consume_time[id] = 1
                else:
                    pig_consume_time[id] += 1
                    music = pyglet.resource.media('pip.wav')
                    music.play()
        last_in_box = in_box
        count_eat = (len(last_in_box))
        
        cv2.putText(frame, "Pig Eating: {}".format(count_eat), (5, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,127), 1)
        for id2 in in_box2:
            if id2 not in last_in_box2:
                if id2 not in pig_consume_time2:
                    pig_consume_time2[id2] = 1
                else:
                    pig_consume_time2[id2] += 1
                    music = pyglet.resource.media('pip.wav')
                    music.play()
        
        
        last_in_box2= in_box2
        count_eat2 = (len(last_in_box2))
        cv2.putText(frame, "Pig Drinking: {}".format(count_eat2), (5, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 1)
        print("Number of times to drink: " + str(pig_consume_time2))
        print(last_in_box2)
        
        print("Number of times to eaten: " + str(pig_consume_time))
        #Timer save image
        localtime = time.asctime( time.localtime(time.time()) )
        time_result = time.localtime(time.time())
        print ("Local current time :", localtime)

        if time_result.tm_hour == 6 and time_result.tm_min == 0  and time_result.tm_sec == 0 :
            path = 'D:\File Code LVTN\yolov4-deepsort\morning'
            cv2.imwrite(os.path.join(path , str(time_result.tm_year)+ '_' +str(time_result.tm_mon) +'_' + str(time_result.tm_mday)+'_' +str(time_result.tm_hour)  +'.png'), result)
        if time_result.tm_hour == 12 and time_result.tm_min == 0  and time_result.tm_sec == 0 :
            path = 'D:\File Code LVTN\yolov4-deepsort\afternoon'
            cv2.imwrite(os.path.join(path , str(time_result.tm_year)+ '_' +str(time_result.tm_mon) +'_' + str(time_result.tm_mday)+'_' +str(time_result.tm_hour)  +'.png'), result)
        if time_result.tm_hour == 17 and time_result.tm_min == 0  and time_result.tm_sec == 0 :
            path = 'D:\File Code LVTN\yolov4-deepsort\everning'
            cv2.imwrite(os.path.join(path , str(time_result.tm_year)+ '_' +str(time_result.tm_mon) +'_' + str(time_result.tm_mday)+'_' +str(time_result.tm_hour)  +'.png'), result)
        #Clear function after end day at 0:0:0 am     
        if time_result.tm_hour == 0 and time_result.tm_min == 0  and time_result.tm_sec == 0 :
            path = 'D:\File Code LVTN\yolov4-deepsort\endday'
            cv2.imwrite(os.path.join(path , str(time_result.tm_year)+ '_' +str(time_result.tm_mon) +'_' + str(time_result.tm_mday)+'_' +str(time_result.tm_hour)  +'.png'), result)
            pig_consume_time.clear()
            centroid_dict.clear()
            
        # calculate frames per second of running detections 
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)                        
        output = cv2.resize(result,(1280, 720))
        
        if not FLAGS.dont_show:            
            cv2.imshow("Output Video", result)                      
                # cv2.imshow("Video", frame)            
        # if output flag is set, save video file
       
            
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break       
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
