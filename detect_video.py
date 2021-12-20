import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/video.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.1, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')

global upper_object, lower_object
upper_object=[]
lower_object=[]

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video
    upper_object=[0]
    lower_object=[0]

    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(width)
        print(height)
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
    
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
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
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        image = utils.draw_bbox(frame, pred_bbox)
        fps = 1.0 / (time.time() - start_time)
        ####
##        coord_bbox=np.array(boxes)
##        coord_bbox_all=coord_bbox[0,:,:]
##        no_zero_coord_bbox_all=coord_bbox_all[~np.all(coord_bbox_all==0,axis=1)]
##        #extract_value_x_y=no_zero_coord_bbox_all[:,:2] ##(no need) (x, 1,3) (y, 0,2)
##        left=no_zero_coord_bbox_all[:,1]
##        right=no_zero_coord_bbox_all[:,3]
##        up=no_zero_coord_bbox_all[:,0]
##        lower=no_zero_coord_bbox_all[:,2]
##        label_pred=classes.numpy()
##        wide_bbox_x=right-left
##        wide_bbox_y=lower-up
##        extract_value_x=left+((right-left)/2)
##        extract_value_y=up+((lower-up)/2)
##        join_bbox=np.stack((extract_value_x,extract_value_y))
##        print("nilai x : ",extract_value_x)
##        print("nilai y : ",extract_value_y)
        
        ## persepsi sb. x
##        left_object=[]
##        middle_object=[]
##        right_object=[]
##        global left_object_counts, middle_object_counts, right_object_counts
##        left_object_counts=0
##        middle_object_counts=0
##        right_object_counts=0
##        for i in extract_value_x:
##            if i<0.4:
##                left_object.append(i)
##                left_object_counts=len(left_object)
##            elif i>0.4 and i<0.6:
##                middle_object.append(i)
##                middle_object_counts=len(middle_object)
##            elif i>0.6:
##                right_object.append(i)
##                right_object_counts=len(right_object)
##        if left_object != []:
##            print(left_object_counts," objek berada di kiri")
##        if middle_object != []:
##            print(middle_object_counts," objek berada di depan")
##        if right_object != []:
##            print(right_object_counts," objek berada di kanan")
##            
##        ## tinggal bikin persepsi sb. y## kurang pengolahan kalau multiclass
##        #up threshold=0.12
##        #low threeshold=0.88
##        far_object=[]
##        close_object=[]
##        coalision_object=[]
##        global far_object_counts, close_object_counts, coalision_object_counts
##        far_object_counts=0
##        close_object_counts=0
##        coalision_object_counts=0
##        for i in extract_value_y:
##            if i<0.35 :
##                far_object.append(i)
##                far_object_counts=len(far_object)
##            elif i>0.35 and i<0.6:
##                close_object.append(i)
##                close_object_counts=len(close_object)
##            elif i>0.6:
##                coalision_object.append(i)
##                coalision_object_counts=len(coalision_object)
##
##        ## Decision making
##        #if left_object_counts !=[]:
##          #  if coalision_object_counts !=[]:
##           #     print("Dilarang menyelip ke kiri")
##           # if close_object_counts !=[]:
##           #     print("Dilarang menyelip ke kiri")
##           # if far_object_counts !=[]:
##           #     print("Silahkan menyelip ke kiri, Hati-hati!")
##      #  if middle_object_counts !=[]:
##           # if coalision_object_counts !
##        
##       # if left_object != []:
##       #     print(left_object_counts," objek berada di kiri, dengan detail ", far_object_counts," objek jauh, ", close_object_counts," objek dekat, ", coalision_object_counts," objek kemungkinan menabrak")                
##       # if middle_object != []:
##       #     print(middle_object_counts," objek berada di depan, dengan detail ", far_object_counts," objek jauh, ", close_object_counts," objek dekat, ", coalision_object_counts," objek kemungkinan menabrak")                
##       # if right_object != []:
##       #     print(right_object_counts," objek berada di kanan, dengan detail ", far_object_counts," objek jauh, ", close_object_counts," objek dekat, ", coalision_object_counts," objek kemungkinan menabrak")                
##                
##        ## skip dlu yg di bawah. bawah untuk tracking
##        #for i in extract_value_y:
##           # if len(upper_object)<5:
##            #    upper_object.append(i)
##           # elif len(upper_object)==5:
##            #    upper_object=np.delete(upper_object,0)
##            #    value= np.array([i])
##             #  array= np.array(upper_object)
##             #   upper_object=np.append(array,value)
##        #abc=upper_object.append(extract_value_y)
##
##        ## testing
####        length=len(extract_value_x)
####        for i=0 to length:
####            if extract_value_x[i]<0.4:
####                if extract_value_y[i]<0.35:
####                    left_coalision_object=left_coalision_object+1
####                elif extract_value_y[i]>0.35 and extract_value_y[i]<0.6:
####                    left_close_object=left_close_object+1
####                elif extract_value_y[i]>0.6:
####                    left_far_object=left_far_object+1
####            elif extract_value_x[i]>0.4 and extract_value_x[i]<0.6:
####                if extract_value_y[i]<0.35:
####                    middle_coalision_object=middle_coalision_object+1
####                elif extract_value_y[i]>0.35 and extract_value_y[i]<0.6:
####                    middle_close_object=middle_close_object+1
####                elif extract_value_y[i]>0.6:
####                    middle_far_object=middle_far_object+1
####            elif extract_value_x[i]>0.6:
####                if extract_value_y[i]<0.35:
####                    right_coalision_object=right_coalision_object+1
####                elif extract_value_y[i]>0.35 and extract_value_y[i]<0.6:
####                    right_close_object=right_close_object+1
####                elif extract_value_y[i]>0.6:
####                    right_far_object=right_far_object+1
##      #  if left_object ==[]:
##           # if left_coalision_object !=[]
##        print("Jumlah objek di layar ",len(upper_object))
##        print("Label: ", label_pred)
##        ####
##        print("Left object coord: ", left_object)
##        print("Front object coord: ", middle_object)
##        print("Right object coord: ", right_object)
##       # print("Join bbox= ")
##        #print(join_bbox)
##        print("Wide bbox X: ", wide_bbox_x)
##        print("Wide bbox y: ", wide_bbox_y)
##        left_object=[]
##        middle_object=[]
##        right_object=[]
        print("FPS: %.2f" % fps)
        print("##################################") ##
        result = np.asarray(image)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("result", result)
        
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
