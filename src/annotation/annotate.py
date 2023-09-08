#!/usr/bin/env python3

#-------------------------------------YOLOV7------------------------------------#
import time
import torch
import cv2
import numpy as np
from torchvision import transforms

import os
import sys
sys.path.append(os.path.abspath('./yolov7'))
sys.path.append(os.path.abspath('./yolov7/utils'))

#from utils.datasets import letterbox
#from utils.general  import non_max_suppression_kpt
#from utils.plots    import output_to_keypoint, plot_skeleton_kpts
from datasets import letterbox
from general  import non_max_suppression_kpt
from plots    import output_to_keypoint, plot_skeleton_kpts, plot_skeleton_kpts2

# Change forward pass input size.
input_size = 256

# Select the device based on hardware configs.
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print('Selected Device : ', device)

# Load keypoint detection model.
weights = torch.load('./yolov7/yolov7-w6-pose.pt', map_location=device)
model = weights['model']
# Load the model in evaluation mode.
_ = model.float().eval()
# Load the model to computation device [cpu/gpu/tpu]
model = model.to(device)
#model = model.half().to(device)
#model = model.double().to(device)
_ = model.eval()
#-------------------------------------------------------------------------------#

import os
import cv2
import sys
import argparse
import numpy as np

sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./src'))

from annotation import utils
from tools import visualize
from natsort import natsorted

#--------------------------------INITIALIZATIONS--------------------------------#
coord = (20, 20)
del_entries = []
# Boolean to control bbox drawing loop.
draw_box = False
remove_box = False
# Update image.
updated_img = None
clean_img = None
org_img = None
max_area = 0
reset = False
PADDING = 10
#Toggle  = False
min_area_ratio = 0.000
manual_assert_boxes = []
#swap_channel = False
#channel_count = 0
#blob_conf_key = False
#blob_conf = False
#-------------------------------------------------------------------------------#



def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-img', '--img',
        help='path to the images file directory'
    )
    parser.add_argument(
        '--ann',
        required=True,
        help='path to the annotations directory'
    )    
    parser.add_argument(
        '-vid', '--vid',
        help='path to the video file'
    )
    #parser.add_argument(
    #    '-T', '--toggle-mask',
    #    dest='toggle',
    #    action='store_true',
    #    help='Toggle Threshold Mask'
    #)
    parser.add_argument(
        '-auto', '--autonomous',
        help='autonomous mode'
    )
    parser.add_argument(
        '--resume',
        help='path to annotations/labels directory'
    )
    parser.add_argument(
        '--skip',
        type=int,
        default=3,
        help="Number of frames to skip."
    )
    #parser.add_argument(
    #    '-blob', '--blob',
    #    dest='blob_conf',
    #    action='store_true',
    #    help='Turn on advanced shape and size filter control '
    #)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    else:
        args = parser.parse_args()
    return args


def image_paths(dir):
    # Iterate through the images.
    images_path = os.listdir(dir)
    # Remove files other than images.
    updated_images_paths = []

    for file in images_path:
        if ('.jpg' in file) or ('.png' in file) or ('.jpeg' in file):
            updated_images_paths.append(file)
    # print(f"Test1: {updated_images_paths}")
    updated_images_paths = natsorted(updated_images_paths)
    # print(f"Test2: {updated_images_paths}")

    with open('names.txt', 'w') as f:
        for path in updated_images_paths:
            ln = [path, '\n']
            f.writelines(ln)

    return updated_images_paths


def get_init_bboxes(img):
    """
    Returns bounding box using contour analysis.
    """
    global max_area, min_area_ratio
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_cnt = sorted(contours, key=cv2.contourArea, reverse = True)
    sorted_cnt.remove(sorted_cnt[0])
    max_area = img.shape[0] * img.shape[1]
    bounding_rect_boxes = []

    for cnt in sorted_cnt:
        x,y,w,h = cv2.boundingRect(cnt)
        cnt_area = w * h
        if (min_area_ratio * max_area < cnt_area):
            x = x - PADDING
            y = y - PADDING
            x = 0 if x < 0 else x
            y = 0 if y < 0 else y
            x = img.shape[1] if x > img.shape[1] else x
            y = img.shape[0] if y > img.shape[0] else y
            bounding_rect_boxes.append(((x, y), (x+w, y+h)))
    return bounding_rect_boxes

def get_init_poses(img):
    """
    Returns bounding box using yolov7
    """
    global max_area, min_area_ratio
    bounding_rect_boxes = []
    bbox_conf = []
    poses = []

    original_height = img.shape[0]
    original_width = img.shape[1]
    max_area = original_height * original_width
        
    # Get the frame, including width and height.
    orig_image = img.copy()
    frame_width = orig_image.shape[0]
    frame_height = orig_image.shape[1]
    frame_max = max(frame_width,frame_height)
        
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = letterbox(image, (frame_max), stride=64, auto=True)[0]
    image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    image = image.to(device)
    #image = image.half()        
    #image = image.double()

    with torch.no_grad():
        output, _ = model(image)
                    
    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
    output = output_to_keypoint(output)
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)        

    for idx in range(output.shape[0]):

        # map coordinates to left side
        kpts = output[idx, 7:].T
        #steps = 3
        #num_kpts = len(kpts) // steps
        #for kid in range(num_kpts):
        #    x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        #    x_coord = x_coord - 640
        #    kpts[steps * kid] = x_coord
        
        #plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

        # Comment/Uncomment the following lines to show bounding boxes around persons.
        xmin, ymin = (output[idx, 2]-output[idx, 4]/2), (output[idx, 3]-output[idx, 5]/2)
        xmax, ymax = (output[idx, 2]+output[idx, 4]/2), (output[idx, 3]+output[idx, 5]/2)
        conf = output[idx,6]
        #cv2.rectangle(
        #    nimg, 
        #    (int(xmin), int(ymin)),
        #    (int(xmax), int(ymax)),
        #    color=(255, 0, 0),
        #    thickness=2, 
        #    lineType=cv2.LINE_AA
        #)
        #print( '[',idx,'] = ',xmin,ymin,xmax,ymax,output[idx,0:7] )

        #x,y,w,h = cv2.boundingRect(cnt)
        x = int(xmin)
        y = int(ymin)
        w = int(xmax-xmin)
        h = int(ymax-ymin)
        
        pose_area = w * h
        if (min_area_ratio * max_area < pose_area):
            x = x - PADDING
            y = y - PADDING
            x = 0 if x < 0 else x
            y = 0 if y < 0 else y
            x = img.shape[1] if x > img.shape[1] else x
            y = img.shape[0] if y > img.shape[0] else y
            bounding_rect_boxes.append(((x, y), (x+w, y+h)))
            bbox_conf.append(conf)
            poses.append(kpts)

    #print(bounding_rect_boxes,poses)
            
    return bounding_rect_boxes, bbox_conf, poses

def write_annotations(ann_path, img_file, shape, boxes, bbox_conf, poses, aspect_ratio):
    """
    Saves annotations to a text file in YOLO format,
    class, x_centre, y_centre, width, height
    """
    height_f = aspect_ratio[0]
    width_f = aspect_ratio[1]
    img_height = int(shape[0]*height_f)
    img_width = int(shape[1]*width_f)
    # print('Check : ', height_f, 'Width : ', width_f)

    # Check if the Annotations folder exists
    if not os.path.exists(ann_path):
        os.mkdir(ann_path)
        
    ann_file = os.path.join(ann_path,img_file + '.txt')
    with open( ann_file , 'w') as f:
        #for box in boxes:
        for i,box in enumerate(boxes):
            x1, y1 = box[0][0], box[0][1]
            x2, y2 = box[1][0], box[1][1]
            # Map to the original image size.
            x1 = int(width_f*x1)
            y1 = int(height_f*y1)
            x2 = int(width_f*x2)
            y2 = int(height_f*y2)

            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            x_centre = int(x1 + width/2)
            y_centre = int(y1 + height/2)

            norm_xc = float(x_centre/img_width)
            norm_yc = float(y_centre/img_height)
            norm_width = float(width/img_width)
            norm_height = float(height/img_height)
            
            conf = bbox_conf[i]
            
            kpts = poses[i]
            steps = 3
            num_kpts = len(kpts) // steps
            #print('keypoints', kpts)
            kpt_annotations = ''
            for kid in range(num_kpts):
                 #print("[",kid,"] x=",kpts[steps * kid], " y=",kpts[steps * kid + 1], "id=",kpts[steps * kid + 2])
                 kx = float(kpts[steps * kid + 0]/img_width)
                 ky = float(kpts[steps * kid + 1]/img_height)
                 kc = kpts[steps * kid + 2]
                 kpt_annotations = kpt_annotations + ' ' + str(kx) + ' ' + str(ky) + ' ' + str(kc)
            

            yolo_annotations = ['0', ' ' + str(norm_xc), ' ' + str(norm_yc), ' ' + str(norm_width), ' ' + str(norm_height), ' ' + str(conf), kpt_annotations, '\n']
            f.writelines(yolo_annotations)

def read_annotations(img, ann_file):
    height, width = img.shape[:2]
    bboxes = []
    bbox_conf = []
    poses = []
    with open(ann_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            name = line.replace('\n', '')
            #class_id, xc, yc, w, h = name.split(' ')
            name_split = name.split(' ')
            class_id = name_split[0]
            xc = name_split[1]
            yc = name_split[2]
            w = name_split[3]
            h = name_split[4]
            
            class_id = int(class_id)
            xc, yc = float(xc), float(yc)
            h, w = float(h), float(w)
            box_h = int(height*h)
            box_w = int(width*w)
            x_center = int(xc*width)
            y_center = int(yc*height)
            x1 = x_center - int(box_w/2)
            y1 = y_center - int(box_h/2)
            x2 = x1 + box_w 
            y2 = y1 + box_h
            p1 = (x1, y1)
            p2 = (x2, y2)
            bboxes.append((p1, p2))

            conf = name_split[5]
            conf = float(conf)
            bbox_conf.append(conf)
            
            kpts = name_split[6:]
            #print(kpts)
            steps = 3
            num_kpts = len(kpts) // steps
            for kid in range(num_kpts):
                 kpts[steps * kid + 0] = float(kpts[steps * kid + 0])*width
                 kpts[steps * kid + 1] = float(kpts[steps * kid + 1])*height
                 kpts[steps * kid + 2] = float(kpts[steps * kid + 2])
                 #print("[",kid,"] x=",kpts[steps * kid], " y=",kpts[steps * kid + 1], "id=",kpts[steps * kid + 2])
            #print(kpts)
            poses.append(kpts)  
            

    return bboxes, bbox_conf, poses

#def draw_init_annotations(img, boxes):    
def draw_init_annotations(img, boxes, bbox_conf, poses, thresh_bbox=0.5, thresh_kpts=0.5):
    #for box in boxes:
    for i, box in enumerate(boxes):
        if bbox_conf[i] >= thresh_bbox:
            x1, y1, x2, y2 = box[0][0], box[0][1], box[1][0], box[1][1]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2, cv2.LINE_AA)

            plot_skeleton_kpts(img, poses[i], 3, thresh=thresh_kpts)    
    
def get_box_area(tlc, brc):
    x1, y1, x2, y2 = tlc[0], tlc[1], brc[0], brc[1]
    area = abs(x2 - x1) * abs(y2 - y1)
    return area

def draw_dotted_lines(img, pt1, pt2, color, thickness=1, style='dotted', gap=10):
    """
    Draw dotted lines. 
    Adopted from StackOverflow.
    """
    dist = ((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**0.5
    pts= []
    for i in  np.arange(0, dist, gap):
        r = i/dist
        x = int((pt1[0] * (1-r) + pt2[0] * r) + 0.5)
        y = int((pt1[1] * (1-r) + pt2[1] * r) + 0.5)
        p = (x,y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i%2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1


def get_coordinate(event, x, y, flags, params):
    global coord, tlc, draw_box, bboxes, remove_box, clean_img, del_entries

    if event == cv2.EVENT_MOUSEMOVE:
        # Current coordinate. Updated every instant with the cursor.
        coord = (x, y)
        
    if event == cv2.EVENT_LBUTTONDOWN:
        # Clicked point.
        tlc = (x, y)
        draw_box = True

    if event == cv2.EVENT_LBUTTONUP:
        draw_box = False
        # Modify the code to draw rectangles only when area is greater than 
        # a particular threshold.
        # Also don't draw 'point' rectangles.
        if tlc != coord:
            cv2.rectangle(clean_img, tlc, coord, (255,0,0), 2, cv2.LINE_AA)
        # Append the final bbox coordinates to a global list.
        # Also remove very very small annotations.
        #area = utils.get_box_area(tlc, coord)
        area = get_box_area(tlc, coord)
        if area > 0.0001 * max_area:
            bboxes.append((tlc, coord))
            manual_assert_boxes.append((tlc, coord))

    # Add logic to remove a particular bounding box of double clicked in that area.
    if event == cv2.EVENT_LBUTTONDBLCLK:
        remove_box = True
        # Update the bboxes container.
        hit_point = (x, y)
        for point in bboxes:
            x1, y1 = point[0][0], point[0][1]
            x2, y2 = point[1][0], point[1][1]

            # Arrange small to large. Swap variables if required.
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            if hit_point[0] in range(x1, x2) and hit_point[1] in range(y1, y2):
                del_entries.append(point)
                bboxes.remove(point)
                # print('removed!')
        # print('Updated Bboxes: \n', bboxes)

        clean_img = org_img # Check point.
                
        # Update the bboxes annotations.
        if len(bboxes) >= 1:
            for point in bboxes:
                cv2.rectangle(clean_img, point[0], point[1], (255,0,0), 2, cv2.LINE_AA)
                # print('Boxes have been redrawn! ', point)
        remove_box = False


def update_bboxes(bboxes, del_entries, manual):
    for deleted_box in del_entries:
        # Deleted box coordinates. Area increased by 10%.
        x1_del, y1_del = int(0.9*deleted_box[0][1]), int(0.9*deleted_box[0][1])
        x2_del, y2_del = int(1.1*deleted_box[1][0]), int(1.1*deleted_box[1][1])
        for box in bboxes:
            x1, y1 = box[0][0], box[0][1]
            x2, y2 = box[1][0], box[1][1]
            # Check if the points are inside the deleted region.
            if (x1_del< x1 < x2_del) and (x1_del < x2 < x2_del) and (y1_del < y1 < y2_del) and (y1_del < y2 < y2_del):
                bboxes.remove(box)
        # Add manually drawn boxes as well, given that it is not from the deleted list.
        if len(manual) > 0:
            for manual_box in manual:
                if (manual_box not in bboxes) and (manual_box not in del_entries):
                    bboxes.append(manual_box)
    return bboxes


# Return a single channel image.
#def channel_select(img, ch_count):
#    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    blue, green, red = cv2.split(img)
#    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#    hue, sat, lightness = cv2.split(img_hsv)
#    channel_list = [gray, blue, green, red, hue, sat, lightness]
#    channel_name_list = ["Gray",
#                         "Blue",
#                         "Green",
#                         "Red",
#                         "Hue",
#                         "Saturation",
#                         "Lightness"]
#    # print(f"Channel count : {channel_count}")
#
#    return channel_list[ch_count]


#def filter_shapes(cnt, type=None):
#    pass

def ignore(x):
    pass

# Load images.
def main():
    global coord, tlc, draw_box, clean_img, org_img
    global min_area_ratio
    #global channel_count
    global remove_box, bboxes, del_entries, reset
    global bbox_conf, poses
    #global Toggle
    global manual_assert_boxes
    #global swap_channel
    #global blob_conf_key, blob_conf
    
    args = parser_opt()

    if args.img is not None:
        file_type = 'img'
        IMAGES_DIR = args.img
        if args.vid is None:
            if not os.path.isdir(IMAGES_DIR):
                print('Please enter correct images directory path.')
                sys.exit()
    else:
        print('Please provide the path to the image folder.')

    if args.vid is not None:
        file_type = 'vid'
        VID_PATH = args.vid
        if not os.path.isfile(VID_PATH):
            print('Please enter correct path to the video file.')
            sys.exit()        

    if args.ann is not None:
        LABELS_DIR = args.ann
        if not os.path.isdir(LABELS_DIR):
            os.makedirs(LABELS_DIR)
    else:
        LABELS_DIR = "labels"

    if args.autonomous is not None:
        print("Autonomous Mode")
    
    if file_type == 'img':
        file_path = IMAGES_DIR
        updated_images_paths = image_paths(file_path)
        if args.resume is not None:
            completed_images = natsorted(os.listdir(args.resume))
            completed_images_names = []

            for file in completed_images:
                completed_images_names.append(file.split('.')[0])
            
            updated_im_paths = []
            for source_file in updated_images_paths:
                if not source_file.split('.')[0] in completed_images_names:
                    updated_im_paths.append(source_file)

            updated_images_paths = updated_im_paths

    elif file_type == 'vid':
        file_path = VID_PATH
        if not os.path.exists(IMAGES_DIR):
            # Delete existing images. Feature to be added.
            os.mkdir(IMAGES_DIR)
        loading_img = np.zeros([400, 640, 3], dtype=np.uint8)
        skip_count = args.skip
        cap = cv2.VideoCapture(file_path)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        i = 0
        count = 0
        while cap.isOpened():
            if count/skip_count == 0:
                ret, frame = cap.read()
                
                load = loading_img.copy()
                if not ret:
                    print('Unable to read frame')
                    break
                cv2.putText(load, f"Frames: {i} / {int(frame_count)}", 
                    (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)
                cv2.putText(load, f"Sequencing...", 
                    (260, 200), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)
                cv2.imshow('Images', load)
                cv2.imwrite(os.path.join(IMAGES_DIR,'img-{}.jpg'.format(i)), frame)
            key = cv2.waitKey(1)
            i += 1
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyWindow('Images')
        updated_images_paths = image_paths(IMAGES_DIR)
        file_path = IMAGES_DIR
        print(f"Images Saved to {IMAGES_DIR}")

    # Named window for Trackbars.
    cv2.namedWindow('Annotate')
    #cv2.createTrackbar('threshold', 'Annotate', 127, 255, utils.ignore)
    #cv2.createTrackbar('minArea', 'Annotate', 5, 500, utils.ignore)
    cv2.createTrackbar('frameNum'  , 'Annotate', 0, len(updated_images_paths), ignore)
    cv2.createTrackbar('threshBBOX', 'Annotate', 50, 100, ignore)
    cv2.createTrackbar('threshKPTS', 'Annotate', 50, 100, ignore)
    cv2.createTrackbar('minArea'   , 'Annotate', 5, 500, ignore)
    
    prev_frame_num = 0
    frame_num = 0
    num = 0
    print(updated_images_paths[num])
    while True:
        if num == len(updated_images_paths):
            print('Task Completed.')
            break

        img_path = os.path.join(file_path, updated_images_paths[num])
        #print(img_path)
        img = cv2.imread(img_path)
        original_height = img.shape[0]
        original_width = img.shape[1]
        
            
        #resized_image = utils.aspect_resize(img)
        #current_height = resized_image.shape[0]
        #current_width = resized_image.shape[1]
        current_height = img.shape[0]
        current_width = img.shape[1]

        aspect_h = original_height/ current_height
        aspect_w = original_width/current_width
        aspect_ratio = [aspect_h, aspect_w]

        # Add all side padding 20 px.
        #prev_thresh = 127
        prev_min_area = 0.00
        
        while True:
            ##img_gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            #img_gray = channel_select(resized_image, channel_count)
            #img_gray_padded = cv2.copyMakeBorder(img_gray, PADDING, PADDING, PADDING, PADDING, cv2.BORDER_CONSTANT, None, value=255)
            #im_annotate = resized_image.copy()
            im_annotate = img.copy()

            # Get trackbar threshold value.
            #thresh_val = cv2.getTrackbarPos('threshold', 'Annotate')
            frame_num = cv2.getTrackbarPos('frameNum', 'Annotate')
            thresh_bbox = cv2.getTrackbarPos('threshBBOX', 'Annotate')
            thresh_bbox = thresh_bbox*(1/100)
            thresh_kpts = cv2.getTrackbarPos('threshKPTS', 'Annotate')
            thresh_kpts = thresh_kpts*(1/100)
            min_area_ratio = cv2.getTrackbarPos('minArea', 'Annotate')
            min_area_ratio = min_area_ratio*(1/10000)

            #ret, thresh = cv2.threshold(img_gray_padded, thresh_val, 255, cv2.THRESH_BINARY)

            # The primary thresh image will be used to adjust thresholding when required.
            #primary_thresh = thresh

            # Store the original image, might require later.
            org_img = im_annotate
            
            #if clean_img is None:
            #    # Find contours and draw bounding rects.
            #    bboxes = get_init_bboxes(thresh)
            #    bboxes = update_bboxes(bboxes, del_entries, manual_assert_boxes)
            if clean_img is None:
                image_file_path = os.path.join(IMAGES_DIR,updated_images_paths[num])
                annotation_file_path = os.path.join(LABELS_DIR,updated_images_paths[num].split('.')[0] + '.txt')
                #print("image = ", image_file_path)
                #print("label = ", annotation_file_path)
                #print("Checking if annotation file exists ...")
                if os.path.isfile(annotation_file_path) is True:
                    print("\tReusing existing annotations ...")
                    bboxes,bbox_conf,poses = read_annotations(img, annotation_file_path)
                else:
                    print("\tCalculating new annotations ...")
                    bboxes,bbox_conf,poses = get_init_poses(img)
                    bboxes = update_bboxes(bboxes, del_entries, manual_assert_boxes)

            # If threshold slider is moved, update bounding rects.
            #elif (clean_img is not None) and (prev_thresh != thresh_val):
            #    reset = False
            #    bboxes = get_init_bboxes(thresh)
            #    bboxes = update_bboxes(bboxes, del_entries, manual_assert_boxes)
            #    # print('Check : ', del_entries)
            #elif (clean_img is not None) and prev_min_area != min_area_ratio:
            #    bboxes = get_init_bboxes(thresh)
            #    bboxes = update_bboxes(bboxes, del_entries, manual_assert_boxes)
            #elif( clean_img is not None) and swap_channel:
            #    bboxes = get_init_bboxes(thresh)
            #    bboxes = update_bboxes(bboxes, del_entries, manual_assert_boxes)
            #    swap_channel = False
            else:
                # Update the thresh image if annotation performed once.
                im_annotate = clean_img
            
            clean_img = im_annotate.copy()
            #prev_thresh = thresh_val
            prev_thresh_bbox = thresh_bbox
            prev_thresh_kpts = thresh_kpts            
            prev_min_area = min_area_ratio

            #draw_init_annotations(im_annotate, bboxes)
            draw_init_annotations(im_annotate, bboxes,bbox_conf,poses,thresh_bbox=thresh_bbox,thresh_kpts=thresh_kpts)
                
            cv2.setMouseCallback('Annotate', get_coordinate)

            h,w = im_annotate.shape[:2]
            horizontal_pt1 = (0, coord[1])
            horizontal_pt2 = (w, coord[1])
            vertical_pt1 = (coord[0], 0)
            vertical_pt2 = (coord[0], h)

            #utils.draw_dotted_lines(im_annotate, horizontal_pt1, horizontal_pt2, (0,0,200))
            #utils.draw_dotted_lines(im_annotate, vertical_pt1, vertical_pt2, (0,0,200))
            draw_dotted_lines(im_annotate, horizontal_pt1, horizontal_pt2, (0,0,200))
            draw_dotted_lines(im_annotate, vertical_pt1, vertical_pt2, (0,0,200))

            if draw_box:
                cv2.rectangle(im_annotate, tlc, coord, (255,0,0), 2, cv2.LINE_AA)

            if reset:
                im_annotate = org_img
      
            #if args.toggle or Toggle:
            #    # Disply mask, resized to half the annotation window.
            #    cv2.imshow('Mask', cv2.resize(thresh, None, fx=0.5, fy=0.5))
            
            cv2.imshow('Annotate', im_annotate)
            
            #if blob_conf_key or blob_conf:
            #    cv2.namedWindow('Filters')
            #    cv2.createTrackbar('circularity', 'Filters', 50, 100, utils.ignore)
            #    cv2.createTrackbar('inertia', 'Filters', 50, 100, utils.ignore)
            #    cv2.createTrackbar('convexity', 'Filters', 50, 100, utils.ignore)
            #    cv2.imshow('Filters', cv2.resize(thresh, None, fx=0.5, fy=0.5))
            # print(f"Org : {im_annotate.shape}, Thresh: {thresh.shape}")

            key = cv2.waitKey(1)
            
            # Store current threshold trackbar value to a temporary variable.
            #thresh_val_prev = thresh_val
            
            # Press n to go to the next image.
            if key == ord('n') or key == ord('d') or args.autonomous is not None:
                clean_img = None
                #utils.save(updated_images_paths[num].split('.')[0], (h, w), bboxes, aspect_ratio)
                write_annotations(LABELS_DIR,updated_images_paths[num].split('.')[0], (h, w), bboxes, bbox_conf, poses, aspect_ratio)
                num += 1
                bboxes = []
                del_entries = []
                manual_assert_boxes = []
                # print(f"Annotations Saved to {os.getcwd()}/labels")
                print(os.path.join(LABELS_DIR,updated_images_paths[num]))
                break
                
            # Press b to go to the previous image.
            if key == ord('b') or key == ord('a'):
                # print('Back Key Pressed.')
                # Go back one step.
                clean_img = None
                #utils.save(updated_images_paths[num].split('.')[0], (h, w), bboxes, aspect_ratio)
                write_annotations(LABELS_DIR,updated_images_paths[num].split('.')[0], (h, w), bboxes, bbox_conf, poses, aspect_ratio)
                if num != 0:
                    num -= 1
                # print(f"Annotations Saved to {os.getcwd()}/labels")
                bboxes = []
                del_entries = []
                manual_assert_boxes = []
                print(os.path.join(LABELS_DIR,updated_images_paths[num]))
                break

            if key == ord('c'):
                reset = not reset
                #utils.save(updated_images_paths[num].split('.')[0], (h, w), bboxes, aspect_ratio)
                write_annotations(LABELS_DIR,updated_images_paths[num].split('.')[0], (h, w), bboxes, bbox_conf, poses, aspect_ratio)
                bboxes = []
                del_entries = []
                manual_assert_boxes = []

            #if key == ord('s'):
            #    channel_count += 1
            #    if channel_count == 7:
            #        channel_count = 0
            #    swap_channel = True
            
            #if key == ord('t'):
            #    Toggle = not Toggle
            #    if Toggle == False:
            #        try:
            #            cv2.destroyWindow('Mask')
            #        except:
            #            pass

            #if key == ord('f'):
            #    print('Filters window initiated.')
            #    blob_conf_key = not blob_conf_key
            #    if blob_conf_key == False:
            #        try:
            #            cv2.destroyWindow('Filters')
            #        except:
            #            pass
                
            if key == ord('q'):
                print(f"Annotations Saved to {os.getcwd()}/labels")
                sys.exit()

            # Use trackbar to specify frame
            if frame_num != prev_frame_num:
                prev_frame_num = frame_num
                num = frame_num
                print('[INFO] frame num = ',num)
                #
                clean_img = None
                bboxes = []
                del_entries = []
                manual_assert_boxes = []
                print(os.path.join(LABELS_DIR,updated_images_paths[num]))
                break
                
    print(f"Annotations Saved to {os.getcwd()}/labels")

if __name__ == '__main__':
    main()
    
