import os
import sys
import csv
import cv2
import argparse
from xml.etree.ElementTree import parse
import codecs

parser = argparse.ArgumentParser(description='Hello parser')
parser.add_argument('--dataset_name', dest='dataset_name', default='parking', help='cctv or parking dataset')
parser.add_argument('--delay', dest='delay', type=int, default=1,
                    help='delay time duration between two continuous frames')
parser.add_argument('--avg_pt', dest='avg_pt', type=float, default=0.,
                    help='average processint time for calculating score')
args = parser.parse_args()


class Evaluation(object):
    def __init__(self, args_):
        self.dataset_name = args_.dataset_name
        self.delay = args_.delay
        self.avg_pt = args.avg_pt
        self.score = 0.

        self.color = (0, 51, 255)
        self.thickness = 1
        self.predictions, self.gt_labels = [], []
        # license plate detection
        self.bbox_num_corrects = 0
        self.bbox_num_examples = 0
        self.bbox_accuracy = 0.
        self.bbox_threshold = 0.7
        # licnese plate recognition
        self.rec_num_corrects = 0
        self.rec_num_examples = 0
        self.rec_accuracy = 0
        # read prediciton csv file
        self._read_csv()

    def _read_csv(self):
        # read prediction result
        #with open(self.dataset_name+'.csv', 'r', newline='') as csvfile:
        #    reader = csv.reader(csvfile, delimiter=',')
        #    for row in reader:
        #        self.predictions.append(row)
        #        # print(row)
        
        delimiter = ','
        predictions = []
        reader = codecs.open(self.dataset_name + '.csv', 'r', encoding='utf-8')
        for line in reader:
            row = line.split(delimiter)
            self.predictions.append(row)

    def __call__(self):
        #window_name = 'Show'
        #cv2.namedWindow(window_name)
        #cv2.moveWindow(window_name, 0, 0)

        # read test image address
        filenames = []
        if self.dataset_name == 'cctv' or self.dataset_name == 'parking':
            for fold in os.listdir(self.dataset_name):
                for fname in os.listdir(os.path.join(self.dataset_name, fold)):
                    if fname.endswith('.jpg') or fname.endswith('.png'):
                        filenames.append(os.path.join(self.dataset_name, fold, fname))
                        # print(os.path.join(self.dataset_name, fold, fname))
            filenames = sorted(filenames)
        else:
            raise NotImplementedError

        for idx, filename in enumerate(filenames):
            #print(filename)

            img = cv2.pyrDown(cv2.imread(filename))  # read and resize image
            gt_labels, gt_boxes = self.read_data(filename)  # load GT

            # draw bounding box
            if gt_labels is not None:
                for sub_idx, label in enumerate(gt_labels):
                    # print(label, gt_boxes[sub_idx])
                    cv2.rectangle(img, (int(0.5 * gt_boxes[sub_idx][0]), int(0.5 * gt_boxes[sub_idx][1])),
                                  (int(0.5 * gt_boxes[sub_idx][2]), int(0.5 * gt_boxes[sub_idx][3])),
                                  self.color, self.thickness)

                    # compare between gt and prediction
                    self.compare(filename, label, gt_boxes[sub_idx])
            #cv2.imshow(window_name, img)  # show image

            #if cv2.waitKey(self.delay) & 0XFF == 27:
            #    sys.exit('Esc clicked!')

        self.write_analysis_csv()

    def write_analysis_csv(self):
        self.bbox_accuracy = (self.bbox_num_corrects/self.bbox_num_examples) * 100.
        self.rec_accuracy = (self.rec_num_corrects/self.rec_num_examples) * 100.
        self.score = self.bbox_accuracy + self.rec_accuracy + 0.1 * (100 - self.avg_pt)

        with open(self.dataset_name+'_analysis.csv', 'w', newline='') as f:
            csvwriter = csv.writer(f, delimiter=',')
            csvwriter.writerow(['num_bbox_examples', self.bbox_num_examples])
            csvwriter.writerow(['num_bbox_corrects', self.bbox_num_corrects])
            csvwriter.writerow(['bbox_accuracy', '{:.2f}'.format(self.bbox_accuracy)])
            csvwriter.writerow(['num_rec_examples', self.rec_num_examples])
            csvwriter.writerow(['num_rec_corrects', self.rec_num_corrects])
            csvwriter.writerow(['rec_accuracy', '{:.2f}'.format(self.rec_accuracy)])
            csvwriter.writerow(['avg_pt', '{:.2f}'.format(self.avg_pt)])
            csvwriter.writerow(['score', '{:.2f}'.format(self.score)])

    def compare(self, target_name, gt_label, gt_box):
        if self.dataset_name == 'cctv':
            gt_label = gt_label[:]  # ignore 'P#_' in GT
        elif self.dataset_name == 'parking':
            gt_label = gt_label
        else:
            raise NotImplementedError

        # count objects
        self.bbox_num_examples += 1
        if '?' not in gt_label:
            self.rec_num_examples += 1

        # self.predictions: filename, LP_number, x1, y1, x2, y2
        for cnt, result in enumerate(self.predictions):
            if target_name == result[0]:
                # bbox evaluation
                pre_box = [int(value) for value in result[2:]]  # string to int
                
                if self.bb_intersection_over_union(gt_box, pre_box) >= self.bbox_threshold:
                    self.bbox_num_corrects += 1

                # recognition evaluation
                pre_label = result[1]
                if ('?' not in gt_label) and (pre_label == gt_label):
                    print("True recognition")
                    self.rec_num_corrects += 1

    @staticmethod
    def bb_intersection_over_union(box_a, box_b):
        iou = 0.

        # check saving type of prediction is correct. We hope saving order is (x1, y1, x2, y2).
        # some trick way is (x2, y2, x1, y1), then the iou always bigger than threshold.
        if (box_b[0] >= box_b[2]) or (box_b[1] >= box_b[3]):
            print('Wrong saving order in csv file!')
            return iou

        # determine the (x, y)-coordinates of the intersection rectangle
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])

        # compute the area of intersection rectangle
        inter_area = (x_b - x_a + 1) * (y_b - y_a + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
        box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        # return the intersection over union value
        return iou

    def read_data(self, filename):
        print(filename)
        if self.dataset_name == 'cctv':
            if not os.path.isfile(filename[:-3] + 'xml'):
                return None, None
            else:
                
                labels, boxes = [], []
                node = parse(filename[:-3] + 'xml').getroot()

                elems = node.findall('object')
                for subelem in elems:
                    # read label
                    labels.append(subelem.find('name').text)
                    # read bounding boxes
                    box = []
                    for idx, item in enumerate(['xmin', 'ymin', 'xmax', 'ymax']):
                        box.append(int(subelem.find('bndbox').find(item).text))  # x1, y1, x2, y2
                   
                    boxes.append(box)

                return labels, boxes
        elif self.dataset_name == 'parking':
            if not os.path.isfile(filename[:-3] + 'txt'):
                return None, None
            else:
                labels, boxes = [], []
                with open(filename[:-3] + 'txt', 'r', encoding='UHC') as f:
                    
                    box = [int(data) for data in f.readline().split()]  # x1, y1, w, h
                    #box[2], box[3] = box[0] + box[2], box[1] + box[3]  # x1, y1, x2, y2
                    label = f.readline().split()

                    boxes.append(box)
                    labels.append(label[0])  # label is list, we wants to save string
                    
                

                return labels, boxes


def main():
    evaluator = Evaluation(args)
    evaluator()  # evaluate pred.csv with GT
    print('=' * 40)
    print('Number of bbox corrects: {}'.format(evaluator.bbox_num_corrects))
    print('Number of bbox examples: {}'.format(evaluator.bbox_num_examples))
    print('Detection accuracy: {:.2f}\n'.format(evaluator.bbox_accuracy))
    print('Number of recognition corrects: {}'.format(evaluator.rec_num_corrects))
    print('Number of recognition examples: {}'.format(evaluator.rec_num_examples))
    print('Recognition accuracy: {:.2f}\n'.format(evaluator.rec_accuracy))
    print('Average pt: {:.2f}\n'.format(args.avg_pt))
    print('Score: {:.2f}'.format(evaluator.score))
    print('=' * 40)


if __name__ == '__main__':
    main()
