# xml and yolo is opencv definition
# rotate yolo label: (cls, cx, cy, w, h, angle) angle ∈ [0,90)
import os.path
import xml.etree.ElementTree as ET
from os import getcwd


classes = ['car', 'van', 'bus', 'traffic_sign', 'person']


def xml2yolor(label_xml, save_path):
    f = open(label_xml)
    label_name = label_xml.split('\\')[-1][:-3]
    out_file = open(os.path.join(save_path, label_name + 'txt'), 'w')
    xml_text = f.read()
    root = ET.fromstring(xml_text)
    f.close()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            print(cls)
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('robndbox')
        rb = (float(xmlbox.find('cx').text)/w, float(xmlbox.find('cy').text)/h,
             float(xmlbox.find('w').text)/w, float(xmlbox.find('h').text)/h,
             float(xmlbox.find('angle').text)*180/3.1416)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in rb]) + '\n')


wd = getcwd()

if __name__ == '__main__':
    xml_file = r'C:\Users\Administrator\Desktop\generate_data\label_xml'
    out_yolo_file = r'C:\Users\Administrator\Desktop\generate_data\yolo_rotate'
    if not os.path.exists(out_yolo_file):
        os.makedirs(out_yolo_file)
    xml_file_list = os.listdir(xml_file)
    xml_file_list_path = [os.path.join(xml_file, xml) for xml in xml_file_list]
    for xml in xml_file_list_path:
        xml2yolor(xml, out_yolo_file)