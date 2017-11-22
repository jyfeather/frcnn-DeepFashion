import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import shutil
import cv2
import pandas as pd

# ----------------- configuration -----------------
dir_deepfashion = '/export/home/yanjin/dataset/DeepFashion/CaAtPrBe'

# ----------------- UDF -----------------
def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def parse_category(s):
    ''' img/Sleek_Draped_Blouse/img_00000062.jpg                               3\n,
    '''
    [filestring, catstring] = s.split()
    filename = '_'.join(filestring.split('/')[1:])
    return [filename, catstring]

def parse_bbox(s):
    ''' img/Paisley_Print_Babydoll_Dress/img_00000054.jpg                      036 063 202 296\n
    '''
    [filestring, xmin, ymin, xmax, ymax] = s.split()
    filename = '_'.join(filestring.split('/')[1:])
    return [filename, xmin, ymin, xmax, ymax]

def mapping_label_type(label):
    ''' map from label (1 ... 50) to type (1,2,3)
    '''
    label = int(label)
    if label >= 37:
        return 'full_body' # full body 
    elif label >= 21:
        return 'lower_body' # lower body
    elif label >= 1:
        return 'upper_body' # upper body
    else: 
        return 'NA'

def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

# ----------------- prepare data -----------------
input_paths = [os.path.join(dir_deepfashion, s) for s in ['Anno', 'Eval', 'Img/img']]

for p in input_paths:
    if os.path.exists(p): print(p + ' exists.')

dir_Anno = os.path.join(dir_deepfashion, 'Anno') 
dir_Eval = os.path.join(dir_deepfashion, 'Eval') 
dir_Img = os.path.join(dir_deepfashion, 'Img/img') 

# read list_category_img.txt
txt_path = os.path.join(dir_Anno, 'list_category_img.txt')
if not os.path.exists(txt_path): print('cannot find: ', txt_path)
with open(txt_path, 'r') as f:
    list_category = f.readlines()
num_category = int(list_category[0].strip())
col_category = list_category[1].strip().split()
parsed_cat = map(parse_category, list_category[2:]) # first 2 rows are not data
df_category = pd.DataFrame(parsed_cat, columns=col_category)
assert len(df_category.index) == num_category, 'num of category instances mismatch!'

# read list_category_cloth.txt
cloth_path = os.path.join(dir_Anno, 'list_category_cloth.txt')
if not os.path.exists(cloth_path): print('cannot find: ', cloth_path)
with open(cloth_path, 'r') as f:
    list_cloth = f.readlines()
num_cloth = int(list_cloth[0].strip())
col_cloth = list_cloth[1].strip().split()
parsed_cloth = [x.split() for x in list_cloth[2:]]  # first 2 rows are not data
df_cloth = pd.DataFrame(parsed_cloth, columns=col_cloth)
assert len(df_cloth.index) == num_cloth, 'num of cloth instances mismatch!'

# read list_bbox.txt
bbox_path = os.path.join(dir_Anno, 'list_bbox.txt')
if not os.path.exists(bbox_path): print('cannot find: ', bbox_path)
with open(bbox_path, 'r') as f:
    list_bbox = f.readlines()
num_bbox = int(list_bbox[0].strip())
col_bbox = list_bbox[1].strip().split()
parsed_cat = map(parse_bbox, list_bbox[2:]) # first 2 rows are not data
df_bbox = pd.DataFrame(parsed_cat, columns=col_bbox)
assert len(df_bbox.index) == num_bbox, 'num of bbox instances mismatch!'

# read train/validation/test splitting file
partition_path = os.path.join(dir_Eval, 'list_eval_partition.txt')
if not os.path.exists(partition_path): print('cannot find: ', partition_path)
with open(partition_path , 'r') as f:
    list_partition = f.readlines()
num_partition = int(list_partition[0].strip())
col_partition = list_partition[1].strip().split()
parsed_split = map(parse_category, list_partition[2:]) # first 2 rows are not data
df_partition = pd.DataFrame(parsed_split, columns=col_partition)
assert len(df_partition.index) == num_partition, 'num of partition instances mismatch!'

# join bbox and category data
df_join = df_category.merge(df_bbox, on='image_name', how='inner')
assert len(df_join.index) == num_bbox, 'num of join instances mismatch the bbox!'
assert len(df_join.index) == num_category, 'num of join instances mismatch the category!'
dict_join = df_join.set_index('image_name').T.to_dict('list')

# prepare output directory
out_anno = os.path.join(dir_deepfashion, 'VOC/Annotations')
out_jpeg = os.path.join(dir_deepfashion, 'VOC/JPEGImages')
out_split = os.path.join(dir_deepfashion, 'VOC/ImageSets')
out_paths = [out_anno, out_split]

for p in out_paths:
    if not os.path.exists(p): 
        shutil.rmtree(p)
        os.makedirs(p)

# ----------------- convert -----------------
num_saved = 0
for x in os.walk(dir_Img):
    pre_cat = x[0].split('/')[-1]
    pre_dir = x[0]
    for y in x[2]:
        img_name = '_'.join([pre_cat, y])
        img_src = os.path.join(pre_dir, y)
        img_dst = os.path.join(out_jpeg, img_name)
        #shutil.copyfile(img_src, img_dst)
        if not img_name in dict_join.keys(): continue

        # ----------------- XML object -----------------
        xml_root = ET.Element("annotation")

        xml_sub_filename = ET.SubElement(xml_root, "filename")
        xml_sub_filename.text = img_name 

        xml_sub_folder = ET.SubElement(xml_root, "folder")
        xml_sub_folder.text = "DeepFashion"

        xml_sub_object = ET.SubElement(xml_root, "object")
        img_label = dict_join[img_name][0]
        img_type = mapping_label_type(img_label)
        ET.SubElement(xml_sub_object, 'label').text = img_label
        ET.SubElement(xml_sub_object, 'name').text = img_type
        xml_subsub_bndbox = ET.SubElement(xml_sub_object, "bndbox")
        ET.SubElement(xml_subsub_bndbox , 'xmax').text = dict_join[img_name][3]
        ET.SubElement(xml_subsub_bndbox , 'xmin').text = dict_join[img_name][1]
        ET.SubElement(xml_subsub_bndbox , 'ymax').text = dict_join[img_name][4]
        ET.SubElement(xml_subsub_bndbox , 'ymin').text = dict_join[img_name][2]
        ET.SubElement(xml_sub_object, 'difficult').text = str(0) 

        xml_sub_seg = ET.SubElement(xml_root, "segmented")
        xml_sub_seg.text = str(0) # since we do not segment images

        xml_sub_size = ET.SubElement(xml_root, "size")
        img = cv2.imread(img_src)
        [h, w, d] = img.shape
        ET.SubElement(xml_sub_size, "depth").text = str(d)
        ET.SubElement(xml_sub_size, "height").text = str(h)
        ET.SubElement(xml_sub_size, "width").text = str(w)

        xml_sub_source = ET.SubElement(xml_root, "source")
        xml_sub_source.text = "TBD" 

        #print(prettify(xml_root))
        tree = ET.ElementTree(xml_root)
	indent(xml_root)
        tree.write(os.path.join(out_anno, img_name.replace('.jpg', '.xml')), encoding='utf-8')
        num_saved += 1

    print(pre_cat + ' is done...')

# ----------------- split train and test -----------------
out_test = os.path.join(out_split, 'test.txt')
df_partition.loc[df_partition['evaluation_status']=='test'].image_name.to_csv(out_test, header=False, index=False)
out_trainval = os.path.join(out_split, 'trainval.txt')
df_partition.loc[df_partition['evaluation_status'].isin(['train', 'val'])].image_name.to_csv(out_trainval, header=False, index=False)

# ----------------- sanity check: xml == jpg -----------------
print(num_saved)
print('Congrats!')
