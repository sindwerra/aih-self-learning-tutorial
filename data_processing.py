# For preparing
import xml.etree.cElementTree as ET
import glob
import os
import json
import shutil

from PIL import Image

dataset_path = "<PLEASE SPECIFY ME>" # If you want try it on your local machine, set this variable properly


def data_preview():
    """
    This function gives an overview of raw data from facemask dataset
    """
    with open(f'{dataset_path}/annotations/maksssksksss10.xml') as f:
        contents = f.read()
        print(contents)

    Image.open(f"{dataset_path}/images/maksssksksss10.png")


def xml_to_yolo_bbox(bbox, w, h):
    """
    YOLO model series has its own special format for bbox annotations, 
    this function transfer PASCAL VOC format annotation to YOLO format.
    """
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h

    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h

    return [x_center, y_center, width, height]


def yolo_to_xml_bbox(bbox, w, h):
    """
    This is the inverse action of previouse function.
    """
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2

    xmin = int((bbox[0] * w) - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len)

    return [xmin, ymin, xmax, ymax]


def transferring_xml_labels_to_txt_labels():
    """
    This function transfer xml annotation labels to 
    text annotation labels for training.
    """
    classes = []

    input_dir = f"{dataset_path}/annotations"
    output_dir = f"{dataset_path}/labels"
    image_dir = f"{dataset_path}/images"

    files = glob.glob(os.path.join(input_dir, "*.xml"))
    for fil in files:
        basename = os.path.basename(fil)
        filename = os.path.splitext(basename)[0]
        if not os.path.exists(os.path.join(image_dir, f"{filename}.png")):
            print(f"{filename} image does not exist!")
            continue

        result = []

        # Read xml file annotations.
        tree = ET.parse(fil)
        root = tree.getroot()
        width = int(root.find("size").find("width").text)
        height = int(root.find("size").find("height").text)

        for obj in root.findall("object"):
            label = obj.find("name").text

            # The original dataset has three classed, we will train a model just classify mask-wearing status
            # So just replace all mask_weared_incorrect class to with_mask class.
            if label == "mask_weared_incorrect":
                label = "with_mask"

            if label not in classes:
                classes.append(label)
            index = classes.index(label)
            pil_bbox = [int(x.text) for x in obj.find("bndbox")]
            yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)

            # Set class label and bbox info pairs.
            bbox_string = " ".join([str(x) for x in yolo_bbox])
            result.append(f"{index} {bbox_string}")

        if result:
            # Write YOLO format annotation text file
            with open(os.path.join(output_dir, f"{filename}.txt"), "w", encoding = "utf-8") as f:
                f.write("\n".join(result))

    # Write class label files as reference
    with open(f"{dataset_path}/labels/classes.txt", "w", encoding = "utf-8") as f:
        f.write(json.dumps(classes))

    # Print two classes labels as reference
    with open(f'{dataset_path}/labels/classes.txt') as f:
        contents = f.read()
        print(contents)


def general_view_of_dataset():
    """
    This function gives an overview of dataset size which helps to split data
    """
    material = []
    for i in os.listdir(f"{dataset_path}/images"):
        srt = i[:-4]
        material.append(srt)
    print(f"The whole dataset has {len(material)} images")
    return material


def preparing_yolo_format_bbox_data(main_txt_file, main_img_file, material, train_size, val_size):
    """
    This function organizing the structure of dataset to adapt YOLO model for training
    """
    for i in range(0,train_size):

        source_txt = main_txt_file + "/" + material[i] + ".txt"
        source_img = main_img_file + "/" + material[i] + ".png"

        train_destination_txt = f"{dataset_path}/train/labels" + "/" + material[i] + ".txt"
        train_destination_png = f"{dataset_path}/train/images" + "/" + material[i] + ".png"

        shutil.copy(source_txt, train_destination_txt)
        shutil.copy(source_img, train_destination_png)

    for l in range(train_size , train_size + val_size):

        source_txt = main_txt_file + "/" + material[l] + ".txt"
        source_img = main_img_file + "/" + material[l] + ".png"

        val_destination_txt = f"{dataset_path}/val/labels" + "/" + material[l] + ".txt"
        val_destination_png = f"{dataset_path}/val/images" + "/" + material[l] + ".png"

        shutil.copy(source_txt, val_destination_txt)
        shutil.copy(source_img, val_destination_png)


def making_yolo_configuration_file():
    """
    YOLO model need to a configuration file to specify dataset path for model training.
    """
    yaml_text = """
    train: <USE dataset_path HERE>/train/images
    val: <USE dataset_path HERE>/val/images
    nc: 2
    names: ["without_mask", "with_mask"]"""

    with open(f"{dataset_path}/data.yaml", 'w') as file:
        file.write(yaml_text)


if __name__ == "__main__":
    data_preview()
    os.makedirs(f"{dataset_path}/labels")
    transferring_xml_labels_to_txt_labels()
    os.mkdir(f"{dataset_path}/train")
    os.mkdir(f"{dataset_path}/val")
    os.mkdir(f"{dataset_path}/train/images")
    os.mkdir(f"{dataset_path}/train/labels")
    os.mkdir(f"{dataset_path}/val/images")
    os.mkdir(f"{dataset_path}/val/labels")
    materail = general_view_of_dataset()
    preparing_yolo_format_bbox_data(f"{dataset_path}/labels", f"{dataset_path}/images", materail, 600, 253)
    making_yolo_configuration_file()
