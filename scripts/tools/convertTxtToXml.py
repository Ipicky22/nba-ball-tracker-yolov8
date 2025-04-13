import os
import xml.etree.ElementTree as ET

# === Paramètres à adapter ===
input_folder = "/batch_00/labels"
output_folder = "/batch_00/xml"
image_width = 1280
image_height = 720
label_name = "ball"
image_ext = ".jpg"

os.makedirs(output_folder, exist_ok=True)

def yolo_to_cvat_box(yolo_line):
    _, x_center, y_center, width, height = map(float, yolo_line.strip().split())
    xtl = (x_center - width / 2) * image_width
    ytl = (y_center - height / 2) * image_height
    xbr = (x_center + width / 2) * image_width
    ybr = (y_center + height / 2) * image_height
    return xtl, ytl, xbr, ybr

def create_xml(filename, boxes):
    annotations = ET.Element("annotations")
    ET.SubElement(annotations, "version").text = "1.1"

    # Meta block
    meta = ET.SubElement(annotations, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = "ball_preannot"
    labels = ET.SubElement(task, "labels")
    label = ET.SubElement(labels, "label")
    ET.SubElement(label, "name").text = label_name
    ET.SubElement(label, "color").text = "#ff0000"
    ET.SubElement(label, "attributes")

    image_tag = ET.SubElement(annotations, "image", {
        "id": "0",
        "name": filename + image_ext,
        "width": str(image_width),
        "height": str(image_height)
    })

    for xtl, ytl, xbr, ybr in boxes:
        ET.SubElement(image_tag, "box", {
            "label": label_name,
            "xtl": f"{xtl:.2f}",
            "ytl": f"{ytl:.2f}",
            "xbr": f"{xbr:.2f}",
            "ybr": f"{ybr:.2f}",
            "occluded": "0",
            "source": "manual"
        })

    tree = ET.ElementTree(annotations)
    output_path = os.path.join(output_folder, filename + ".xml")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

for txt_file in os.listdir(input_folder):
    if txt_file.endswith(".txt"):
        file_path = os.path.join(input_folder, txt_file)
        with open(file_path, "r") as f:
            lines = f.readlines()
            boxes = [yolo_to_cvat_box(line) for line in lines]

        base_name = os.path.splitext(txt_file)[0]
        create_xml(base_name, boxes)

print("Conversion completed.")
