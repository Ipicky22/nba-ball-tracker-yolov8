import os
import xml.etree.ElementTree as ET

input_folder = "/batch_00/xml" 
output_file = "/batch_00/merged_cvat.xml"

annotations = ET.Element("annotations")
ET.SubElement(annotations, "version").text = "1.1"

meta = ET.SubElement(annotations, "meta")
task = ET.SubElement(meta, "task")
ET.SubElement(task, "name").text = "ball_preannot"
labels = ET.SubElement(task, "labels")
label = ET.SubElement(labels, "label")
ET.SubElement(label, "name").text = "ball"
ET.SubElement(label, "color").text = "#ff0000"
ET.SubElement(label, "attributes")

image_id_counter = 0 

for xml_file in sorted(os.listdir(input_folder)):
    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(input_folder, xml_file))
    root = tree.getroot()

    for image in root.findall("image"):
        image.attrib["id"] = str(image_id_counter)
        image_id_counter += 1
        annotations.append(image)

tree = ET.ElementTree(annotations)
tree.write(output_file, encoding="utf-8", xml_declaration=True)

print(f"Merge complete.")
