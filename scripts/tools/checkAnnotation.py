import os

image_folder = "/batch_00/images"
label_folder = "/batch_00/labels"
image_ext = ".jpg"
label_ext = ".txt"

image_names = {os.path.splitext(f)[0] for f in os.listdir(image_folder) if f.endswith(image_ext)}
label_names = {os.path.splitext(f)[0] for f in os.listdir(label_folder) if f.endswith(label_ext)}

only_images = image_names - label_names
only_labels = label_names - image_names
matched = image_names & label_names

print(f"🔎 Matches found : {len(matched)}")
print(f"❌ Images without .txt : {len(only_images)}")
print(f"❌ .txt without image : {len(only_labels)}")

if only_images:
    print("\nImages without .txt :")
    for name in sorted(only_images):
        print(f"  {name}{image_ext}")

if only_labels:
    print("\n.txt without image :")
    for name in sorted(only_labels):
        print(f"  {name}{label_ext}")
