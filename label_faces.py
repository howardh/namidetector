import os
import sys
import csv
import cv2

# Open csv
# For every face, check if a record exists for it in the csv
# If there's no record, display the image and prompt the user to label (nami, not nami)

def load_csv(csv_file_name):
    print("Loading CSV")
    records = dict()
    with open(csv_file_name, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            face_file_name = row[0]
            label = row[1]
            records[face_file_name] = label
    return records

def save_csv(csv_file_name, dictionary):
    print("Saving CSV")
    with open(csv_file_name, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for k,v in dictionary.items():
            writer.writerow([k,v])

def update_labels(faces_dir, records):
    all_files = [f for f in os.listdir(faces_dir) if os.path.isfile(os.path.join(faces_dir, f))]
    for fn in all_files:
        if fn in records:
            continue
        img = cv2.imread(os.path.join(faces_dir,fn))
        cv2.imshow("Nami?", img)
        while True:
            x = cv2.waitKey(0)
            if x == 121:
                print("Is Nami")
                records[fn] = True
                break
            elif x == 110:
                print("Not Nami")
                records[fn] = False
                break
    return records

if __name__ == "__main__":
    faces_dir = sys.argv[1]
    csv_file_name = sys.argv[2]

    records = load_csv(csv_file_name)
    try:
        update_labels(faces_dir, records)
    except KeyboardInterrupt:
        print("Interrupted by user")
    save_csv(csv_file_name, records)
