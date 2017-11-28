import os
import sys
import cv2

def find_next_free_file(prefix, suffix, directory):
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    i = 0
    while True:
        while True:
            path=os.path.join(directory,"%s-%d.%s" % (prefix, i, suffix))
            if not os.path.isfile(path):
                break
            i += 1
        # Create the file to avoid a race condition.
        # Will give an error if the file already exists.
        try:
            f = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(f)
        except FileExistsError as e:
            # Trying to create a file that already exists.
            # Try a new file name
            continue
        break
    return path, i

def extract_from_image(path, output_dir):
    try:
        faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
            #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        print("Found %d faces!" % len(faces))

        # Save faces
        for (x, y, w, h) in faces:
            p, i = find_next_free_file("face", "png", output_dir)
            cv2.imwrite(p, image[y:(y+h), x:(x+h)])

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    except Exception:
        print("Unable to process file. Skipping.")

def extract_from_dir(path, output_dir):
    all_files = [os.path.join(path,f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for fn in all_files:
        extract_from_image(fn, output_dir)

if __name__=="__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    if os.path.isfile(input_path):
        extract_from_image(input_path, output_path)
    else:
        extract_from_dir(input_path, output_path)
