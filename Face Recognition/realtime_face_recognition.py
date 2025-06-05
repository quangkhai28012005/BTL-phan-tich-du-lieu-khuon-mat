import sys
import subprocess

def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_and_import('cv2')
install_and_import('face_recognition')

import cv2
import face_recognition

# Capture video from default camera
webcam_video_stream = cv2.VideoCapture(0)

def safe_face_encoding(image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if len(encodings) == 0:
        print(f"Warning: No face found in {image_path}")
        return None
    return encodings[0]

face_4_encodings = safe_face_encoding('images/obama.jpg')
face_4_name = 'obama'

face_3_encodings = safe_face_encoding('images/khai.png')
face_3_name = 'khai'

face_5_encodings = safe_face_encoding('images/hien.png')
face_5_name = 'hien'

face_7_encodings = safe_face_encoding('images/khue.png')
face_7_name = 'khue'

face_9_encodings = safe_face_encoding('images/hien.png')
face_9_name = 'hien'

face_10_encodings = safe_face_encoding('images/huy.png')
face_10_name = 'huy'

face_13_encodings = safe_face_encoding('images/linh.png')
face_13_name = 'linh'

# Save encodings and corresponding labels to separate arrays in same order
known_face_encodings = [
    enc for enc in [
        face_3_encodings, face_4_encodings, face_5_encodings,
        face_7_encodings, face_9_encodings, face_10_encodings, face_13_encodings
    ] if enc is not None
]
known_face_names = [
    name for enc, name in zip(
        [face_3_encodings, face_4_encodings, face_5_encodings,
         face_7_encodings, face_9_encodings, face_10_encodings, face_13_encodings],
        [face_3_name, face_4_name, face_5_name, face_7_name, face_9_name, face_10_name, face_13_name]
    ) if enc is not None
]

# Initialize arrays for face locations, encodings, and names
all_face_locations = []
all_face_encodings = []
all_face_names = []

# Loop through each video frame until user exits
while True:
    ret, current_frame = webcam_video_stream.read()

    # Lets use a smaller version (0.25x) of the image for faster processing
    scale_factor = 4
    current_frame_small = cv2.resize(
        current_frame, (0, 0), fx=1/scale_factor, fy=1/scale_factor)

    # Find total number of faces, encodings, set names to empty
    all_face_locations = face_recognition.face_locations(
        current_frame_small, number_of_times_to_upsample=2, model='hog')
    
    all_face_encodings = face_recognition.face_encodings(current_frame_small, all_face_locations)
    
    #all_face_names = []
    
    # Iterate through each face location and encoding in our test image
    for current_face_location, current_face_encoding in zip(all_face_locations, all_face_encodings):
        # Splitting up tuple of face location
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        
        # Correct positions based on scale factor
        top_pos *= scale_factor
        right_pos *= scale_factor
        bottom_pos *= scale_factor
        left_pos *= scale_factor
        
        # Now we'll slice our image array to isolate the faces
        current_face_image = current_frame[top_pos: bottom_pos,
                                                left_pos:right_pos]

        # Compare to known faces to check for matches
        all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)

        # Initialize name string as unknown face
        name_of_person = 'Unknown'

        # Check if all_matches isn't empty
        # If yes get the index number corresponding to the face in the first index
        if True in all_matches:
            first_match_index = all_matches.index(True)
            name_of_person = known_face_names[first_match_index]

        # Draw rectangle around face
        cv2.rectangle(current_frame, (left_pos, top_pos),
                      (right_pos, bottom_pos), (255, 255, 255), 2)

        # Write corresponding name below face
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, name_of_person, (left_pos,
                    bottom_pos + 20), font, 0.5, (255, 255, 255), 1)

        # Display image with rectangle and text
        cv2.imshow('Identified Faces', current_frame)
        
    # Press 'enter' key to exit loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam_video_stream.release()
cv2.destroyAllWindows()