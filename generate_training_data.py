import face_recognition
import cv2
from pathlib import Path
 
from utils import get_image_paths

image_filenames = get_image_paths( 'raw_data/obama' )

output_dir = Path( 'training_data/obama' )
output_dir.mkdir( parents=True, exist_ok=True )
 
for filename in image_filenames:
    image = cv2.imread(filename)
    top, right, bottom, left = face_recognition.face_locations(image)
    cropped_image = image[top:bottom, left:right]
    output_file = output_dir / Path(filename).name
    cv2.imwrite( str(output_file), cropped_image )