import argparse
import cv2
import json
import numpy
from pathlib import Path
from tqdm import tqdm

from model import autoencoder_A
from model import autoencoder_B
from model import encoder, decoder_A, decoder_B

encoder  .load_weights( "models/encoder.h5"   )
decoder_A.load_weights( "models/decoder_A.h5" )
decoder_B.load_weights( "models/decoder_B.h5" )

def convert_one_image( autoencoder, image, mat ):
    size = 64
    face = cv2.warpAffine( image, mat * size, (size,size) )
    face = numpy.expand_dims( face, 0 )
    new_face = autoencoder.predict( face / 255.0 )[0]
    new_face = numpy.clip( new_face * 255, 0, 255 ).astype( image.dtype )
    new_image = numpy.copy( image )
    image_size = image.shape[1], image.shape[0]
    cv2.warpAffine( new_face, mat * size, image_size, new_image, cv2.WARP_INVERSE_MAP, cv2.BORDER_TRANSPARENT )
    return new_image

def main( args ):
    input_dir = Path( args.input_dir )
    assert input_dir.is_dir()

    alignments = input_dir / args.alignments
    with alignments.open() as f:
        alignments = json.load(f)

    output_dir = input_dir / args.output_dir
    output_dir.mkdir( parents=True, exist_ok=True )

    if args.direction == 'AtoB': autoencoder = autoencoder_B
    if args.direction == 'BtoA': autoencoder = autoencoder_A

    for image_file, face_file, mat in tqdm( alignments ):
        image = cv2.imread( str( image_file ) )
        face  = cv2.imread( str( face_file  ) )

        mat = numpy.array(mat).reshape(2,3)

        if image is None: 
            print("No image found")
            continue
        if face  is None: 
            print("No face found")
            continue

        new_image = convert_one_image( autoencoder, image, mat )

        output_file = output_dir / Path(image_file).name
        cv2.imwrite( str(output_file), new_image )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( "input_dir", type=str )
    parser.add_argument( "alignments", type=str, nargs='?', default='alignments.json' )
    parser.add_argument( "output_dir", type=str, nargs='?', default='merged' )
    parser.add_argument( "--direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
    main( parser.parse_args() )

