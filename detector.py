
from pathlib import Path

import face_recognition
import pickle


DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")


def encode_known_faces(
    model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    names = []
    encodings = []


    for filepath in Path("training").glob("*/*"):  #boucle pour aller dans chaque dossiers/images de training
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)


        # Detecte les visages sur les images et recupere leur encodage (librairie face_recognistion)
        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

        # j'utilise pickle pour save l'encodage sur le disque
        name_encodings = {"names": names, "encodings": encodings}
        with encodings_location.open(mode="wb") as f:
            pickle.dump(name_encodings, f)

encode_known_faces()