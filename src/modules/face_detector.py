from typing import Any, List, Tuple
import dlib


class FaceDetector(object):
    def __init__(self) -> None:
        self.detector = dlib.get_frontal_face_detector()

    def __call__(self, f) -> List[List[Tuple[int, int]]]:
        """_summary_

        Args:
            f (_type_): _description_

        Returns:
            List[List[Tuple[int, int]]]: List of coordinate pairs for bboxes (x1, y1), (x2, y2)
        """
        if isinstance(f, str):
            img = dlib.load_rgb_image(f)
        else:
            img = f
        dets = self.detector(img, 1)
        return dets


if __name__ == "__main__":
    from PIL import Image, ImageDraw
    f = "/mnt/e/WORK_DL/datasets/landmarks_task/300W/test/261068_2.jpg"
    face_detector = FaceDetector()
    dets = face_detector(f)
    image = Image.open(f)
    image_d = ImageDraw.Draw(image)

    for box in dets:
        t = [box.left(), box.top(), box.right(), box.bottom()]
        image_d.rectangle(t, outline=(255, 0, 0), width=2)

    image.save("imh.png")