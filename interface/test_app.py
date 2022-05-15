import unittest
import os

from app import *
from inference import *

class FlaskTestCase(unittest.TestCase):

    def test_app(self):
        # test server loads correctly
        response = app.test_client(self)
        response = response.get('/', content_type='html/text')
        self.assertEqual(response.status_code, 200)
    
    def test_get_single_prediction(self):
        # test with a valid image
        image_path = f'{BASE_DIR}/interface/test_images/000000.jpeg'
        image = transform_image(image_path)
        class_name, class_id = get_prediction(image_bytes=image, model=MODEL)
        self.assertEqual((class_name, class_id), ('CXR', 3))

    def test_get_list_predictions(self):
        # test with a lists of valid images
        df = {'filename':[],'class_name': [], 'class_id': []}
        for file in os.listdir(f'{BASE_DIR}/interface/test_images/'):
            image_path = f'{BASE_DIR}/interface/test_images/{file}'
            image = transform_image(image_path)
            class_name, class_id = get_prediction(image_bytes=image, model=MODEL)
            df['filename'].append(file)
            df['class_name'].append(class_name)
            df['class_id'].append(class_id)

        self.assertEqual((df['class_name'][2], df['class_id'][2]), ('Hand', 4))

if __name__ == '__main__':
    unittest.main()