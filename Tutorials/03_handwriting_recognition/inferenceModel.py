import cv2
import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
       
        if image.size == 0:
            raise ValueError("Input image is empty")

    
        image = cv2.resize(image, self.input_shape[:2][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text

if __name__ == "__main__":
    from mltu.configs import BaseModelConfigs

    
    custom_image_path = r"C:\Users\srita\Downloads\mltu-main\Tutorials\03_handwriting_recognition\b02-097-05-05.png"

    configs = BaseModelConfigs.load(r"C:\Users\srita\Downloads\mltu-main\Models\03_handwriting_recognition\202301111911\configs.yaml")

    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

  
    custom_image = cv2.imread(custom_image_path)

    
    if custom_image is None:
        raise ValueError(f"Failed to load the image from {custom_image_path}")

    
    prediction_text = model.predict(custom_image)

  
    print(f"Custom Image: {custom_image_path}, Prediction: {prediction_text}")

    
    custom_image = cv2.resize(custom_image, (custom_image.shape[1] * 4, custom_image.shape[0] * 4))
    cv2.imshow("Custom Image", custom_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
