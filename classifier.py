import tensorflow as tf

IMAGE_SIZE = (537, 618)
def preprocess_image(image):
    image_array = tf.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dins(image_array, 0)
    return image_array

def load_and_preprocess_image(path):
    image = tf.preprocessing.image.load_img(
        path, IMAGE_SIZE
    )

    return preprocess_image(image)

def classify(model, image_path):
    preprocessed_image = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)
    score = predictions[0][0]
    return score
