import tensorflow as tf
import sys

def converter(path_model_h5, path_model_name):

    model = tf.keras.models.load_model(path_model_h5)
    
    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.      
    ]
    
    tflite_model = converter.convert()

    # Save the model.
    with open(f'{path_model_name}.tflite', 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    args = sys.argv
    if len(args) == 3: #args[0] is file name
        converter(args[1], args[2])
    else :
        print('Wrong arguments')
