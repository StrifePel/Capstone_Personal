import tensorflow as tf
import os
import numpy as np

def inspect_tfrecord(file_path):
    """Inspect the content and structure of a TFRecord file."""
    print(f"\nInspecting TFRecord file: {file_path}")
    
    try:
        # Try to read the raw records first
        dataset = tf.data.TFRecordDataset(file_path)
        
        # Get the first record
        for raw_record in dataset.take(1):
            print("\nRaw record found. Attempting to parse...")
            
            # Try parsing without a feature description first
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            
            # Print the available features
            print("\nAvailable features in the TFRecord:")
            features = example.features.feature
            feature_info = {}
            for key in features.keys():
                feature = features[key]
                # Check which field is populated
                if feature.HasField('float_list'):
                    value_type = 'float_list'
                    dims = len(feature.float_list.value)
                    feature_info[key] = ('float_list', dims)
                elif feature.HasField('int64_list'):
                    value_type = 'int64_list'
                    dims = len(feature.int64_list.value)
                    feature_info[key] = ('int64_list', dims)
                elif feature.HasField('bytes_list'):
                    value_type = 'bytes_list'
                    dims = len(feature.bytes_list.value)
                    feature_info[key] = ('bytes_list', dims)
                else:
                    value_type = 'unknown'
                    dims = 0
                    feature_info[key] = ('unknown', 0)
                print(f"- {key}: {value_type} with {dims} elements")
            
            return feature_info
            
    except Exception as e:
        print(f"Error inspecting TFRecord: {str(e)}")
        return None

def create_feature_description(feature_info):
    """Create a feature description based on the actual data structure."""
    feature_description = {}
    for key, (value_type, dims) in feature_info.items():
        if value_type == 'float_list':
            # Assuming the data is a square image (64x64)
            feature_description[key] = tf.io.FixedLenFeature([dims], tf.float32)
        elif value_type == 'int64_list':
            feature_description[key] = tf.io.FixedLenFeature([dims], tf.int64)
        elif value_type == 'bytes_list':
            feature_description[key] = tf.io.FixedLenFeature([dims], tf.string)
    return feature_description

def test_parse_tfrecord(file_path, feature_description):
    """Test parsing a TFRecord with the given feature description."""
    print("\nTesting TFRecord parsing with feature description:")
    
    try:
        dataset = tf.data.TFRecordDataset(file_path)
        parsed_dataset = dataset.map(lambda x: tf.io.parse_single_example(x, feature_description))
        
        # Try to read one example
        for parsed_record in parsed_dataset.take(1):
            print("\nSuccessfully parsed record!")
            print("\nFeature shapes:")
            for key, tensor in parsed_record.items():
                print(f"- {key}: shape {tensor.shape}, dtype {tensor.dtype}")
                
                # Print some basic statistics
                values = tensor.numpy()
                print(f"  Min: {np.min(values):.4f}, Max: {np.max(values):.4f}, Mean: {np.mean(values):.4f}")
            return True
            
    except Exception as e:
        print(f"Error parsing TFRecord: {str(e)}")
        return False

if __name__ == "__main__":
    # Test with one TFRecord file
    data_dir = "C:/data/train"
    tfrecord_files = [f for f in os.listdir(data_dir) if f.endswith('.tfrecord')]
    
    if not tfrecord_files:
        print(f"No TFRecord files found in {data_dir}")
        exit(1)
        
    test_file = os.path.join(data_dir, tfrecord_files[0])
    print(f"Testing with file: {test_file}")
    
    # Inspect the TFRecord structure
    feature_info = inspect_tfrecord(test_file)
    
    if feature_info:
        # Create feature description based on inspection
        feature_description = create_feature_description(feature_info)
        print("\nCreated feature description:")
        for key, feature in feature_description.items():
            print(f"- {key}: {feature}")
            
        # Test parsing
        test_parse_tfrecord(test_file, feature_description)