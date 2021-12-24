from io_utils import exif_io


def generate_plant_density_heatmap(dataset, predictions):
    for img in dataset.imgs:
        metadata = exif_io.get_exif_metadata(img.img_path)
        print("metadata: \n{}".format(metadata))

        latitude = metadata["EXIF:GPSLatitude"]
        longitude = metadata["EXIF:GPSLongitude"]

        vals = predictions[img.img_name]["pred_class_counts"]



