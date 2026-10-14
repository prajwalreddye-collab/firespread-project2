import rasterio

image_path = "dataset/images/2017-04-23_VIIRS_Day.tif"
mask_path = "dataset/masks/2017-04-23_FirePred.tif"

print("SATELLITE IMAGE")
with rasterio.open(image_path) as src:
    print("Width:", src.width)
    print("Height:", src.height)
    print("Number of bands:", src.count)
    print("Data type:", src.dtypes)
    print("CRS:", src.crs)

print("\nFIREPRED FILE")
with rasterio.open(mask_path) as src:
    print("Width:", src.width)
    print("Height:", src.height)
    print("Number of bands:", src.count)
    print("Data type:", src.dtypes)
    print("CRS:", src.crs)