import tifffile
import numpy as np
import os

folder = "dataset/images"

dates = [
    "2017-04-23",
    "2017-04-24",
    "2017-04-25",
    "2017-04-26",
    "2017-04-27"
]

for date in dates:
    file_path = os.path.join(
        folder, f"{date}_VIIRS_Day.tif"
    )

    image = tifffile.imread(file_path)

    af = image[:, :, 6]
    valid = af[np.isfinite(af)]

    print("\nDate:", date)
    print("Active Fire valid pixels:", len(valid))

    if len(valid) > 0:
        print("Minimum:", valid.min())
        print("Maximum:", valid.max())
        print("Unique values:", np.unique(valid))
    else:
        print("No valid pixels found.")