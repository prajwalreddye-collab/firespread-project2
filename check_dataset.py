import os

folders = [
    "dataset/images",
    "dataset/masks"
]

for folder in folders:
    print(f"\n--- {folder} ---")

    if os.path.exists(folder):
        files = sorted(os.listdir(folder))

        print("Number of files:", len(files))

        for name in files[:10]:
            print(name)
    else:
        print("Folder not found")