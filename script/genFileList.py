import os
import hashlib
import json
import shutil

# Directory to process
url = "https://cvat-ota.cocogoat.cn/download/cvautotrack/cvat_rc_beta"
directory = "../cvat_rc_beta"
output_file = os.path.join(directory, "dependents.json")

# Function to calculate MD5 hash of a file
def calculate_md5(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest().upper()

# Function to calculate sumMD5
def calculate_sum_md5(file_hashes):
    sum_md5 = 0
    for file_hash in file_hashes:
        sum_md5 ^= int(file_hash, 16)
    return f"{sum_md5:032X}"

# Function to increment version number
def increment_version(version):
    major, minor, patch = map(int, version.split('.'))
    patch += 1
    return f"{major}.{minor}.{patch}"

# Main logic
def main():
    filelist = []
    file_hashes = []
    existing_files = set()
    current_files = set()

    # Traverse the directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file == "dependents.json":
                continue

            filepath = os.path.join(root, file)
            relative_path = os.path.relpath(filepath, directory)
            relative_path = relative_path.replace('\\', '/')
            current_files.add(relative_path)

            file_md5 = calculate_md5(filepath)
            file_hashes.append(file_md5)

            filelist.append({
                "filename": relative_path,
                "url": f"{url}/{relative_path}",
                "md5": file_md5
            })

    # Calculate sumMD5
    sum_md5 = calculate_sum_md5(file_hashes)

    # Check if dependents.json exists
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            data = json.load(f)

        existing_files = {item["filename"] for item in data.get("filelist", [])}

        if data.get("sumMD5") != sum_md5:
            data["version"] = increment_version(data["version"])
            data["sumMD5"] = sum_md5
            data["filelist"] = filelist

            with open(output_file, "w") as f:
                json.dump(data, f, indent=4)
    else:
        # Create a new dependents.json
        data = {
            "version": "1.0.0",
            "sumMD5": sum_md5,
            "filelist": filelist
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)

if __name__ == "__main__":
    main()