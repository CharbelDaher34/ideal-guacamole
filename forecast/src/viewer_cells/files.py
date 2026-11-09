# Each Parquet file of the current version and its size (no file is read here).
files = pd.DataFrame({"file": paths, "size_mb": [round(folder.get_path_details(p).get("size", 0) / 1e6, 2)
                                                 for p in paths]})
print(f"{len(files)} files, {files['size_mb'].sum():,.1f} MB in all")
display(files)
