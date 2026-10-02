"""Download the original AG News CSV files (Zhang et al., 2015) into data/.

Run:  python scripts/download_data.py
"""
import os
import urllib.request

BASE = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/"
FILES = ["train.csv", "test.csv", "classes.txt"]


def main(out_dir="data"):
    os.makedirs(out_dir, exist_ok=True)
    for name in FILES:
        dest = os.path.join(out_dir, name)
        if os.path.exists(dest):
            print(f"{name}: already present")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(BASE + name, dest)
    print("done. train.csv = 120,000 rows, test.csv = 7,600 rows")


if __name__ == "__main__":
    main()
