"""script to convert stain quantification output .csv into tiatoobox annotation store format"""

from tiatoolbox.annotation.storage import SQLiteStore, Annotation
import pandas as pd
from shapely.geometry import Polygon
import argparse

def df2store(df: pd.DataFrame, store_path: str):
    store = SQLiteStore()
    anns = []
    for _, row in df.iterrows():
        geom = Polygon.from_bounds(row["xmin"], row["ymin"], row["xmax"], row["ymax"])
        props = {k: v for k, v in row.items() if k not in ["xmin", "ymin", "xmax", "ymax"]}
        anns.append(Annotation(geom, props))
    store.append_many(anns)
    store.dump(store_path)
    return store

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str, help="path to the csv file")
    parser.add_argument("store_path", type=str, help="path to the store file")
    args = parser.parse_args()
    df = pd.read_csv(args.csv_path)
    store = df2store(df, args.store_path)




