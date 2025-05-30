import argparse
import ast
import re
import pandas as pd

PSNR_KEY = "psnr/adjacent test camera"
SCENE_REMAP = {"lego_gen12": "lego"}
SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]

NAME_RGX = re.compile(
    r"^(?P<scene>[^_]+)_"              # chair, drums …
    r"(?P<views>\d+)views_"            # num views
    r"(?P<method>[^_]+)_"              # singleview | multiplexing
    r"resolution(?P<resolution>\d+)_"  # resolution800 → 800
    r"dls(?P<dls>\d+)_"                # dls14 → 14
    r"tv(?P<tv>[0-9.]+)_"              # tv1.0 → 1.0
    r"unseen(?P<tv_unseen>[0-9.]+)$"   # unseen0.1 → 0.1
)

def explode_runs(raw: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in raw.iterrows():
        try:
            summary = ast.literal_eval(row["summary"])
        except Exception:
            print(f"Skipping row with invalid summary: {row['summary']}")
            continue
        psnr = summary.get(PSNR_KEY, None)
        if psnr is None: print(f"No PSNR found in {row['name']}"); continue
        m = NAME_RGX.match(row["name"])
        if m is None: print(f"Regex not matched in {row['name']}"); continue

        g = m.groupdict()
        scene = SCENE_REMAP.get(g["scene"], g["scene"])

        records.append(
            dict(
                resolution=int(g["resolution"]),
                method=g["method"],
                views=int(g["views"]),
                dls=int(g["dls"]),
                tv=float(g["tv"]),
                tv_unseen=float(g["tv_unseen"]),
                scene=scene,
                psnr=psnr,
            )
        )
    return pd.DataFrame.from_records(records)

def consolidate(df: pd.DataFrame) -> pd.DataFrame:
    wide = (
        df
        .pivot_table(
            index=["resolution", "method", "views", "dls", "tv", "tv_unseen"],
            columns="scene",
            values="psnr",
            aggfunc="mean",
        )
        .reset_index()
    )
    wide = wide.rename(
        columns={s: f"{s}_psnr" for s in wide.columns if s not in ["method", "views", "resolution", "dls", "tv", "tv_unseen"]}
    )
    scene_cls = [c for c in wide.columns if c.endswith("_psnr")]
    wide["avg_psnr"] = wide[scene_cls].mean(axis=1)
    wide["std_psnr"] = wide[scene_cls].std(axis=1)
    wide[scene_cls + ["avg_psnr", "std_psnr"]] = wide[scene_cls + ["avg_psnr", "std_psnr"]].round(3)

    ordered = (
        ['resolution', 'method', 'views', 'dls', 'tv', 'tv_unseen'] +
        [f"{s}_psnr" for s in SCENES] +
        ['avg_psnr', 'std_psnr']
    )

    return wide.reindex(columns=ordered)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_csv", type=str, help="Input CSV file with runs data")
    parser.add_argument("-o", "--output_csv", default="results.csv")
    args = parser.parse_args()

    raw = pd.read_csv(args.input_csv, index_col=0)
    processed = explode_runs(raw)
    tidy = consolidate(processed)
    tidy.to_csv(args.output_csv, index=False)
    print(f"Saved to {args.output_csv}")

if __name__ == "__main__":
    main()