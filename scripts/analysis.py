import argparse
import ast
import re
import pandas as pd
from typing import Any, Optional, Dict, List

PSNR_KEY: str = "psnr/full test camera"
SCENE_REMAP: Dict[str, str] = {"lego_gen12": "lego"}
SCENES: List[str] = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]

NAME_RGX = re.compile(
    r"^(?P<scene>[^_]+)_"
    r"(?P<views>\d+)views_"
    r"(?P<method>[^_]+)_"
    r"dls(?P<dls>\d+)$"
)

def explode_runs(raw: pd.DataFrame) -> pd.DataFrame:
    records: List[dict[str, Any]] = []
    for _, row in raw.iterrows():
        try:
            summary: Dict[str, Any] = ast.literal_eval(row["summary"])
        except Exception:
            print(f"Skipping row with invalid summary: {row['summary']}")
            continue
        
        psnr = summary.get(PSNR_KEY)
        if psnr is None:
            print(f"No PSNR found in {row['name']}")
            continue
            
        m = NAME_RGX.match(row["name"])
        if m is None:
            print(f"Regex not matched in {row['name']}")
            continue
            
        g: Dict[str, str] = m.groupdict()
        scene: str = SCENE_REMAP.get(g["scene"], g["scene"])

        records.append(
            dict(
                method=g["method"],
                views=int(g["views"]),
                dls=int(g["dls"]),
                scene=scene,
                psnr=psnr,
            )
        )
    return pd.DataFrame.from_records(records)

def consolidate(df: pd.DataFrame) -> pd.DataFrame:
    singleview_df = df[df["method"] == "singleview"]
    other_methods_df = df[df["method"] != "singleview"]

    if not singleview_df.empty:
        singleview_agg = (
            singleview_df.groupby(["method", "views", "scene"], as_index=False)
            .psnr.mean()
        )
        singleview_agg["dls"] = "any"
        df = pd.concat([other_methods_df, singleview_agg], ignore_index=True)

    wide: pd.DataFrame = (
        df.pivot_table(
            index=["method", "views", "dls"],
            columns="scene",
            values="psnr",
            aggfunc="mean",
        )
        .reset_index()
    )

    wide = wide.rename(
        columns={s: f"{s}_psnr" for s in wide.columns if s not in ["method", "views", "dls"]}
    )

    scene_cls: List[str] = [c for c in wide.columns if c.endswith("_psnr")]
    wide["avg_psnr"] = wide[scene_cls].mean(axis=1)
    wide["std_psnr"] = wide[scene_cls].std(axis=1)
    wide[scene_cls + ["avg_psnr", "std_psnr"]] = wide[scene_cls + ["avg_psnr", "std_psnr"]].round(3)

    if 'dls' in wide.columns:
        mask = wide['dls'] == 12
        wide.loc[mask, 'method'] = 'lightfield'
    
    wide = wide.sort_values('views')

    ordered: List[str] = (
        ['views', 'method', 'dls'] +
        [f"{s}_psnr" for s in SCENES] +
        ['avg_psnr', 'std_psnr']
    )

    return wide.reindex(columns=ordered)

def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_csv", type=str, required=True, help="Input CSV file with runs data")
    parser.add_argument("-o", "--output_csv", default="results.csv")
    args: argparse.Namespace = parser.parse_args()

    # Create a dummy CSV for testing if one is not provided
    raw: pd.DataFrame = pd.read_csv(args.input_csv, index_col=0)
    processed: pd.DataFrame = explode_runs(raw)
    tidy: pd.DataFrame = consolidate(processed)
    tidy.to_csv(args.output_csv, index=False)
    print(f"Saved to {args.output_csv}")


if __name__ == "__main__":
    main()
