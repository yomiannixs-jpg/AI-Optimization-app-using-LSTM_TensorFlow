import io
import pandas as pd
import numpy as np

DATE_CANDIDATES = ["date","Date","DATE","TradingDate","timestamp","Datetime","datetime","time","Time"]

def _find_date_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c) in DATE_CANDIDATES or "date" in str(c).lower() or "time" in str(c).lower():
            tmp = pd.to_datetime(df[c], errors="coerce")
            if tmp.notna().mean() > 0.6:
                return c
    tmp = pd.to_datetime(df.iloc[:,0], errors="coerce")
    if tmp.notna().mean() > 0.6:
        return df.columns[0]
    raise ValueError("Could not detect a date/time column. Rename it to include 'Date' or 'Time'.")

def load_any_prices(file_bytes: bytes, filename: str, sheet_name: str | None = None) -> pd.DataFrame:
    fn = filename.lower()
    if fn.endswith(".xlsx"):
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
        sheets = xl.sheet_names
        if len(sheets) > 1:
            frames = []
            for sh in sheets:
                df = xl.parse(sh)
                df.columns = [str(c).strip() for c in df.columns]
                dcol = _find_date_col(df)
                price_candidates = ["PX_LAST","Close","CLOSE","close","Price","price","Last","last","Value","value","IndexValue"]
                pcol = None
                for c in df.columns:
                    if str(c) in price_candidates:
                        pcol = c; break
                if pcol is None:
                    num = df.select_dtypes(include=[np.number]).columns
                    if len(num) == 0:
                        continue
                    pcol = num[-1]
                tmp = df[[dcol, pcol]].copy()
                tmp[dcol] = pd.to_datetime(tmp[dcol], errors="coerce")
                tmp = tmp.dropna(subset=[dcol]).sort_values(dcol)
                tmp[pcol] = pd.to_numeric(tmp[pcol], errors="coerce")
                tmp = tmp.dropna(subset=[pcol])
                tmp = tmp.set_index(dcol).rename(columns={pcol: sh})
                frames.append(tmp)
            if not frames:
                raise ValueError("Could not read any sheets into price series. Each sheet needs Date and a price column.")
            wide = pd.concat(frames, axis=1).sort_index().ffill().bfill()
            wide = wide.loc[:, wide.notna().mean() > 0.7]
            if wide.shape[1] < 2:
                raise ValueError("After reading multi-sheet XLSX, fewer than 2 usable series remain.")
            return wide
        df = xl.parse(sheet_name or sheets[0])
    elif fn.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        raise ValueError("Unsupported file type. Upload .xlsx or .csv")

    df.columns = [str(c).strip() for c in df.columns]
    dcol = _find_date_col(df)
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).sort_values(dcol)

    id_candidates = ["Sector","sector","Index","index","Ticker","ticker","Symbol","symbol","Name","name"]
    price_candidates = ["PX_LAST","Close","CLOSE","close","Price","price","Last","last","Value","value","IndexValue"]
    id_col = next((c for c in df.columns if c in id_candidates), None)
    price_col = next((c for c in df.columns if c in price_candidates), None)

    if id_col is not None and price_col is not None:
        wide = df.pivot_table(index=dcol, columns=id_col, values=price_col, aggfunc="last")
        wide = wide.sort_index()
        wide = wide.apply(pd.to_numeric, errors="coerce").ffill().bfill()
        wide = wide.loc[:, wide.notna().mean() > 0.7]
        if wide.shape[1] < 2:
            raise ValueError("Long format pivot left fewer than 2 usable series.")
        return wide

    df = df.set_index(dcol)
    num = df.select_dtypes(include=[np.number]).copy()
    if num.shape[1] == 0:
        num = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    drop_keywords = ["volume","turnover","qty","quantity"]
    vol_like = [c for c in num.columns if any(k in str(c).lower() for k in drop_keywords)]
    non_vol = [c for c in num.columns if c not in vol_like]
    if len(non_vol) >= 2 and len(vol_like) > 0:
        num = num[non_vol]
    num = num.replace([np.inf,-np.inf], np.nan).ffill().bfill()
    num = num.loc[:, num.notna().mean() > 0.7]
    if num.shape[1] < 2:
        raise ValueError("Wide format has fewer than 2 usable numeric series.")
    return num
