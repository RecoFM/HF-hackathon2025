import pandas as pd
import re

def clean_title(title: str) -> str:
    """
    Remove common “noise” from a movie title:
      1. Anything in square or round brackets (e.g. “[YTS]”, “(Director’s Cut)”).
      2. Standalone format tokens (VHS, DVD, Blu-Ray).
      3. Resolution tags (480p, 720p, 1080p, 2160p, etc.) and “4K”/“UHD”.
      4. Source tags (DVDRip, BRRip, HDRip, WEB-DL, WEBRip, HDTV, CAM, TS, etc.).
      5. Codec/encoding tags (x264, XviD, H.264, HEVC).
      6. Audio tags (AC3, AAC, DTS, Dolby, etc.).
      7. Edition markers (Director’s Cut, Extended Edition, Special Edition, Unrated, Remastered).
      8. Trailer/Teaser markers.
      9. File extensions (.avi, .mkv, .mp4, .mov, .wmv, etc.).
     10. Release-group tags after a hyphen or underscore (e.g. “-YTS”, “_RARBG”).
     11. Collapse repeated punctuation (extra hyphens/underscores) and whitespace.
    """

    # 1. Remove anything in square or round brackets
    title = re.sub(r"\[.*?\]|\(.*?\)", "", title)

    # 2. Remove standalone format tokens (case-insensitive)
    title = re.sub(r"\b(VHS|DVD|Blu[- ]?Ray)\b", "", title, flags=re.IGNORECASE)

    # 3a. Remove resolution tags (e.g. 480p, 720p, 1080p, 2160p)
    title = re.sub(r"\b[0-9]{3,4}p\b", "", title, flags=re.IGNORECASE)
    # 3b. Remove “4K” or “UHD”
    title = re.sub(r"\b(4K|UHD)\b", "", title, flags=re.IGNORECASE)

    # 4. Remove source tags (DVDRip, BRRip, HDRip, WEB[- ]DL, WEBRip, HDTV, CAM, TS, etc.)
    title = re.sub(r"\b(DVDRip|BRRip|HDRip|WEB[- ]?DL|WEBRip|HDTV|CAM|TS|SCR|DVDSCR)\b", "", title, flags=re.IGNORECASE)

    # 5. Remove codec/encoding tags (x264, XviD, H.264, HEVC, etc.)
    title = re.sub(r"\b(x264|XviD|H\.264|HEVC|H265|AVC|DivX)\b", "", title, flags=re.IGNORECASE)

    # 6. Remove audio tags (AC3, AAC, DTS, DD5.1, Dolby, Atmos, etc.)
    title = re.sub(r"\b(AC3|AAC|DTS|DD5\.1|Dolby|Atmos)\b", "", title, flags=re.IGNORECASE)

    # 7. Remove edition markers (Director’s Cut, Extended Edition, Special Edition, Unrated, Remastered)
    title = re.sub(
        r"\b(Director[’']?s Cut|Extended Edition|Special Edition|Unrated|Remastered|Anniversary Edition)\b",
        "",
        title,
        flags=re.IGNORECASE
    )

    # 8. Remove “Trailer” or “Teaser”
    title = re.sub(r"\b(Trailer|Teaser)\b", "", title, flags=re.IGNORECASE)

    # 9. Remove common video file extensions (at end of string)
    title = re.sub(r"\.(avi|mkv|mp4|mov|wmv|flv|mpeg)$", "", title, flags=re.IGNORECASE)

    # 10. Remove release-group tags after a hyphen or underscore (e.g., “-YTS”, “_RARBG”)
    title = re.sub(r"(?:-|_)\b[A-Za-z0-9]{2,}\b$", "", title)

    # 11a. Replace underscores with spaces
    title = title.replace("_", " ")

    # 11b. Collapse multiple hyphens, pipes, or dots into a single space
    title = re.sub(r"[-|\.]{2,}", " ", title)

    # 11c. Collapse any run of whitespace into a single space and strip
    title = re.sub(r"\s+", " ", title).strip()

    return title

def has_year_tag(title: str) -> bool:
    """
    Return True if the cleaned title contains a four-digit year (1900–2099).
    """
    return bool(re.search(r"\b(19|20)\d{2}\b", title))

def clean_movie_csv(input_path: str, output_path: str) -> None:
    """
    1. Load the original CSV.
    2. Drop any rows where the raw title contains “4K” or “UHD” (case-insensitive).
    3. Create a “clean_title” column by stripping out noise via clean_title().
    4. Drop any rows where clean_title contains “Collection” or “Collector” (case-insensitive).
    5. Mark which rows have a year in clean_title().
    6. Keep:
       - All rows that do have a year.
       - From rows without a year, only the first occurrence of each distinct clean_title.
    7. Rename “clean_title” back to “title” and write out a cleaned CSV.
    """
    # Load the CSV into a DataFrame
    df = pd.read_csv(input_path)

    # 2. Drop rows where the raw title contains “4K” or “UHD”
    df = df[~df["title"].str.contains(r"\b4K\b|\bUHD\b", case=False, na=False)].copy()

    # Apply cleaning to the “title” column
    df["clean_title"] = df["title"].astype(str).apply(clean_title)

    # 4. Drop rows where clean_title contains “Collection” or “Collector”
    df = df[~df["clean_title"].str.contains(r"\bCollection\b|\bCollector\b", case=False, na=False)].copy()

    # Detect which cleaned titles contain a 4-digit year (1900–2099)
    df["has_year"] = df["clean_title"].apply(has_year_tag)

    # Split into two DataFrames
    df_with_year = df[df["has_year"]].copy()
    df_without_year = df[~df["has_year"]].copy()

    # From the “no year” set, drop duplicates based on clean_title (keep first)
    df_without_year_unique = df_without_year.drop_duplicates(subset="clean_title", keep="first")

    # Combine: all “with year” rows + unique “no year” rows
    df_combined = pd.concat([df_with_year, df_without_year_unique], ignore_index=True)

    # Drop helper columns and rename “clean_title” back to “title”
    df_combined = df_combined.drop(columns=["title", "has_year"])
    df_combined = df_combined.rename(columns={"clean_title": "title"})

    # Write out the cleaned CSV
    df_combined.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Example usage:
    clean_movie_csv(
        input_path="amazon_movies_2023/title_embeddings_mapping.csv",
        output_path="amazon_movies_2023/title_embeddings_mapping_cleaned.csv"
    )
