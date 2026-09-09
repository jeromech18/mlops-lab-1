from pathlib import Path
from PIL import Image

CATEGORIES = {
    0: "Bread",
    1: "Dairy product",
    2: "Dessert",
    3: "Egg",
    4: "Fried food",
    5: "Meat",
    6: "Noodles-Pasta",
    7: "Rice",
    8: "Seafood",
    9: "Soup",
    10: "Vegetable-Fruit",
}

SPLITS = ["training", "evaluation", "validation"]
SIZE = (128, 128)
MINI_LIMIT = 100

RAW = Path("data/food11_raw")
PROCESSED = Path("data/food11_processed")
MINI = Path("data/food11_processed_mini")


def process_split(split: str, mini: bool) -> None:
    counts: dict[int, int] = {k: 0 for k in CATEGORIES}
    src_dir = RAW / split
    for img_path in sorted(src_dir.iterdir()):
        if not img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            continue
        category_id = int(img_path.stem.split("_")[0])
        category_name = CATEGORIES[category_id]
        if mini and counts[category_id] >= MINI_LIMIT:
            continue
        dst_dir = (MINI if mini else PROCESSED) / split / category_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(img_path) as img:
            img.resize(SIZE).save(dst_dir / img_path.name)
        counts[category_id] += 1


def main() -> None:
    for split in SPLITS:
        print(f"Processing {split}...")
        process_split(split, mini=False)
        print(f"Processing {split} (mini)...")
        process_split(split, mini=True)
    print("Done.")


if __name__ == "__main__":
    main()
