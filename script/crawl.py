import os
from icrawler.builtin import BingImageCrawler

QUERIES = ["cat", "dog", "fox", "bear"]
NUM_IMAGES = 100
SAVE_DIR = "./animal_data/train"

for query in QUERIES:
    out_dir = os.path.join(SAVE_DIR, query)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[{query}] 크롤링 시작 → {out_dir}")

    crawler = BingImageCrawler(
        storage={"root_dir": out_dir},
        downloader_threads=4,
    )
    crawler.crawl(
        keyword=query,
        max_num=NUM_IMAGES,
        min_size=(100, 100),
        file_idx_offset=0,
    )

    saved = len([f for f in os.listdir(out_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))])
    print(f"[{query}] {saved}장 저장 완료")

print("\n전체 크롤링 완료")
