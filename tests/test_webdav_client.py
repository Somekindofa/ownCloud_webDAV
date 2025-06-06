from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.webdav_client import list_dir, upload, download, delete, exists

if __name__ == "__main__":
    print("🗂  Root listing:")
    print(list_dir("/"))          # --- Search/List

    local_demo = str(Path("C:\\Users\\The Beast\\Documents\\my_projects\\ownCloud_webDAV\\tests\\demo_text.txt"))
    remote_demo = "/ai-exchange/demo_2.txt"

    print("⤴️  Uploading", local_demo)
    upload(local_demo, remote_demo)  # --- Create/Update

    print("✅ Exists?", exists(remote_demo))

    print("⬇️  Downloading back …")
    download(remote_demo, "downloaded_demo.txt")  # --- Read

    # delete(remote_demo)         # Uncomment to test Delete