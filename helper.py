from pathlib import Path

path = Path(__file__).parent
data_dir = path / "data"

def get_image_paths(data_dir, keyword = ""):
  no = list((data_dir / f"no{keyword}").glob("*"))
  yes = list((data_dir / f"yes{keyword}").glob("*"))
  return [no, yes]
