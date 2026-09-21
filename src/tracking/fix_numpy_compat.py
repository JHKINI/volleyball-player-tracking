from pathlib import Path

ROOT = Path(
    r"C:\Users\AISW_203_115\Desktop\volleyball-tracking\TrackEval"
)

REPLACEMENTS = {
    "np.float": "float",
    "np.int": "int",
    "np.bool": "bool",
    "np.object": "object",
    "np.str": "str",
}

changed_files = []

for path in ROOT.rglob("*.py"):
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        continue

    new_text = text

    for old, new in REPLACEMENTS.items():
        new_text = new_text.replace(old, new)

    if new_text != text:
        path.write_text(new_text, encoding="utf-8")
        changed_files.append(path)

print("=" * 60)
print("NumPy 호환성 수정 완료")
print("=" * 60)

for path in changed_files:
    print(path)

print(f"\n총 {len(changed_files)}개 파일 수정")