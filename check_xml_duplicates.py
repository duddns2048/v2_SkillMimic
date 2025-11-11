#!/usr/bin/env python3
"""
XML 파일들이 실제로 동일한지 확인하는 스크립트
"""

import os
import hashlib
from pathlib import Path

def get_file_hash(filepath):
    """파일의 SHA256 해시를 계산합니다."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def compare_files(file_groups):
    """파일 그룹들을 비교합니다."""
    mjcf_dir = Path("skillmimic/data/assets/mjcf")

    for group_name, files in file_groups.items():
        print(f"\n{'='*70}")
        print(f"Group: {group_name}")
        print('='*70)

        hashes = {}
        all_same = True
        reference_hash = None

        for filename in files:
            filepath = mjcf_dir / filename
            if not filepath.exists():
                print(f"  ❌ {filename}: FILE NOT FOUND")
                continue

            file_hash = get_file_hash(filepath)
            file_size = filepath.stat().st_size

            # 첫 번째 파일을 기준으로 설정
            if reference_hash is None:
                reference_hash = file_hash
                print(f"  📄 {filename}")
                print(f"     ├─ Hash: {file_hash[:16]}...")
                print(f"     └─ Size: {file_size:,} bytes")
            else:
                is_same = (file_hash == reference_hash)
                symbol = "✅" if is_same else "❌"
                print(f"  📄 {filename}")
                print(f"     ├─ Hash: {file_hash[:16]}...")
                print(f"     ├─ Size: {file_size:,} bytes")
                print(f"     └─ {symbol} {'SAME' if is_same else 'DIFFERENT'}")

                if not is_same:
                    all_same = False

            hashes[filename] = file_hash

        # 결과 요약
        print(f"\n  {'🎉 All files are IDENTICAL!' if all_same else '⚠️  Files are DIFFERENT!'}")

        # 파일들 간의 차이 확인
        if not all_same:
            unique_hashes = {}
            for filename, file_hash in hashes.items():
                if file_hash not in unique_hashes:
                    unique_hashes[file_hash] = []
                unique_hashes[file_hash].append(filename)

            print(f"\n  Unique file groups: {len(unique_hashes)}")
            for i, (hash_val, filenames) in enumerate(unique_hashes.items(), 1):
                print(f"    Group {i}:")
                for fn in filenames:
                    print(f"      - {fn}")

if __name__ == "__main__":
    # 확인할 파일 그룹들
    file_groups = {
        "ParaHome Standard (기본 신체 비율)": [
            "mocap_parahome_boxhand.xml",
            "mocap_parahome_boxhand_multiobj.xml",
            "mocap_parahome_boxhand_refobj.xml",
            "mocap_parahome_boxhand_hist.xml",
            "mocap_parahome_boxhand_multirefobj.xml",
        ],
        "ParaHome S22 (Subject 22 신체 비율)": [
            "mocap_parahome_boxhand_s22.xml",
            "mocap_parahome_boxhand_refobj_s22.xml",
            "mocap_parahome_boxhand_hist_s22.xml",
        ],
        "Humanoid (비교용)": [
            "mocap_humanoid.xml",
            "mocap_humanoid_boxhand.xml",
        ],
    }

    print("\n🔍 Checking XML file duplicates...")
    compare_files(file_groups)
    print("\n" + "="*70)
    print("Done!")
    print("="*70 + "\n")
