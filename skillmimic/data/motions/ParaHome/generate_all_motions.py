#!/usr/bin/env python3
"""
ParaHome 데이터셋에서 자동으로 모든 모션 데이터를 생성하는 스크립트
"""

import os
import json
import subprocess
from pathlib import Path

# 경로 설정
PARAHOME_ROOT = '/home/kimyw/github/ParaHome/data/seq'
SKILL_OBJ_DICT_PATH = 'skillmimic/data/motions/ParaHome/skill_obj_dict.json'
PT_DATA_SCRIPT = 'skillmimic/data/motions/ParaHome/pt_data_from_parahome.py'

def load_skill_obj_dict(path):
    """skill_obj_dict.json 로드"""
    with open(path, 'r') as f:
        return json.load(f)

def load_text_annotation(seq_path):
    """특정 시퀀스의 text_annotation.json 로드"""
    annotation_path = os.path.join(seq_path, 'text_annotation.json')
    if not os.path.exists(annotation_path):
        return None
    with open(annotation_path, 'r') as f:
        return json.load(f)

def check_required_files(seq_path):
    """필요한 파일들이 모두 존재하는지 확인"""
    required_files = [
        'bone_vectors.pkl',
        'body_global_transform.pkl',
        'body_joint_orientations.pkl',
        'hand_joint_orientations.pkl',
        'joint_positions.pkl',
        'object_transformations.pkl',
        'local_body_rot.npy',
        'text_annotation.json'
    ]
    for file in required_files:
        if not os.path.exists(os.path.join(seq_path, file)):
            return False
    return True

def run_pt_data_generation(seq_num, start_frame, end_frame, obj_names, motion_name, skill_num):
    """pt_data_from_parahome.py 실행"""
    cmd = [
        'python', PT_DATA_SCRIPT,
        '--seq_num', str(seq_num),
        '--start_frame', str(start_frame),
        '--end_frame', str(end_frame),
        '--obj_names'] + obj_names + [
        '--motion_name', motion_name,
        '--skill_num', str(skill_num)
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing s{seq_num} [{start_frame}-{end_frame}] {motion_name}")
        print(f"  Error: {e.stderr}")
        return False

def main():
    # skill_obj_dict 로드
    print("Loading skill_obj_dict.json...")
    skill_obj_dict = load_skill_obj_dict(SKILL_OBJ_DICT_PATH)
    print(f"Found {len(skill_obj_dict)} skills to process:")
    for motion_name, info in skill_obj_dict.items():
        print(f"  - {motion_name} (skill {info['skill_num']}, objects: {info['objects']})")
    print()

    # 통계
    total_processed = 0
    total_found = 0
    total_skipped = 0
    total_errors = 0

    # s1~s207 순회
    for seq_num in range(1, 208):
        seq_path = os.path.join(PARAHOME_ROOT, f's{seq_num}')

        # 폴더 존재 확인
        if not os.path.exists(seq_path):
            continue

        # 필요한 파일들 존재 확인
        if not check_required_files(seq_path):
            print(f"s{seq_num}: Missing required files, skipping...")
            total_skipped += 1
            continue

        # text_annotation.json 로드
        text_annotation = load_text_annotation(seq_path)
        if text_annotation is None:
            print(f"s{seq_num}: No text_annotation.json, skipping...")
            total_skipped += 1
            continue

        print(f"\n{'='*80}")
        print(f"Processing s{seq_num}...")
        print(f"{'='*80}")

        # 각 annotation 확인
        for frame_range, expression in text_annotation.items():
            # skill_obj_dict에 해당하는 motion이 있는지 확인
            if expression in skill_obj_dict:
                # 프레임 범위 파싱
                start_frame, end_frame = map(int, frame_range.split())

                # skill 정보 가져오기
                skill_info = skill_obj_dict[expression]
                skill_num = skill_info['skill_num']
                obj_names = skill_info['objects']

                print(f"\n  Found match: {expression}")
                print(f"    Frames: {start_frame}-{end_frame}")
                print(f"    Skill: {skill_num}")
                print(f"    Objects: {obj_names}")

                total_found += 1

                # pt_data_from_parahome.py 실행
                success = run_pt_data_generation(
                    seq_num=seq_num,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    obj_names=obj_names,
                    motion_name=expression,
                    skill_num=skill_num
                )

                if success:
                    total_processed += 1
                else:
                    total_errors += 1

    # 최종 결과 출력
    print(f"\n{'='*80}")
    print(f"Generation Complete!")
    print(f"{'='*80}")
    print(f"Total matched motions: {total_found}")
    print(f"Successfully processed: {total_processed}")
    print(f"Errors: {total_errors}")
    print(f"Skipped sequences: {total_skipped}")
    print()

if __name__ == "__main__":
    main()