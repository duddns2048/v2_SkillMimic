#!/usr/bin/env python3
"""
MJCF 파일을 시각화하는 스크립트
여러 XML 파일들의 차이를 확인할 수 있습니다.
"""

import mujoco
import mujoco.viewer
import sys
import os

def visualize_mjcf(xml_path):
    """MuJoCo XML 파일을 시각화합니다."""
    if not os.path.exists(xml_path):
        print(f"Error: File not found: {xml_path}")
        return

    try:
        # XML 파일로부터 모델 로드
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)

        print(f"Successfully loaded: {xml_path}")
        print(f"Number of bodies: {model.nbody}")
        print(f"Number of joints: {model.njnt}")
        print(f"Number of actuators: {model.nu}")
        print(f"Number of DOFs: {model.nv}")

        # 뷰어 실행
        print("\nPress ESC to close the viewer")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # 시뮬레이션 루프
            while viewer.is_running():
                # 한 스텝 실행
                mujoco.mj_step(model, data)
                # 뷰어 업데이트
                viewer.sync()

    except Exception as e:
        print(f"Error loading {xml_path}: {e}")

if __name__ == "__main__":
    # 기본 경로
    mjcf_dir = "skillmimic/data/assets/mjcf"

    # 사용 가능한 파일들
    available_files = {
        "1": "mocap_humanoid.xml",
        "2": "mocap_humanoid_boxhand.xml",
        "3": "mocap_parahome_boxhand.xml",
        "4": "mocap_parahome_boxhand_s22.xml",
        "5": "mocap_parahome_boxhand_s11.xml",
        "6": "mocap_parahome_boxhand_hist.xml",
        "7": "mocap_parahome_boxhand_multiobj.xml",
        "8": "mocap_parahome_boxhand_refobj.xml",
    }

    if len(sys.argv) > 1:
        # 명령줄 인자로 파일 지정
        xml_file = sys.argv[1]
        if not xml_file.endswith('.xml'):
            xml_file = os.path.join(mjcf_dir, xml_file)
    else:
        # 대화형으로 선택
        print("Available MJCF files:")
        for key, filename in available_files.items():
            print(f"  {key}. {filename}")

        choice = input("\nSelect a file (1-8) or press Enter for default (5-s11): ").strip()
        if not choice:
            choice = "5"

        if choice in available_files:
            xml_file = os.path.join(mjcf_dir, available_files[choice])
        else:
            print("Invalid choice, using default")
            xml_file = os.path.join(mjcf_dir, available_files["3"])

    print(f"\nVisualizing: {xml_file}\n")
    visualize_mjcf(xml_file)
