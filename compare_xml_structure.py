#!/usr/bin/env python3
"""
두 XML 파일의 구조를 비교하는 스크립트
"""
import xml.etree.ElementTree as ET
from collections import defaultdict

def analyze_xml_structure(xml_path):
    """XML 파일의 구조를 분석"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Joint 이름 추출
    joints = []
    for joint in root.findall('.//joint'):
        joint_name = joint.get('name')
        if joint_name:
            joints.append(joint_name)

    # Body 이름 추출
    bodies = []
    for body in root.findall('.//body'):
        body_name = body.get('name')
        if body_name:
            bodies.append(body_name)

    # Actuator (motor) 이름 추출
    actuators = []
    for motor in root.findall('.//motor'):
        motor_name = motor.get('name')
        if motor_name:
            actuators.append(motor_name)

    # Freejoint 추출
    freejoints = []
    for freejoint in root.findall('.//freejoint'):
        freejoint_name = freejoint.get('name')
        if freejoint_name:
            freejoints.append(freejoint_name)

    # Body 계층 구조 추출
    def get_body_hierarchy(element, parent_name=None, level=0):
        hierarchy = []
        for body in element.findall('./body'):
            body_name = body.get('name')
            hierarchy.append({
                'name': body_name,
                'parent': parent_name,
                'level': level
            })
            # 재귀적으로 자식 추출
            hierarchy.extend(get_body_hierarchy(body, body_name, level + 1))
        return hierarchy

    worldbody = root.find('.//worldbody')
    body_hierarchy = get_body_hierarchy(worldbody)

    return {
        'joints': sorted(joints),
        'bodies': sorted(bodies),
        'actuators': sorted(actuators),
        'freejoints': freejoints,
        'body_hierarchy': body_hierarchy,
        'joint_count': len(joints),
        'body_count': len(bodies),
        'actuator_count': len(actuators),
    }

def compare_structures(file1, file2, name1, name2):
    """두 XML 파일의 구조를 비교"""
    print("="*80)
    print(f"Comparing {name1} vs {name2}")
    print("="*80)

    struct1 = analyze_xml_structure(file1)
    struct2 = analyze_xml_structure(file2)

    # 기본 통계
    print("\n📊 Basic Statistics:")
    print(f"{'Metric':<20} {name1:>20} {name2:>20} {'Match':>10}")
    print("-"*80)

    metrics = [
        ('Bodies', struct1['body_count'], struct2['body_count']),
        ('Joints', struct1['joint_count'], struct2['joint_count']),
        ('Actuators', struct1['actuator_count'], struct2['actuator_count']),
        ('Freejoints', len(struct1['freejoints']), len(struct2['freejoints'])),
    ]

    for metric, val1, val2 in metrics:
        match = "✅" if val1 == val2 else "❌"
        print(f"{metric:<20} {val1:>20} {val2:>20} {match:>10}")

    # Body 이름 비교
    print("\n\n🔍 Body Names Comparison:")
    bodies1_set = set(struct1['bodies'])
    bodies2_set = set(struct2['bodies'])

    if bodies1_set == bodies2_set:
        print("  ✅ All body names are IDENTICAL!")
        print(f"  Total: {len(bodies1_set)} bodies")
    else:
        print("  ❌ Body names are DIFFERENT!")
        only_in_1 = bodies1_set - bodies2_set
        only_in_2 = bodies2_set - bodies1_set

        if only_in_1:
            print(f"\n  Only in {name1}:")
            for body in sorted(only_in_1):
                print(f"    - {body}")

        if only_in_2:
            print(f"\n  Only in {name2}:")
            for body in sorted(only_in_2):
                print(f"    - {body}")

    # Joint 이름 비교
    print("\n\n🔍 Joint Names Comparison:")
    joints1_set = set(struct1['joints'])
    joints2_set = set(struct2['joints'])

    if joints1_set == joints2_set:
        print("  ✅ All joint names are IDENTICAL!")
        print(f"  Total: {len(joints1_set)} joints")
    else:
        print("  ❌ Joint names are DIFFERENT!")
        only_in_1 = joints1_set - joints2_set
        only_in_2 = joints2_set - joints1_set

        if only_in_1:
            print(f"\n  Only in {name1}:")
            for joint in sorted(only_in_1):
                print(f"    - {joint}")

        if only_in_2:
            print(f"\n  Only in {name2}:")
            for joint in sorted(only_in_2):
                print(f"    - {joint}")

    # Actuator 이름 비교
    print("\n\n🔍 Actuator Names Comparison:")
    actuators1_set = set(struct1['actuators'])
    actuators2_set = set(struct2['actuators'])

    if actuators1_set == actuators2_set:
        print("  ✅ All actuator names are IDENTICAL!")
        print(f"  Total: {len(actuators1_set)} actuators")
    else:
        print("  ❌ Actuator names are DIFFERENT!")
        only_in_1 = actuators1_set - actuators2_set
        only_in_2 = actuators2_set - actuators1_set

        if only_in_1:
            print(f"\n  Only in {name1}:")
            for actuator in sorted(only_in_1):
                print(f"    - {actuator}")

        if only_in_2:
            print(f"\n  Only in {name2}:")
            for actuator in sorted(only_in_2):
                print(f"    - {actuator}")

    # Tree 구조 비교 (상위 10개만)
    print("\n\n🌳 Body Hierarchy Sample (first 15):")
    print(f"\n{name1}:")
    for item in struct1['body_hierarchy'][:15]:
        indent = "  " * item['level']
        print(f"{indent}└─ {item['name']} (parent: {item['parent']})")

    print(f"\n{name2}:")
    for item in struct2['body_hierarchy'][:15]:
        indent = "  " * item['level']
        print(f"{indent}└─ {item['name']} (parent: {item['parent']})")

    # 계층 구조 비교
    print("\n\n🔍 Hierarchy Structure Comparison:")
    hierarchy1 = [(item['name'], item['parent']) for item in struct1['body_hierarchy']]
    hierarchy2 = [(item['name'], item['parent']) for item in struct2['body_hierarchy']]

    if hierarchy1 == hierarchy2:
        print("  ✅ Tree structures are IDENTICAL!")
    else:
        print("  ❌ Tree structures are DIFFERENT!")
        print(f"  {name1} has {len(hierarchy1)} parent-child relationships")
        print(f"  {name2} has {len(hierarchy2)} parent-child relationships")

    print("\n" + "="*80)
    print("Comparison Complete!")
    print("="*80 + "\n")

if __name__ == "__main__":
    file1 = "/home/kimyw/github/v2_SkillMimic/skillmimic/data/assets/mjcf/mocap_parahome_boxhand_s11.xml"
    file2 = "/home/kimyw/github/v2_SkillMimic/skillmimic/data/assets/mjcf/mocap_parahome_boxhand_s22.xml"

    compare_structures(file1, file2, "s11 (no Tip)", "s22")
