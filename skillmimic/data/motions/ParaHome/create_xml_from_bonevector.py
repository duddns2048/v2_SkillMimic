#!/usr/bin/env python3
"""
bone_vectors.pkl로부터 MuJoCo XML 파일을 생성하는 스크립트
"""
import pickle
import numpy as np
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom

# Subject 번호 설정
for seq_num in range(1,208):
    # Load bone vectors
    print(f"Loading bone vectors for subject {seq_num}...")
    with open(f'/home/kimyw/github/ParaHome/data/seq/s{seq_num}/bone_vectors.pkl', 'rb') as f:
        bone_vectors = pickle.load(f)

    # Joint information
    body_vector = bone_vectors.get('body', {})
    lhand_vector = bone_vectors.get('lhand', {})
    rhand_vector = bone_vectors.get('rhand', {})

    # Create MuJoCo XML structure
    mujoco = Element('mujoco', model='humanoid')
    SubElement(mujoco, 'compiler', coordinate='local')
    SubElement(mujoco, 'statistic', extent='2', center='0 0 1')
    SubElement(mujoco, 'option', timestep='0.00555')

    # Default settings
    default = SubElement(mujoco, 'default')
    SubElement(default, 'motor', ctrlrange='-1 1', ctrllimited='true')
    SubElement(default, 'geom', type='capsule', condim='1', friction='1.0 0.05 0.05',
            solimp='.9 .99 .003', solref='.015 1')
    SubElement(default, 'joint', type='hinge', damping='0.1', stiffness='5', armature='.007',
            limited='true', solimplimit='0 .99 .01')
    SubElement(default, 'site', size='.04', group='3')

    # Asset
    asset = SubElement(mujoco, 'asset')
    SubElement(asset, 'texture', type='skybox', builtin='gradient',
            rgb1='.4 .5 .6', rgb2='0 0 0', width='100', height='100')
    SubElement(asset, 'texture', builtin='flat', height='1278', mark='cross',
            markrgb='1 1 1', name='texgeom', random='0.01',
            rgb1='0.8 0.6 0.4', rgb2='0.8 0.6 0.4', type='cube', width='127')
    SubElement(asset, 'texture', builtin='checker', height='100', name='texplane',
            rgb1='0 0 0', rgb2='0.8 0.8 0.8', type='2d', width='100')
    SubElement(asset, 'material', name='MatPlane', reflectance='0.5', shininess='1',
            specular='1', texrepeat='60 60', texture='texplane')
    SubElement(asset, 'material', name='geom', texture='texgeom', texuniform='true')

    # Worldbody
    worldbody = SubElement(mujoco, 'worldbody')
    SubElement(worldbody, 'light', cutoff='100', diffuse='1 1 1', dir='-0 0 -1.3',
            directional='true', exponent='1', pos='0 0 1.3', specular='.1 .1 .1')
    SubElement(worldbody, 'geom', conaffinity='1', condim='3', name='floor',
            pos='0 0 0', rgba='0.8 0.9 0.8 1', size='100 100 .2', type='plane', material='MatPlane')

    # Helper function to format vector
    def vec_str(vec):
        """Convert numpy array to space-separated string"""
        return ' '.join(f'{v:.8f}' if abs(v) > 1e-10 else '0' for v in vec)

    # Helper function to create joints
    def add_joints(body_elem, name, stiffness='500', damping='500'):
        """Add 3-DOF hinge joints to a body"""
        SubElement(body_elem, 'joint', name=f"{name}_x", type='hinge', pos='0 0 0',
                axis='1 0 0', stiffness=stiffness, damping=damping, armature='0.02',
                range='-180.0000 180.0000')
        SubElement(body_elem, 'joint', name=f"{name}_y", type='hinge', pos='0 0 0',
                axis='0 1 0', stiffness=stiffness, damping=damping, armature='0.02',
                range='-180.0000 180.0000')
        SubElement(body_elem, 'joint', name=f"{name}_z", type='hinge', pos='0 0 0',
                axis='0 0 1', stiffness=stiffness, damping=damping, armature='0.02',
                range='-180.0000 180.0000')

    # Store created bodies for reference
    created_bodies = {}

    # Recursive function to build hierarchy
    def build_hierarchy(parent_elem, parent_name, joint_dict, part='body'):
        """Recursively build body hierarchy from bone vectors"""
        if parent_name not in joint_dict:
            return

        children = joint_dict[parent_name]
        if not isinstance(children, dict):
            return

        for child_name, bone_vec in children.items():
            # Skip Tip joints to match s22 structure (ends at DIP)
            if 'Tip' in child_name:
                continue

            # Calculate position and length
            pos = bone_vec
            length = np.linalg.norm(bone_vec)

            # Create body element
            body = SubElement(parent_elem, 'body', name=child_name, pos=vec_str(pos))

            # Add geometry based on body part and joint type
            if part == 'body':
                # Body joints get larger capsules/spheres
                if child_name in ['jT9T8']:
                    # Torso - use sphere
                    SubElement(body, 'geom', type='sphere', contype='1', conaffinity='1',
                            density='1000', size=f'{length*0.4:.5f}',
                            pos=f'0.000 0.0000 {length*0.4:.6f}')
                elif child_name in ['jC1Head']:
                    # Head - use cylinder
                    SubElement(body, 'geom', type='cylinder', contype='1', conaffinity='1',
                            density='1000', fromto=f'0 0 0 0 0 {length:.5f}', size=f'{length*0.3:.5f}')
                elif 'Wrist' in child_name:
                    # Wrist - use box for hands
                    SubElement(body, 'geom', density='1000', type='box', pos='0 0 0',
                            size='0.035 0.03 0.01', quat='1.0000 0.0000 0.0000 0.0000')
                elif 'Ankle' in child_name:
                    # Ankle/Foot - use box
                    SubElement(body, 'geom', density='1000', type='box',
                            pos='0.04 0. -0.03', size='0.1 0.05 0.03', quat='1.0 0 0 0')
                elif 'BallFoot' in child_name:
                    # Toe - small box
                    SubElement(body, 'geom', density='1000', type='box',
                            pos='0 0 0.02', size='0.01 0.05 0.01', quat='1.0 0 0 0')
                else:
                    # Regular body segments - use capsule
                    mid = length / 2
                    if length > 0.15:  # Long bones (arms, legs)
                        SubElement(body, 'geom', type='capsule', contype='1', conaffinity='1',
                                density='1000', fromto=f'0 0 0 {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}',
                                size=f'{min(length*0.12, 0.06):.5f}')
                    elif length > 0.05:  # Medium bones (spine segments)
                        SubElement(body, 'geom', type='capsule', contype='1', conaffinity='1',
                                density='1000', fromto=f'0 0 {length*0.2:.5f} 0 0 {length*0.7:.5f}',
                                size=f'{length*0.35:.5f}')
                    else:
                        # Small segments
                        SubElement(body, 'geom', type='sphere', contype='1', conaffinity='1',
                                density='1000', size=f'{length*0.4:.5f}', pos='0 0 0')
            else:
                # Hand joints get small capsules (Tip joints are already filtered out)
                SubElement(body, 'geom', type='capsule', contype='1', conaffinity='1',
                        density='1000', fromto=f'0 0 0 {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}',
                        size='0.005')

            # Add joints
            add_joints(body, child_name)

            # Store for recursive building
            created_bodies[child_name] = body

            # Recursively build children
            build_hierarchy(body, child_name, joint_dict, part)

    # Start building from root (pHipOrigin)
    print("Building body hierarchy...")
    root_body = SubElement(worldbody, 'body', name='pHipOrigin', pos='0 0 0')
    SubElement(root_body, 'freejoint', name='pHipOrigin')

    # Add root sphere
    if 'pHipOrigin' in body_vector:
        # Calculate approximate hip size from children
        children = body_vector['pHipOrigin']
        if 'jL5S1' in children:
            spine_vec = children['jL5S1']
            hip_size = np.linalg.norm(spine_vec) * 0.98
            SubElement(root_body, 'geom', type='sphere', contype='1', conaffinity='1',
                    density='4629.6296296296305', size=f'{hip_size:.5f}', pos='0.0000 0.0000 -0.0000')

    created_bodies['pHipOrigin'] = root_body

    # Build body hierarchy
    build_hierarchy(root_body, 'pHipOrigin', body_vector, 'body')

    # Build left hand hierarchy
    if 'jLeftWrist' in created_bodies:
        build_hierarchy(created_bodies['jLeftWrist'], 'jLeftWrist', lhand_vector, 'lhand')

    # Build right hand hierarchy
    if 'jRightWrist' in created_bodies:
        build_hierarchy(created_bodies['jRightWrist'], 'jRightWrist', rhand_vector, 'rhand')

    # Add actuators
    print("Adding actuators...")
    actuator = SubElement(mujoco, 'actuator')

    def add_actuators_for_body(body_name):
        """Add motor actuators for a body's joints"""
        SubElement(actuator, 'motor', name=f"{body_name}_x", joint=f"{body_name}_x", gear='500')
        SubElement(actuator, 'motor', name=f"{body_name}_y", joint=f"{body_name}_y", gear='500')
        SubElement(actuator, 'motor', name=f"{body_name}_z", joint=f"{body_name}_z", gear='500')

    # Add actuators for all created bodies
    for body_name in created_bodies:
        if body_name != 'pHipOrigin':  # Skip root (has freejoint)
            add_actuators_for_body(body_name)

    # Format and save XML
    print(f"Saving XML file...")
    xml_str = tostring(mujoco, encoding='unicode')
    dom = xml.dom.minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")

    # Remove extra blank lines
    lines = [line for line in pretty_xml.split('\n') if line.strip()]
    pretty_xml = '\n'.join(lines)

    output_path = f"/home/kimyw/github/v2_SkillMimic/skillmimic/data/assets/mjcf/mocap_parahome_boxhand_s{seq_num}.xml"
    with open(output_path, "w") as f:
        f.write(pretty_xml)

    print(f"\n✅ XML file created successfully!")
    print(f"   Output: {output_path}")
    print(f"   Bodies: {len(created_bodies)}")
    print(f"   Actuators: {(len(created_bodies)-1)*3}")
