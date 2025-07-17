import genesis as gs
import numpy as np

gs.init(backend=gs.gpu)

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0, -3.5, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        max_FPS=60,
    ),
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    show_viewer=True,
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)
ainex = scene.add_entity(
    gs.morphs.URDF(
        file='src/ainex_description/urdf/ainex.urdf',
        fixed=False,
        pos = [0.0, 0.0, 0.2],
    ),
    # gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)

########################## build ##########################
scene.build()

# joints_name = [joint.name for joint in ainex.joints][1:]
joints_name = [
            'head_pan', 'head_tilt',                                                  # 头部
            'l_sho_pitch', 'l_sho_roll', 'l_el_pitch', 'l_el_yaw', 'l_gripper',       # 左手
            'r_sho_pitch', 'r_sho_roll', 'r_el_pitch', 'r_el_yaw', 'r_gripper',       # 右手
            'l_hip_yaw', 'l_hip_roll', 'l_hip_pitch', 'l_knee', 'l_ank_pitch', 'l_ank_roll',  # 左腿
            'r_hip_yaw', 'r_hip_roll', 'r_hip_pitch', 'r_knee', 'r_ank_pitch', 'r_ank_roll',  # 右腿
            ]
num_actuated_joints = len(joints_name)
print("Ainex All Joints:", len(joints_name), joints_name)
motors_dof_idx = [ainex.get_joint(name).dofs_idx_local[0] for name in joints_name]
print(motors_dof_idx)

ainex.set_dofs_kp(kp=np.ones(num_actuated_joints) * 100.0, dofs_idx_local=motors_dof_idx)
ainex.set_dofs_kv(kv=np.ones(num_actuated_joints) * 20.0, dofs_idx_local=motors_dof_idx)
ainex.set_dofs_force_range(
    lower=np.ones(num_actuated_joints) * -50,
    upper=np.ones(num_actuated_joints) *  50,
    dofs_idx_local=motors_dof_idx,
)

q_home = np.array(
        [0.0, 0.0,                      # 'head_pan', 'head_tilt',                                                  # 头部
         0.0,-1.4, 0.0, 0.0, 0.0,       # 'l_sho_pitch', 'l_sho_roll', 'l_el_pitch', 'l_el_yaw', 'l_gripper',       # 左手
         0.0, 1.4, 0.0, 0.0, 0.0,       # 'r_sho_pitch', 'r_sho_roll', 'r_el_pitch', 'r_el_yaw', 'r_gripper',       # 右手
         0.0, 0.0, -0.0,  0.0, -0.0, 0.0,  # 'l_hip_yaw', 'l_hip_roll', 'l_hip_pitch', 'l_knee', 'l_ank_pitch', 'l_ank_roll',  # 左腿
         0.0, 0.0, 0.4000,  -0.4000, 0.2000, 0.0]  # 'r_hip_yaw', 'r_hip_roll', 'r_hip_pitch', 'r_knee', 'r_ank_pitch', 'r_ank_roll',  # 右腿
        )

target_angles = 0.4 * np.ones(num_actuated_joints)
for i in range(1000):
    # target_angles = np.sin(i * 0.01) * 0.4 * np.ones(num_actuated_joints)
    if i % 50 == 0:
        if target_angles[0] < 0.0:
            target_angles = 0.4 * np.ones(num_actuated_joints)
        else:
            target_angles = -0.4 * np.ones(num_actuated_joints)
    
    target_angles = np.zeros(num_actuated_joints)

    ainex.control_dofs_position(
        q_home,
        motors_dof_idx,
    )

    scene.step()


# TODO:
# walking controller:
#   obs: joint positions, base pos, vel, acc
#   act: joint position commands
# ainex env:
# 