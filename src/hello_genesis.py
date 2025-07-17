import genesis as gs
import numpy as np

gs.init(backend=gs.gpu)

scene = gs.Scene(
    # viewer_options=gs.options.ViewerOptions(
    #     camera_pos=(0, -3.5, 2.5),
    #     camera_lookat=(0.0, 0.0, 0.5),
    #     camera_fov=30,
    #     max_FPS=60,
    # ),
    # sim_options=gs.options.SimOptions(
    #     dt=0.01,
    # ),
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

joints_name = [joint.name for joint in ainex.joints][1:]
num_actuated_joints = len(joints_name)
print("Ainex All Joints:", len(joints_name), joints_name)
motors_dof_idx = [ainex.get_joint(name).dofs_idx_local[0] for name in joints_name]

ainex.set_dofs_kp(kp=np.ones(num_actuated_joints) * 100.0, dofs_idx_local=motors_dof_idx)
ainex.set_dofs_kv(kv=np.ones(num_actuated_joints) * 20.0, dofs_idx_local=motors_dof_idx)
ainex.set_dofs_force_range(
    lower=np.ones(num_actuated_joints) * -50,
    upper=np.ones(num_actuated_joints) *  50,
    dofs_idx_local=motors_dof_idx,
)

target_angles = 0.4 * np.ones(num_actuated_joints)
for i in range(1000):
    # target_angles = np.sin(i * 0.01) * 0.4 * np.ones(num_actuated_joints)
    if i % 50 == 0:
        if target_angles[0] < 0.0:
            target_angles = 0.4 * np.ones(num_actuated_joints)
        else:
            target_angles = -0.4 * np.ones(num_actuated_joints)

    ainex.control_dofs_position(
        target_angles,
        motors_dof_idx,
    )

    scene.step()


# TODO:
# walking controller:
#   obs: joint positions, base pos, vel, acc
#   act: joint position commands
# ainex env:
# 