import math
import numpy as np
from isaacgym import gymapi, gymutil


def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)


# simple asset descriptor for selecting from a list


class AssetDesc:
    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments


asset_descriptors = [
    AssetDesc("soccerball.urdf", False),
]
asset_root = "./"


# parse arguments
args = gymutil.parse_arguments(
    description="Joint monkey: Animate degree-of-freedom ranges",
    custom_parameters=[{
        "name":
        "--asset_id",
        "type":
        int,
        "default":
        0,
        "help":
        "Asset id (0 - %d)" % (len(asset_descriptors) - 1)
    }, {
        "name": "--speed_scale",
        "type": float,
        "default": 1.0,
        "help": "Animation speed scale"
    }, {
        "name": "--show_axis",
        "action": "store_true",
        "help": "Visualize DOF axis"
    }])

if args.asset_id < 0 or args.asset_id >= len(asset_descriptors):
    print("*** Invalid asset_id specified.  Valid range is 0 to %d" %
          (len(asset_descriptors) - 1))
    quit()

# initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
if args.physics_engine == gymapi.SIM_FLEX:
    pass
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id,
                     args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
static_friction_all = 1
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)

plane_params.static_friction = static_friction_all
plane_params.dynamic_friction = 1
plane_params.restitution = 0

gym.add_ground(sim, plane_params)

#############################################
top = 9
mesh_vertices = np.array([
    [-10, -10, top + .2],
    [-10, 10, 0.2],
    [10, -10, 0.2],
    [10, 10, 0.2],
]).astype(np.float32)
mesh_triangles = np.array([
    [1, 2, 3],
    [0, 2, 3],
    [0, 1, 3],
    ]).astype(np.uint32)

tm_params = gymapi.TriangleMeshParams()
tm_params.nb_vertices = mesh_vertices.shape[0]
tm_params.nb_triangles = mesh_triangles.shape[0]
tm_params.transform.p.x = 0.0
tm_params.transform.p.y = 0.0
tm_params.transform.p.z = 0.0
tm_params.static_friction = static_friction_all
tm_params.dynamic_friction = 1
tm_params.restitution = 0
gym.add_triangle_mesh(sim, mesh_vertices.flatten(order='C'),
                        mesh_triangles.flatten(order='C'),
                        tm_params)
#############################################

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load asset

asset_file = asset_descriptors[args.asset_id].file_name

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.flip_visual_attachments = asset_descriptors[
    args.asset_id].flip_visual_attachments
asset_options.use_mesh_materials = True

print("Loading asset '%s' from '%s'" % (asset_file, asset_root))

asset = gym.load_asset(sim, asset_root, asset_file, asset_options)


# set up the env grid
num_envs = 1
num_per_row = 6
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# position the camera
cam_pos = gymapi.Vec3(0, -3.0, top)
cam_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# cache useful handles
envs = []
actor_handles = []

print("Creating %d environments" % num_envs)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(1.0, 0, top)
    pose.r = gymapi.Quat(0, 0.0, 0.0, 1)

    actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)

    ################################################
    props = gym.get_actor_rigid_shape_properties(env, actor_handle)
    for p_idx in range(len(props)):
        props[p_idx].friction = 0
        props[p_idx].rolling_friction = 0
        props[p_idx].torsion_friction = 0
        props[p_idx].restitution = 0
    gym.set_actor_rigid_shape_properties(env, actor_handle, props)
    ################################################

    actor_handles.append(actor_handle)

# joint animation states
ANIM_SEEK_LOWER = 1
ANIM_SEEK_UPPER = 2
ANIM_SEEK_DEFAULT = 3
ANIM_FINISHED = 4

# initialize animation state
anim_state = ANIM_SEEK_LOWER
current_dof = 0

while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.

    # if len(gym.get_env_rigid_contacts(env)) > 0:
    #     gym_names = gym.get_env_rigid_contacts(env)[0]
    #     dtype_names = gym.get_env_rigid_contacts(env).dtype.names
    #     print([(gym_names[i], dtype_names[i]) for i in range(16)])
    #     import ipdb; ipdb.set_trace()
    #     print("contact!!!")
    gym.draw_env_rigid_contacts(viewer, env, gymapi.Vec3(1.0, 0.3, 1.0), 5.0, False)
    gym.sync_frame_time(sim)


print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
