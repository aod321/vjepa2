import numpy as np
import h5py

from app.vjepa_droid.droid import load_states_and_extrinsics, resolve_video_metadata_path


def test_resolve_video_metadata_path_fallbacks_to_video_key():
    metadata = {
        "video_path": "recordings/MP4/episode_camera.mp4",
        "trajectory_path": "trajectory.h5",
    }

    key, value = resolve_video_metadata_path(metadata, ["left_mp4_path"])

    assert key == "video_path"
    assert value == "recordings/MP4/episode_camera.mp4"


def test_load_states_and_extrinsics_robot_schema(tmp_path):
    h5_path = tmp_path / "robot.h5"
    with h5py.File(h5_path, "w") as f:
        observation = f.create_group("observation")
        robot_state = observation.create_group("robot_state")
        robot_state.create_dataset("cartesian_position", data=np.zeros((4, 6), dtype=np.float32))
        robot_state.create_dataset("gripper_position", data=np.zeros((4,), dtype=np.float32))
        camera = observation.create_group("camera_extrinsics")
        camera.create_dataset("nav_camera_left", data=np.zeros((4, 6), dtype=np.float32))

    with h5py.File(h5_path, "r") as trajectory:
        states, extrinsics, mode = load_states_and_extrinsics(trajectory, "nav_camera")

    assert mode == "robot"
    assert states.shape == (4, 9)
    assert np.count_nonzero(states[:, 7:]) == 0
    assert extrinsics.shape == (4, 6)


def test_load_states_and_extrinsics_navigation_schema(tmp_path):
    h5_path = tmp_path / "nav.h5"
    door_counts = np.arange(21, dtype=np.float32).reshape(3, 7)
    with h5py.File(h5_path, "w") as f:
        episode = f.create_group("episode")
        episode.create_dataset("door_counts", data=door_counts)

    with h5py.File(h5_path, "r") as trajectory:
        states, extrinsics, mode = load_states_and_extrinsics(trajectory, "episode_camera")

    assert mode == "door_counts"
    assert states.shape == (3, 9)
    assert np.array_equal(states[:, : door_counts.shape[1]], door_counts)
    assert np.count_nonzero(states[:, door_counts.shape[1]:]) == 0
    assert extrinsics.shape == (3, 6)
    assert np.count_nonzero(extrinsics) == 0


def test_load_states_and_extrinsics_actions_only_schema(tmp_path):
    h5_path = tmp_path / "nav_actions_only.h5"
    actions = np.array([0, 1, 1, 2], dtype=np.int64)
    expected_counts = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 2, 0, 0, 0, 0, 0],
            [1, 2, 1, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    with h5py.File(h5_path, "w") as f:
        episode = f.create_group("episode")
        episode.create_dataset("actions", data=actions)

    with h5py.File(h5_path, "r") as trajectory:
        states, extrinsics, mode = load_states_and_extrinsics(trajectory, "episode_camera")

    assert mode == "door_counts"
    assert states.shape == (4, 9)
    assert np.allclose(states[:, : expected_counts.shape[1]], expected_counts)
    assert np.count_nonzero(states[:, expected_counts.shape[1]:]) == 0
    assert extrinsics.shape == (4, 6)
    assert np.count_nonzero(extrinsics) == 0
