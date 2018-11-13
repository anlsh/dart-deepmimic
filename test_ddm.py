import pytest
import numpy as np
import random
from visak_dartdeepmimic import VisakDartDeepMimicEnv
from env_jesus import DartHumanoid3D_cartesian
from baselines.ppo1 import mlp_policy
import itertools
from baselines.common import set_global_seeds, tf_util as U
import os

#################################################
# Number of random test samples to fuzz against #
#################################################
NUM_NN_OUTPUT = 20
NUM_UNSANITIZED_OBS = 50
NUM_UNSANITIZED_SKELQ = 20
NUM_RANDOM_FRAMES = 10

NUM_TEST_RESETS = 20


class RandomPolicyAgent:
    """The world's simplest agent!"""
    def __init__(self, obs_space, action_space, hid_size, num_hid_layers):
        self.actor = mlp_policy.MlpPolicy("pi", obs_space, action_space,
                                          hid_size, num_hid_layers)

    def act(self, observation, reward, done):
        a, _ = self.actor.act(True, observation)
        return a

@pytest.fixture(scope="module")
def rng_seed():
    print("Making a random seed!")
    seed = random.randint(0, 100000)
    set_global_seeds(seed)
    return seed

@pytest.fixture
def policy(vddm_env):
    ret = RandomPolicyAgent(vddm_env.observation_space,
                            vddm_env.action_space,
                            hid_size=128, num_hid_layers=2)
    U.initialize()
    return ret

@pytest.fixture(scope="module")
def vddm_env(rng_seed):
    dir_prefix = os.path.dirname(os.path.realpath(__file__)) + "/"
    env = VisakDartDeepMimicEnv(
        skel_path=dir_prefix + "assets/skel/kima_original.skel",
        mocap_path=dir_prefix + "assets/mocap/jump/positions.txt",
        statemode=1,
        actionmode=2,
        pos_noise=0.005, vel_noise=0.005,
        pos_weight=1.65, pos_decay=-2,
        vel_weight=0.1, vel_decay=-1e-1,
        ee_weight=0.1, ee_decay=-40,
        com_weight=0.25, com_decay=-40,
        # default_damping=10, default_spring=0,
        # default_friction=20,
        # visualize=False,
        # gravity=True,
        # self_collide=True,
        # delta_actions=False,
        seed=rng_seed,
    )
    # TODO Does this need to be enabled?
    # env.seed(rng_seed)
    return env

@pytest.fixture(scope="module")
def raw_env(rng_seed):
    env = DartHumanoid3D_cartesian(rng_seed)
    return env

@pytest.fixture
def nn_output(vddm_env, raw_env):
    return [np.random.rand(vddm_env.action_dim)
            for _ in range(NUM_NN_OUTPUT)]

@pytest.fixture
def random_unsanitized_obs(vddm_env):
    # Literally a random string of numbers with the correct dimension,
    # no garuntee such a state is even possible
    return [np.random.rand(vddm_env.obs_dim)
            for _ in range(NUM_UNSANITIZED_OBS)]

@pytest.fixture
def random_unsanitized_skelq(vddm_env):
    # Literally a random string of numbers with the correct dimension,
    # no garuntee such a state is even possible
    return [np.random.rand(len(vddm_env.robot_skeleton.q))
            for _ in range(NUM_UNSANITIZED_SKELQ)]

@pytest.fixture
def random_framenum(vddm_env):
    return [random.randint(0, vddm_env.num_frames - 1)
            for _ in range(NUM_RANDOM_FRAMES)]

def test_raw_framenums(raw_env):
    # To achieve parity here I had to drop the last frame of the
    # endeffector/com data in raw_env
    assert(len(raw_env.com)
           == len(raw_env.rarm_endeffector)
           == len(raw_env.larm_endeffector)
           == len(raw_env.lfoot_endeffector)
           == len(raw_env.rfoot_endeffector)
           == len(raw_env.MotionPositions)
           == len(raw_env.MotionVelocities)
           == raw_env.num_frames)

def test_parity_number_frames(vddm_env, raw_env):
    # To achieve parity here I had to actually calculate the
    # number of frames in raw_env
    assert(vddm_env.num_frames == raw_env.num_frames)

def test_parity_skel(vddm_env, raw_env):
    assert(len(vddm_env.robot_skeleton.q) == len(raw_env.robot_skeleton.q))

def test_parity_obs_dim(vddm_env, raw_env):
    # Had to create an obs_dim variable in raw_env
    # Also, I had to replace a section of my obs_dim calculation with his
    assert(vddm_env.obs_dim == raw_env.obs_dim)

def test_parity_obs(vddm_env, raw_env):
    # Had to change some indices around to make things work
    np.testing.assert_array_equal(vddm_env._get_obs(), raw_env._get_obs())
    assert(np.isfinite(vddm_env._get_obs()).all())

def test_parity_action_dim(vddm_env, raw_env):
    # Had to create an action_dim variable in raw_env
    assert(vddm_env.action_dim == raw_env.action_dim)

def test_parity_mocap_q(vddm_env, raw_env):
    np.testing.assert_array_equal(vddm_env.ref_q_frames,
                          raw_env.MotionPositions)
    assert(np.isfinite(vddm_env.ref_q_frames).all())

def test_parity_mocap_dq(vddm_env, raw_env):
    np.testing.assert_array_equal(vddm_env.ref_dq_frames,
                          raw_env.MotionVelocities)
    assert(np.isfinite(vddm_env.ref_dq_frames).all())

def test_parity_mocap_com(vddm_env, raw_env):
    # TODO My mocap parsing is DIFFERENT, had to override my own mocap
    # data with Visak's
    np.testing.assert_array_equal(vddm_env.ref_com_frames,
                          raw_env.com)
    assert(np.isfinite(vddm_env.ref_com_frames).all())

def test_parity_mocap_eepositions(vddm_env, raw_env):
    # TODO My mocap parsing is DIFFERENT, had to override my own mocap
    # data with Visak's
    np.testing.assert_array_equal(vddm_env.ref_ee_frames[:,0],
                          raw_env.rarm_endeffector)
    np.testing.assert_array_equal(vddm_env.ref_ee_frames[:,1],
                          raw_env.larm_endeffector)
    np.testing.assert_array_equal(vddm_env.ref_ee_frames[:,2],
                          raw_env.rfoot_endeffector)
    np.testing.assert_array_equal(vddm_env.ref_ee_frames[:,3],
                          raw_env.lfoot_endeffector)
    assert(np.isfinite(vddm_env.ref_ee_frames).all())

def test_parity_angle_calculation(vddm_env, raw_env, nn_output):

    for nno in nn_output:
        tmp = vddm_env.angles_from_netvector(nno)
        assert np.array_equal(tmp, raw_env.transformActions(nno))
        assert(np.isfinite(tmp).all())

def test_parity_torques(vddm_env, raw_env, nn_output):
    # Had to modify raw_env.PID to take in a target angle

    # Angle calculation parity has already been tested at this point,
    # so I can safely use this one method

    for nno in nn_output:
        angles = vddm_env.angles_from_netvector(nno)
        tmp = vddm_env.PID(vddm_env.robot_skeleton, angles)
        np.testing.assert_array_equal(tmp, raw_env.PID(raw_env.robot_skeleton,
                                                       np.concatenate([np.zeros(6),
                                                                       angles])))
        assert(np.isfinite(tmp).all())

def test_parity_termination(vddm_env, raw_env, random_unsanitized_obs):
    # Had to create a should_terminate method in raw_env for this

    # Also, it doesn't test rude terminatino yet!!
    for ruo in random_unsanitized_obs:
        assert(vddm_env.should_terminate(ruo)[0]
               == raw_env.should_terminate(vddm_env.robot_skeleton,
                                           ruo))

def test_parity_reset_framenums(vddm_env, raw_env):

    test_parity_obs(vddm_env, raw_env)
    for _ in range(NUM_TEST_RESETS):
        assert(vddm_env.get_random_framenum()
               == raw_env.get_random_framenum())

def test_parity_reset(vddm_env, raw_env):
    # IMPURE This test will CHANGE ENVIRONMENT STATE
    test_parity_obs(vddm_env, raw_env)
    for i in range(20):
        vddm_env.reset()
        raw_env.reset()
        test_parity_obs(vddm_env, raw_env)

def test_parity_ee_reward(vddm_env, raw_env, random_framenum):
    skel = vddm_env.robot_skeleton
    for rf in itertools.product(random_framenum):
        tmp = vddm_env.ee_reward(skel, rf)
        assert(tmp == raw_env.ee_reward(skel, rf))
        assert np.isfinite(tmp)

def test_parity_com_reward(vddm_env, raw_env, random_framenum):
    skel = vddm_env.robot_skeleton
    for rf in itertools.product(random_framenum):
        tmp = vddm_env.com_reward(skel, rf)
        assert(tmp == raw_env.com_reward(skel, rf))
        assert(np.isfinite(tmp))

def test_parity_vel_reward(vddm_env, raw_env, random_framenum):
    skel = vddm_env.robot_skeleton
    for rf in itertools.product(random_framenum):
        tmp = vddm_env.vel_reward(skel, rf)
        assert(tmp == raw_env.vel_reward(skel, rf))

def test_parity_quat_reward(vddm_env, raw_env, random_framenum):
    skel = vddm_env.robot_skeleton
    for rf in random_framenum:
        tmp = vddm_env.quat_reward(skel, rf)
        assert(tmp == raw_env.quat_reward(skel, rf))

def test_parity_reward(vddm_env, raw_env, random_framenum):
    skel = vddm_env.robot_skeleton
    for rf in random_framenum:
        tmp = vddm_env.reward(skel, rf)
        assert(tmp == raw_env.reward(skel, rf))

def test_parity_simstep(vddm_env, raw_env, random_framenum):
    # IMPURE Runnning this test will CHANGE THE ENVIRONMENT STATES

    for rf in random_framenum:
        vddm_env.reset(rf)
        raw_env.reset_model(rf)
        test_parity_obs(vddm_env, raw_env)

        tau = np.random.rand(len(vddm_env.robot_skeleton.q))
        vddm_env.robot_skeleton.set_forces(tau)
        raw_env.robot_skeleton.set_forces(tau)
        vddm_env.dart_world.step()
        raw_env.dart_world.step()

        test_parity_obs(vddm_env, raw_env)

def test_parity_step(vddm_env, raw_env):
    # IMPURE Running this test will CHANGE EVIRONMENT STATES
    # There's some redundancy in the test but whatever...

    test_parity_obs(vddm_env, raw_env)
    nn_output = np.random.rand(vddm_env.action_dim)

    v_obs, v_reward, v_term, _ = vddm_env._step(nn_output)
    r_obs, r_reward, r_term, _ = raw_env._step(nn_output)

    test_parity_obs(vddm_env, raw_env)
    assert(v_reward == r_reward)
    assert(v_term == r_term)

    return (v_obs, v_reward, v_term, _)

def test_parity_rollout(vddm_env, raw_env, policy):
    # Might be some code duplication in here but whatever

    v_term = False
    v_reward = 0

    while not v_term:
        test_parity_obs(vddm_env, raw_env)
        nn_output = policy.act(vddm_env._get_obs(), v_term, v_reward)

        ######################################################
        # The code in this section is duplicated from above! #
        ######################################################

        test_parity_obs(vddm_env, raw_env)
        nn_output = np.random.rand(vddm_env.action_dim)

        v_obs, v_reward, v_term, _ = vddm_env._step(nn_output)
        r_obs, r_reward, r_term, _ = raw_env._step(nn_output)

        test_parity_obs(vddm_env, raw_env)
        assert(v_reward == r_reward)
        assert(v_term == r_term)

        #######################
        # End duplicated code #
        #######################
