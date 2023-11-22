
from communicate_gsm8k_env import GSM8kCoordinator
from communicate_hotpotqa_env import ReactQACoordinator
from communicate_alfworld_env import ReactAlfworldCoordinator
from communicate_chameleon_env import ChameleonCoordinator


ALL_ENVIRONMENTS = [
    GSM8kCoordinator,
    ReactQACoordinator,
    ReactAlfworldCoordinator,
    ChameleonCoordinator,
]