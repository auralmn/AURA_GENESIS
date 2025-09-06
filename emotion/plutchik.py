from enum import Enum, auto

class PlutchikEmotion(Enum):
    JOY = auto()
    TRUST = auto()
    FEAR = auto()
    SURPRISE = auto()
    SADNESS = auto()
    ANTICIPATION = auto()
    ANGER = auto()
    DISGUST = auto()
    # You can extend with blended or nuanced emotions as needed

PLUTCHIK_EMBEDDING_PARAMS = {
    PlutchikEmotion.JOY:     {"freq": 2.0, "amp": 1.0, "phase": 0.0},
    PlutchikEmotion.TRUST:   {"freq": 1.2, "amp": 0.8, "phase": 0.3},
    PlutchikEmotion.FEAR:    {"freq": 1.8, "amp": 1.2, "phase": 1.4},
    PlutchikEmotion.SURPRISE:{"freq": 2.5, "amp": 1.1, "phase": 2.0},
    PlutchikEmotion.SADNESS: {"freq": 0.9, "amp": 0.6, "phase": 0.6},
    PlutchikEmotion.ANTICIPATION: {"freq": 1.4, "amp": 0.9, "phase": 0.8},
    PlutchikEmotion.ANGER:   {"freq": 2.2, "amp": 1.3, "phase": 1.7},
    PlutchikEmotion.DISGUST: {"freq": 0.7, "amp": 0.7, "phase": 2.5}
}
