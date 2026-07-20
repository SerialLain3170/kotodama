"""Data model for a single generated VN scene ("shot")."""
from typing import Literal
from pydantic import BaseModel, Field


class ActingBeat(BaseModel):
    time: float = Field(..., description="Seconds from the start of the shot.")
    action: str = Field(..., description="Short description of a gaze/expression/body action, in Japanese.")


class BGMSpec(BaseModel):
    mood: str
    instrument: str
    tempo_bpm: int


class CameraMove(BaseModel):
    """A single smooth Ken-Burns move across a static illustration, normalized 0-1."""
    start_zoom: float = Field(1.0, description="1.0 = whole frame visible. Higher = more zoomed in.")
    end_zoom: float = 1.0
    start_focus_x: float = Field(0.5, description="0=left edge, 1=right edge.")
    start_focus_y: float = Field(0.5, description="0=top edge, 1=bottom edge.")
    end_focus_x: float = 0.5
    end_focus_y: float = 0.5


class SceneInput(BaseModel):
    character_name: str
    character_description: str
    background_description: str
    persona: str
    context: str
    genre: str


class SceneScript(BaseModel):
    shot_type: Literal["talking_head", "illustration"] = Field(
        "talking_head",
        description=(
            "talking_head: normal dialogue beat. The unified scene illustration (see "
            "illustration_prompt) is animated directly — the face is cropped, lip-synced, "
            "and seamlessly pasted back into the same image, so the background is never a "
            "separately-composited layer. illustration: a single dramatic/climactic event-CG "
            "moment, rendered as the same kind of unified illustration but with a slow camera "
            "pan/zoom instead of character animation."
        ),
    )
    duration_sec: float
    emotion: str
    dialogue: str = Field(..., description="The Japanese line of dialogue to be spoken.")
    acting_beats: list[ActingBeat] = Field(
        default_factory=list, description="Used only when shot_type is talking_head."
    )
    illustration_prompt: str = Field(
        ...,
        description=(
            "ALWAYS filled in, for both shot types. A single Danbooru-tag-style, "
            "comma-separated English prompt describing character AND background together "
            "in one cohesive composition (pose, expression, setting, lighting, mood, camera "
            "framing) so they render in one unified image — never a portrait with an "
            "unrelated/simple background. For shot_type=talking_head, the framing MUST be a "
            "clear, front-facing, unobstructed bust/upper-body shot (face fully visible, not "
            "turned away, not covered by hair/effects/hands) since the face will be animated. "
            "For shot_type=illustration, any dramatic framing/angle is fine since nothing is "
            "animated."
        ),
    )
    camera_move: CameraMove | None = Field(
        None, description="Used only when shot_type is illustration."
    )
    bgm: BGMSpec
    text_start: float = Field(..., description="Seconds from shot start when the textbox begins revealing text.")
    speech_start: float
    speech_end: float


SCENE_SCRIPT_TOOL = {
    "name": "emit_scene_script",
    "description": "Emit a single structured VN shot script.",
    "input_schema": SceneScript.model_json_schema(),
}
