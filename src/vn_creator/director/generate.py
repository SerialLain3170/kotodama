"""Scene Director: turns a scene description into a structured shot script.

Uses the Anthropic Messages API with a forced tool call so the model's
output is validated JSON rather than free text we'd have to parse.
"""
import json

import anthropic
from pydantic import ValidationError

from vn_creator import config
from vn_creator.director.schema import SCENE_SCRIPT_TOOL, SceneInput, SceneScript

SYSTEM_PROMPT = """\
You are the scene director for a Japanese visual novel (galge/eroge-adjacent \
dating-sim style, but keep content all-ages). Given a character, a background, \
a persona, prior context, and a genre, you generate ONE short shot: 5-10 seconds, \
exactly one line of Japanese dialogue, plus a BGM mood description, plus an \
illustration_prompt (ALWAYS required, for both shot types) describing character \
and background together as ONE unified image — never generate the character and \
background as separate, independently-composed layers; a single image model \
renders illustration_prompt as one cohesive picture.

There are two shot types:

- talking_head (the default, most common): a normal dialogue beat. The unified \
  illustration is animated directly (the face region is cropped, lip-synced, and \
  pasted back into the same image) — so illustration_prompt's framing MUST be a \
  clear, front-facing, fully unobstructed bust/upper-body shot: face centered and \
  visible, not turned away, not covered by hair strands, hands, or effects, plain \
  enough for a face-detector to find it reliably. Also fill in acting_beats: 2-5 \
  entries, each timestamped within [0, duration_sec], describing a SPECIFIC small \
  gaze/expression/body action in Japanese (not a restatement of the emotion \
  label). Leave camera_move null.

- illustration: reserve this ONLY for a genuinely climactic/dramatic/emotionally \
  peak moment (a confession, a shocking reveal, a striking visual beat) — the kind \
  of moment a real visual novel would render as a single hand-composed event CG \
  instead of the normal standing-sprite shot. Do not use it for ordinary \
  conversation. When you use it: leave acting_beats empty. illustration_prompt can \
  use any dramatic framing/angle (wide shot, dramatic angle, close-up) since \
  nothing is animated — include pose, expression, setting, lighting, and mood. \
  Also fill camera_move with a SUBTLE, SLOW Ken Burns pan/zoom (start_zoom/end_zoom \
  typically within 1.0-1.25, focus points typically within 0.3-0.7) that suits the \
  emotional beat — e.g. a slow zoom-in toward the face for tension, or a slight pan \
  for a reveal. Never move fast or zoom drastically.

Rules:
- The dialogue must be natural spoken Japanese appropriate to the persona and \
  context, and should imply subtext rather than state emotions outright \
  (e.g. hesitation, a deflected feeling, an unfinished thought) when the genre \
  calls for it.
- speech_start/speech_end must fit inside [0, duration_sec] and the dialogue must \
  plausibly be speakable in (speech_end - speech_start) seconds at natural pace \
  (~6-8 mora/sec for Japanese).
- text_start should be at or slightly before speech_start (textbox often appears \
  just before the voice starts).
- Always call the emit_scene_script tool exactly once with the full result. Do not \
  reply in plain text.
"""


def generate_scene(scene_input: SceneInput, force_shot_type: str | None = None, max_retries: int = 2) -> SceneScript:
    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it before running the Scene "
            "Director module."
        )

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    user_prompt = (
        f"Character: {scene_input.character_name} — {scene_input.character_description}\n"
        f"Background: {scene_input.background_description}\n"
        f"Persona: {scene_input.persona}\n"
        f"Context so far: {scene_input.context}\n"
        f"Genre: {scene_input.genre}\n\n"
        "Generate the next shot."
    )
    if force_shot_type:
        user_prompt += f"\n\nUse shot_type=\"{force_shot_type}\" for this shot regardless of your own judgment."

    last_error = None
    for attempt in range(max_retries + 1):
        attempt_prompt = user_prompt
        if last_error is not None:
            # Retrying with the identical prompt tends to reproduce the exact
            # same mistake (low-entropy tool-call formatting habits) — naming
            # the specific error breaks the repetition.
            attempt_prompt += (
                f"\n\n(Your previous attempt was rejected: {last_error} "
                "Every field must match the tool's JSON schema exactly — in "
                "particular, bgm must be a JSON OBJECT with mood/instrument/"
                "tempo_bpm fields, never a plain string. Fix this and retry.)"
            )

        response = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[SCENE_SCRIPT_TOOL],
            tool_choice={"type": "tool", "name": "emit_scene_script"},
            messages=[{"role": "user", "content": attempt_prompt}],
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == "emit_scene_script":
                try:
                    return SceneScript.model_validate(block.input)
                except ValidationError as e:
                    last_error = e
                    break
        else:
            last_error = RuntimeError(f"Model did not call emit_scene_script: {response.content}")

    raise RuntimeError(f"Scene Director failed validation after {max_retries + 1} attempts: {last_error}")


def _cli():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--character-name", required=True)
    parser.add_argument("--character-description", required=True)
    parser.add_argument("--background-description", required=True)
    parser.add_argument("--persona", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--genre", required=True)
    parser.add_argument("--force-shot-type", choices=["talking_head", "illustration"], default=None)
    parser.add_argument("--out", default=None, help="Path to write JSON to (default: stdout)")
    args = parser.parse_args()

    scene_input = SceneInput(
        character_name=args.character_name,
        character_description=args.character_description,
        background_description=args.background_description,
        persona=args.persona,
        context=args.context,
        genre=args.genre,
    )
    script = generate_scene(scene_input, force_shot_type=args.force_shot_type)
    out_json = json.dumps(script.model_dump(), ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_json)
    else:
        print(out_json)


if __name__ == "__main__":
    _cli()
