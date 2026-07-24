"""Shared sample scene used by the omni-model prototypes, so results across
different candidate models are actually comparable. Mirrors the README's
classroom example; not run through the real Director LLM (director/generate.py)
-- the point of these prototypes is to test a candidate model's own
story+image(+BGM) generation against the same kind of input the Director
would produce, in isolation.
"""

SAMPLE_SCENE = {
    "character_name": "ヒロイン",
    "character_description": "長い銀髪、青い目、清楚な女子高生、セーラー服",
    "background_description": "夕暮れの誰もいない教室、桜の木が窓の外に見える",
    "persona": "普段は強気だが、恋愛には不器用。主人公のことが好きだが素直に言えない。",
    "context": "卒業式の後、二人きりで教室に残っている。",
    "genre": "青春恋愛、切ない",
    "bgm_mood": "bittersweet, nostalgic, gentle",
    "bgm_instrument": "solo piano with soft strings",
}

ILLUSTRATION_PROMPT = (
    f"{SAMPLE_SCENE['character_description']}, {SAMPLE_SCENE['background_description']}, "
    "anime illustration, event CG, dusk lighting"
)

STORY_PROMPT_JA = (
    "次のシーンのナレーションとセリフを日本語で短く書いてください。"
    f"キャラクター: {SAMPLE_SCENE['character_name']}、{SAMPLE_SCENE['character_description']}。"
    f"背景: {SAMPLE_SCENE['background_description']}。"
    f"性格: {SAMPLE_SCENE['persona']}。文脈: {SAMPLE_SCENE['context']}。"
    f"ジャンル: {SAMPLE_SCENE['genre']}。"
    "ナレーションとセリフを一つずつ、合わせて数行。"
)
