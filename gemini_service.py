import base64
import json
import asyncio
from google import genai
from google.genai import types

client: genai.Client | None = None
MODEL = "gemini-2.5-flash"
IMAGE_MODEL = "gemini-2.5-flash-image"


def init_client(api_key: str):
    global client
    client = genai.Client(api_key=api_key)


async def analyze_object(image_base64: str) -> dict:
    """Analyze a camera image to detect the object and classify it as a weapon attribute."""
    if client is None:
        raise RuntimeError("Gemini client not initialized")

    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    image_bytes = base64.b64decode(image_base64)

    prompt = """あなたは対象物の本質を見抜く鑑定眼を持ったAIです。
この画像には、人が手に何かを持ってカメラに見せている様子が映っています。
人物の説明は一切不要です。見せている「物体」を【戦闘での武器】として使った場合を想定し、その属性の強さを示す「パワー」（0〜100）と、「属性」を分析してください。

【重要】
属性は必ず以下の4つの中から、最もふさわしいものを1つだけ選んでください。それ以外の属性は絶対に使わないでください。
- 斬撃
- 打撃
- 盾
- 毒

また、その物体が何であるかを簡潔に説明してください。

以下のJSON形式でのみ出力してください。
{
    "object_name": "物体の名前",
    "attribute": "属性名",
    "power": 85
}"""

    last_error = None
    for attempt in range(3):
        try:
            response = await client.aio.models.generate_content(
                model=MODEL,
                contents=[
                    types.Content(
                        parts=[
                            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                            types.Part.from_text(text=prompt),
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.9,
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            last_error = e
            if attempt < 2:
                await asyncio.sleep(1)

    raise RuntimeError(f"Gemini API failed after 3 attempts: {last_error}")


async def analyze_image(image_base64: str, object_info: dict | None = None) -> dict:
    """Analyze a camera image and generate a battle character.

    If object_info is provided (from analyze_object), it will be used as
    context to generate a character that reflects the detected weapon.
    """
    if client is None:
        raise RuntimeError("Gemini client not initialized")

    # Strip data URL prefix if present
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    image_bytes = base64.b64decode(image_base64)

    if object_info:
        object_name = object_info.get("object_name", "不明な物体")
        attribute = object_info.get("attribute", "打撃")
        power = object_info.get("power", 50)

        prompt = f"""この画像に写っている物体は「{object_name}」と鑑定されました。
武器としての属性は【{attribute}】、パワーは【{power}/100】です。

この鑑定結果を元に、バトルゲームのキャラクターを生成してください。
- キャラクターの名前は「{object_name}」をベースにした創造的な名前にしてください
- 属性【{attribute}】を活かした必殺技にしてください
- パワー{power}を反映したステータス配分にしてください（パワーが高いほど強い）
- キャラクターの説明には、元の物体と属性について触れてください

以下のJSON形式で出力してください:
{{
  "name": "キャラクター名（日本語）",
  "hp": 数値(50-200),
  "attack": 数値(10-100),
  "defense": 数値(10-100),
  "speed": 数値(10-100),
  "special_move": "必殺技名（日本語）",
  "attribute": "{attribute}",
  "power": {power},
  "description": "キャラクターの説明（2-3文、日本語）"
}}

ステータスの合計は250-400の範囲にしてください。"""
    else:
        prompt = """この画像に写っているものを元に、バトルゲームのキャラクターを生成してください。
画像の内容を創造的に解釈して、ユニークで面白いキャラクターにしてください。

以下のJSON形式で出力してください:
{
  "name": "キャラクター名（日本語）",
  "hp": 数値(50-200),
  "attack": 数値(10-100),
  "defense": 数値(10-100),
  "speed": 数値(10-100),
  "special_move": "必殺技名（日本語）",
  "description": "キャラクターの説明（2-3文、日本語）"
}

ステータスの合計は250-400の範囲にしてください。画像の特徴に合ったステータス配分にしてください。"""

    last_error = None
    for attempt in range(3):
        try:
            response = await client.aio.models.generate_content(
                model=MODEL,
                contents=[
                    types.Content(
                        parts=[
                            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                            types.Part.from_text(text=prompt),
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.9,
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            last_error = e
            if attempt < 2:
                await asyncio.sleep(1)

    raise RuntimeError(f"Gemini API failed after 3 attempts: {last_error}")


async def generate_character_image(character: dict) -> str:
    """Generate a character illustration and return as base64 data URL."""
    if client is None:
        raise RuntimeError("Gemini client not initialized")

    prompt = f"""以下のバトルキャラクターのイラストを1枚描いてください。

キャラクター名: {character['name']}
必殺技: {character['special_move']}
説明: {character['description']}

【絶対に守るルール】
- 文字、テキスト、ロゴ、名前、数字、記号は一切描かないでください
- キャラクターのみを描いてください
- 背景はシンプルな単色グラデーションにしてください
- アニメ・ゲーム風の迫力あるポーズで描いてください
- 正方形の構図で、キャラクターを画面中央に大きく配置してください"""

    last_error = None
    for attempt in range(3):
        try:
            response = await client.aio.models.generate_content(
                model=IMAGE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    img_b64 = base64.b64encode(part.inline_data.data).decode()
                    return f"data:{part.inline_data.mime_type};base64,{img_b64}"
            raise RuntimeError("No image in response")
        except Exception as e:
            last_error = e
            if attempt < 2:
                await asyncio.sleep(1)

    raise RuntimeError(f"Image generation failed after 3 attempts: {last_error}")


async def generate_random_character() -> dict:
    """Generate a random battle character without an image input."""
    if client is None:
        raise RuntimeError("Gemini client not initialized")

    prompt = """ランダムなバトルゲームのキャラクターを1体生成してください。
創造的でユニークなキャラクターにしてください。

属性は必ず以下の4つの中から1つ選んでください:
- 斬撃
- 打撃
- 盾
- 毒

以下のJSON形式で出力してください:
{
  "name": "キャラクター名（日本語）",
  "hp": 数値(50-200),
  "attack": 数値(10-100),
  "defense": 数値(10-100),
  "speed": 数値(10-100),
  "special_move": "必殺技名（日本語）",
  "attribute": "属性名",
  "power": 数値(30-90),
  "description": "キャラクターの説明（2-3文、日本語）"
}

ステータスの合計は250-400の範囲にしてください。"""

    last_error = None
    for attempt in range(3):
        try:
            response = await client.aio.models.generate_content(
                model=MODEL,
                contents=[types.Content(parts=[types.Part.from_text(text=prompt)])],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=1.2,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            last_error = e
            if attempt < 2:
                await asyncio.sleep(1)

    raise RuntimeError(f"Random character generation failed after 3 attempts: {last_error}")


async def resolve_battle(character1: dict, character2: dict) -> dict:
    """Resolve a battle between two characters and generate narration."""
    if client is None:
        raise RuntimeError("Gemini client not initialized")

    attr1 = character1.get('attribute', '不明')
    power1 = character1.get('power', '?')
    attr2 = character2.get('attribute', '不明')
    power2 = character2.get('power', '?')

    prompt = f"""2人のキャラクターのバトルを審判してください。

【プレイヤー1のキャラクター】
名前: {character1['name']}
属性: {attr1}（パワー: {power1}）
HP: {character1['hp']}
攻撃力: {character1['attack']}
防御力: {character1['defense']}
素早さ: {character1['speed']}
必殺技: {character1['special_move']}
説明: {character1['description']}

【プレイヤー2のキャラクター】
名前: {character2['name']}
属性: {attr2}（パワー: {power2}）
HP: {character2['hp']}
攻撃力: {character2['attack']}
防御力: {character2['defense']}
素早さ: {character2['speed']}
必殺技: {character2['special_move']}
説明: {character2['description']}

ステータスと属性の相性を考慮して、勝者を決定してください。
属性の相性: 斬撃→盾に強い、打撃→斬撃に強い、盾→毒に強い、毒→打撃に強い

以下のJSON形式で出力してください:
{{
  "winner": 1 または 2（勝者のプレイヤー番号）,
  "reason": "勝敗の決め手（日本語、1文、熱い表現で）"
}}"""

    last_error = None
    for attempt in range(3):
        try:
            response = await client.aio.models.generate_content(
                model=MODEL,
                contents=[types.Content(parts=[types.Part.from_text(text=prompt)])],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=1.0,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            last_error = e
            if attempt < 2:
                await asyncio.sleep(1)

    raise RuntimeError(f"Gemini API failed after 3 attempts: {last_error}")
