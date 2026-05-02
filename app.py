import os
from dotenv import load_dotenv
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# .envを読み込む（スクリプトの最初に必ず呼ぶ）
load_dotenv()

# 環境変数から設定を取得
base_url = os.getenv("LMSTUDIO_BASE_URL")
api_key = os.getenv("LMSTUDIO_API_KEY")
model_name = os.getenv("LMSTUDIO_MODEL", "google/gemma-4")

# LLM設定
llm = ChatOpenAI(
    base_url=base_url,
    api_key=api_key,
    model=model_name,
    temperature=float(os.getenv("LLM_TEMPERATURE", 0.7)),
    max_tokens=int(os.getenv("LLM_MAX_TOKENS", 2048)),
)


# 状態定義
class NovelState(TypedDict):
    world: str
    characters: str
    plot: str
    text: str
    review: str
    target_length: int
    loop_count: int

# ① 執筆ノード
def write_node(state: NovelState):
    prompt = f"""
    あなたは小説家です。以下の設定に従い、小説の続きを執筆してください。

    # 世界観:
    {state['world']}

    # キャラクター:
    {state['characters']}

    # ストーリー:
    {state['plot']}

    # 現在の文章:
    {state['text']}

    # 制約
    * 出力は小説本文のみ
    * 解説、メタ発言、コメントは禁止
    * 「続きは〜」などの説明は禁止

    続きを自然につながる形で執筆してください
    """
    res = llm.invoke(prompt)
    text = state["text"]
    contents = text + res.content

    print("=== write node ===")
    print(contents)

    return {"text": contents}


# ② 校正ノード
def review_node(state: NovelState):
    prompt = f"""
    以下の小説を批評してください。改善点を具体的に指摘してください。

    # 批評対象の文章
    {state['text']}

    # 出力形式（厳守）
    * 改善点:
    * 修正方針:
    """
    res = llm.invoke(prompt)
    review_content = res.content
    print("=== review node ===")
    print(review_content)

    return {"review": review_content}


# ③ 修正ノード
def revise_node(state: NovelState):
    prompt = f"""
    あなたはプロの編集者です。以下の小説を改善してください。

    # 小説
    {state['text']}

    # 改善指示
    {state['review']}

    # 制約（重要）
    * 全文を書き直すこと
    * 「続きを書く」は禁止
    * 既存の内容をベースに改善する
    * 解説やコメントは一切出力しない
    * 出力は小説本文のみ

    改善済みの全文を出力してください。
    """
    res = llm.invoke(prompt)
    revise_content = res.content
    print("=== revise node ===")
    print(revise_content)

    return {"text": revise_content, "loop_count": state["loop_count"] + 1}  # 執筆回数をここでカウントアップ


# ④ 終了判定
def judge_node(state: NovelState):
    print("=== judge node ===")

    current_text_length = len(state["text"])
    max_text_length = state["target_length"]

    print(f"Current Count: {current_text_length}, limit={max_text_length}")

    next_state = "end"
    if state["loop_count"] >= 3:
        next_state = "end"
    elif current_text_length < max_text_length:
        next_state = "continue"
    else:
        next_state = "end"

    if next_state == "continue":
        loop_count = state["loop_count"]
        print(f"loop_count: {loop_count}")
        return next_state
    else:
        print("End of Work...")
        return next_state


def main(world, characters, plot, text, target_length):
    """
    Agentグラフ構築
    """
    builder = StateGraph(NovelState)
    builder.add_node("write", write_node)
    builder.add_node("review", review_node)
    builder.add_node("revise", revise_node)

    builder.set_entry_point("write")

    builder.add_edge("write", "review")
    builder.add_edge("review", "revise")
    # builder.add_edge("revise", "write") # ★←これのせいで無限ループしていた。

    # 条件分岐
    builder.add_conditional_edges(
        "revise",
        judge_node,
        {
            "continue": "write",
            "end": END
        }
    )

    graph = builder.compile()
    result = graph.invoke({
        "world": world,
        "characters": characters,
        "plot": plot,
        "text": text,
        "target_length": target_length,
        "loop_count": 0,
    })
    print(result["text"])


if __name__ == '__main__':
    world = """
    魔法と機械が共存する世界
    """

    characters = """
    若い魔法技師の少女とAIロボット
    """

    plot = """
    失われた古代兵器を巡る冒険
    """

    text = ""
    target_length = 1500

    main(world=world, characters=characters, plot=plot, text=text, target_length=target_length)
