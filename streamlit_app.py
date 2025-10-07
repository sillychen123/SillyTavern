"""SillyTavern 核心工具的 Streamlit 调试界面。"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional

import streamlit as st

from sillytavern_core.character import load_character_card
from sillytavern_core.model_interface import ModelConfig, ModelInterface
from sillytavern_core.models import ConversationTurn, PromptContext
from sillytavern_core.prompt import PromptBuilder
from sillytavern_core.world import load_world_info


def _persist_upload(upload) -> Path:
    suffix = Path(upload.name).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.getbuffer())
        tmp.flush()
        return Path(tmp.name)


def _init_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("memories", [])
    st.session_state.setdefault("world_info", [])
    st.session_state.setdefault("character_card", None)


def main() -> None:
    st.set_page_config(page_title="SillyTavern 调试面板", layout="wide")
    st.title("SillyTavern 核心调试面板")

    _init_state()

    builder = PromptBuilder()

    with st.sidebar:
        st.header("配置")
        api_key = st.text_input("OpenRouter / OpenAI API Key", type="password")
        base_url = st.text_input("Base URL", value="https://openrouter.ai/api/v1")
        model_name = st.text_input("模型名称", value="google/gemini-2.5-pro")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        max_tokens = st.number_input("最大输出 tokens", min_value=0, value=0, step=50)
        referer = st.text_input("HTTP Referer", value="")
        title = st.text_input("X-Title", value="")

        st.divider()
        st.subheader("资源加载")
        card_upload = st.file_uploader("上传角色卡", type=["json", "png"])
        if card_upload is not None:
            card_path = _persist_upload(card_upload)
            st.session_state["character_card"] = load_character_card(card_path)
            st.success(f"已加载角色卡：{st.session_state['character_card'].name}")

        world_uploads = st.file_uploader("上传世界信息", type=["json"], accept_multiple_files=True)
        if world_uploads:
            entries = []
            for upload in world_uploads:
                world_path = _persist_upload(upload)
                entries.extend(load_world_info(world_path))
            st.session_state["world_info"] = entries
            st.success(f"已加载 {len(entries)} 条世界信息")

        st.caption("如未配置模型参数，将仅展示提示词组装结果，便于调试。")
        if st.button("清空对话", use_container_width=True):
            st.session_state["history"] = []
            st.session_state.pop("last_prompt", None)

    system_prompt = st.text_area("系统提示", value=st.session_state.get("system_prompt", ""))
    persona = st.text_area("用户 Persona", value=st.session_state.get("persona", ""))
    story_in_chat = st.checkbox("故事模板放入对话", value=st.session_state.get("story_in_chat", False))

    st.session_state["system_prompt"] = system_prompt
    st.session_state["persona"] = persona
    st.session_state["story_in_chat"] = story_in_chat

    conversation_container = st.container()

    with st.form("chat-input", clear_on_submit=True):
        user_message = st.text_area("输入消息", height=120)
        submitted = st.form_submit_button("发送")

    if submitted and user_message.strip():
        context = PromptContext(
            system_prompt=system_prompt or None,
            character=st.session_state["character_card"],
            world_info=st.session_state["world_info"],
            memories=st.session_state["memories"],
            history=st.session_state["history"],
            user_message=user_message,
            persona=persona or None,
            story_in_chat=story_in_chat,
        )

        response_text: Optional[str] = None
        if api_key and model_name:
            headers = {k: v for k, v in {"HTTP-Referer": referer, "X-Title": title}.items() if v}
            config = ModelConfig(
                model=model_name,
                api_key=api_key,
                base_url=base_url or None,
                temperature=temperature,
                max_output_tokens=max_tokens or None,
                extra_headers=headers or None,
            )
            try:
                interface = ModelInterface(config, builder=builder)
                response = interface.complete(context)
                if isinstance(response, str):
                    response_text = response
                else:
                    response_text = "".join(response)
            except Exception as exc:  # pragma: no cover - UI 呈现错误
                st.error(f"模型调用失败：{exc}")
        else:
            st.info("未配置模型，将仅展示提示词。")

        new_history: List[ConversationTurn] = [*st.session_state["history"]]
        new_history.append(ConversationTurn(role="user", content=user_message))
        if response_text:
            new_history.append(ConversationTurn(role="assistant", content=response_text))
        st.session_state["history"] = new_history

        st.session_state["last_prompt"] = builder.build_prompt(context)

    with conversation_container:
        st.subheader("对话记录")
        if st.session_state["history"]:
            for turn in st.session_state["history"]:
                role = "assistant" if turn.role.lower() in {"assistant", "bot"} else "user" if turn.role.lower() == "user" else "system"
                with st.chat_message(role):
                    st.markdown(turn.content)
        else:
            st.write("暂无对话，输入上方表单开始调试。")

    with st.expander("最新提示词预览", expanded=False):
        prompt_preview = st.session_state.get("last_prompt")
        if prompt_preview:
            st.code(prompt_preview)
        else:
            st.caption("发送消息后可查看提示词组合结果。")


if __name__ == "__main__":
    main()
