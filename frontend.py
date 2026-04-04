import os

import requests
import streamlit as st


def main():
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try: 
            backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
            response = requests.post(
                f"{backend_url}/chat",
                json={"query": prompt},
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
            result = payload.get("answer", "")
            sources = payload.get("sources", [])

            # Display answer
            st.chat_message('assistant').markdown(result)

            # Display sources separately with consistent formatting
            with st.expander("View Source Documents"):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"**Source {i}:**")
                    st.text(source)
                    st.divider()

            # Store in session
            st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
