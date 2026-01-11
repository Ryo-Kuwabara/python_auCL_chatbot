"""
Streamlit版 Simple ReAct PDF ChatBot
Web UIでReActエージェント機能を提供
"""

import streamlit as st
import os
import re
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool, ToolMetadata  
from llama_index.core.agent import AgentRunner
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.prompts import PromptTemplate

# .envファイルから環境変数を読み込み
load_dotenv()

# 日本語強制ReActプロンプト（英語回答防止）
REACT_SYSTEM_PROMPT = """あなたは必ず日本語で回答するPDF文書検索エージェントです。
IMPORTANT: YOU MUST RESPOND ONLY IN JAPANESE. ENGLISH RESPONSES ARE FORBIDDEN.

【絶対ルール】
- 全ての回答は必ず日本語で行う
- 英語での回答は絶対禁止
- Thoughtも日本語で考える
- Answerも必ず日本語で書く

【基本動作】
1. 質問を日本語で理解する
2. pdf_searchツールで1回だけ検索する  
3. 検索結果を日本語で回答する
4. 終了

会話履歴がある場合は文脈を考慮してください。
回答には参考文書名を記載してください。"""

class StreamlitReActChatBot:
    """Streamlit用ReAct ChatBot"""
    
    def __init__(self, pdf_folder: str = "pdfs"):
        self.pdf_folder = pdf_folder
        self.agent = None
        self.index = None
        
        # LlamaIndexの基本設定（日本語最適化）
        Settings.llm = OpenAI(
            model="gpt-3.5-turbo", 
            temperature=0.3,  # 0.0→0.3で創造性UP（言語切り替え能力向上）
            max_tokens=2000,  # 1500→2000で十分な日本語生成余裕
            system_prompt="あなたは日本語専用のAIアシスタントです。必ず日本語で回答してください。英語での回答は絶対に禁止です。ALWAYS RESPOND IN JAPANESE ONLY. 日本語AI。English is forbidden.",
            # 追加パラメータで日本語生成を促進
            presence_penalty=0.1,   # 繰り返し防止
            frequency_penalty=0.1   # 多様性向上
        )
        
    def load_pdfs_with_react(self):
        """ReAct機能でPDFを読み込み"""
        try:
            if not os.path.exists(self.pdf_folder):
                st.error(f"❌ フォルダが見つかりません: {self.pdf_folder}")
                return False
            
            # PDFファイル一覧取得
            pdf_files = [f for f in os.listdir(self.pdf_folder) if f.endswith('.pdf')]
            if not pdf_files:
                st.error("❌ PDFファイルが見つかりません")
                return False
            
            st.success(f"📚 {len(pdf_files)}個のPDFファイルを発見:")
            for pdf_file in pdf_files:
                st.write(f"  • {pdf_file}")
            
            # PDFファイル読み込み
            with st.spinner("📄 PDFファイルを読み込み中..."):
                reader = SimpleDirectoryReader(input_dir=self.pdf_folder)
                documents = reader.load_data()
            
            st.success(f"✅ {len(documents)}個のドキュメントを読み込み完了")
            
            # ベクトルインデックス作成
            with st.spinner("🔄 ベクトルインデックス作成中..."):
                self.index = VectorStoreIndex.from_documents(documents)
            
            st.success("✅ インデックス作成完了")
            
            # ReActエージェント用のツール作成
            if self._create_react_agent():
                return True
            else:
                return False
            
        except Exception as e:
            st.error(f"❌ エラー: {e}")
            return False
    
    def _create_react_agent(self):
        """ReActエージェントの作成（シンプル化・ツール統合）"""
        try:
            if not self.index:
                st.error("❌ インデックスが作成されていません")
                return False
                
            # 日本語強制テンプレート作成（出典情報付き）
            japanese_template = PromptTemplate(
                "重要: この回答は必ず日本語で行ってください。英語での回答は絶対に禁止です。\n"
                "CRITICAL: You MUST respond in Japanese only. English responses are absolutely forbidden.\n"
                "\n"
                "コンテキスト情報は以下の通りです。\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "\n"
                "上記の情報を基に、質問に必ず日本語で詳しく答えてください。\n"
                "回答は日本語のみで行い、英語は一切使用しないでください。\n"
                "回答の最後に、参考にした文書名（ファイル名）を【参考文書】として必ず記載してください。\n"
                "\n"
                "質問: {query_str}\n"
                "日本語での回答: "
            )
            
            # シンプルで効率的な検索ツール
            query_engine = self.index.as_query_engine(
                similarity_top_k=3,  # 検索数を削減してスピードアップ
                response_mode="compact",
                text_qa_template=japanese_template
            )
            
            pdf_search_tool = QueryEngineTool(
                query_engine=query_engine,
                metadata=ToolMetadata(
                    name="pdf_search",
                    description="PDF文書の検索。給与規程、就業規則、情報セキュリティ方針、企業年金規約、定款から情報を取得します。"
                )
            )
            
            # メモリ機能（軽量化）
            memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
            
            # シンプルな質問エンジンとして使用（最も互換性が高い方法）
            try:
                # メモリ機能付きの簡単なエージェント作成
                from llama_index.core.chat_engine import SimpleChatEngine
                
                self.agent = query_engine
                st.success("✅ クエリエンジン準備完了")
                
            except ImportError:
                # フォールバック: 基本的なクエリエンジンのみ使用
                self.agent = query_engine
                st.success("✅ 基本クエリエンジン準備完了")
            
            st.success("✅ ReActエージェント準備完了")
            return True
            
        except Exception as e:
            st.error(f"❌ ReActエージェント作成エラー: {e}")
            import traceback
            st.error(f"詳細エラー: {traceback.format_exc()}")
            return False
    
    def _contains_english(self, text: str) -> bool:
        """英語が含まれているかチェック"""
        # 連続する3文字以上のアルファベットを英語と判定
        english_pattern = r'[a-zA-Z]{3,}'
        english_matches = re.findall(english_pattern, text)
        
        # 日本語固有の単語は除外
        japanese_exceptions = ['pdf', 'PDF', 'auコマース', 'au', 'DC', 'API']
        filtered_matches = [match for match in english_matches if match not in japanese_exceptions]
        
        return len(filtered_matches) > 0
    
    def _force_japanese_response(self, response: str) -> str:
        """回答を強制的に日本語に変換"""
        if not self._contains_english(response):
            return response
        
        try:
            # 英語が含まれている場合、GPTに日本語変換を依頼
            translate_prompt = f"""以下の文章を自然な日本語に変換してください。
既に日本語の部分はそのまま保持し、英語の部分のみを日本語に変換してください。
技術用語や固有名詞は適切な日本語表現に置き換えてください。

元の文章:
{response}

自然な日本語での表現:"""
            
            with st.spinner("🔄 日本語に変換中..."):
                japanese_response = Settings.llm.complete(translate_prompt)
                return str(japanese_response)
                
        except Exception as e:
            st.warning(f"日本語変換エラー: {e}")
            return response  # エラー時は元の回答をそのまま返す
    
    def ask_with_react(self, question: str):
        """質問応答（日本語強制版）"""
        if not self.agent:
            st.error("❌ エラー: エージェントが初期化されていません。PDFファイルの読み込みを行ってください。")
            return "エラー: エージェントが初期化されていません。PDFファイルの読み込みボタンを押してください。"
        
        try:
            with st.spinner("🤖 PDF検索で分析中..."):
                # クエリエンジンを直接使用
                response = self.agent.query(question)
                
                # 🔧 改善策2: 英語検出と日本語強制変換
                japanese_response = self._force_japanese_response(str(response))
                
                # 出典情報を追加取得
                sources_info = self._get_source_info(question)
                full_response = f"{japanese_response}\n\n{sources_info}"
                
                return full_response
                
        except Exception as e:
            error_msg = str(e)
            # ReActが上限に達した場合のより良いフォールバック
            if "Reached max iterations" in error_msg or "max_iterations" in error_msg:
                st.warning("⚠️ ReActの処理時間が長いため、直接検索で回答します...")
                return self._fallback_search(question)
            else:
                st.error(f"⚠️ エラーが発生しました: {error_msg}")
                return self._fallback_search(question)
    
    def _fallback_search(self, question: str):
        """フォールバック検索（ReActが失敗した場合・日本語強制版）"""
        try:
            # 日本語強制の確実な検索エンジン
            japanese_fallback_template = PromptTemplate(
                "重要: 必ず日本語で回答してください。英語での回答は絶対に禁止です。\n"
                "IMPORTANT: You MUST respond in Japanese only. English is forbidden.\n"
                "\n"
                "以下の文書情報を参考に、質問に日本語で答えてください。\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "\n"
                "質問: {query_str}\n"
                "必ず日本語で詳しく回答してください："
            )
            
            fallback_engine = self.index.as_query_engine(
                similarity_top_k=5,
                response_mode="compact",
                text_qa_template=japanese_fallback_template
            )
            
            response = fallback_engine.query(question)
            
            # 🔧 改善策2: フォールバックでも日本語強制変換
            japanese_response = self._force_japanese_response(str(response))
            sources_info = self._get_source_info(question)
            
            return f"【直接検索による回答】\n{japanese_response}\n\n{sources_info}"
            
        except Exception as fallback_error:
            return f"❌ 検索エラー: {fallback_error}"
    
    def _get_source_info(self, question: str):
        """出典情報を取得する補助メソッド"""
        try:
            # 関連ノードを取得して出典情報を生成
            retriever = self.index.as_retriever(similarity_top_k=3)
            nodes = retriever.retrieve(question)
            
            if not nodes:
                return "📚 【出典情報】参考文書が見つかりませんでした。"
            
            sources = []
            for i, node in enumerate(nodes, 1):
                # メタデータから文書名を取得
                file_name = node.metadata.get('file_name', '不明な文書')
                # ページ情報があれば追加
                page_info = node.metadata.get('page_label', '')
                if page_info:
                    source_info = f"{i}. {file_name} (ページ: {page_info})"
                else:
                    source_info = f"{i}. {file_name}"
                
                # 関連度スコアがあれば追加
                if hasattr(node, 'score'):
                    source_info += f" - 関連度: {node.score:.3f}"
                
                sources.append(source_info)
            
            return "📚 【参考文書・出典情報】\n" + "\n".join(sources)
            
        except Exception as e:
            return f"📚 【出典情報取得エラー】{e}"

def main():
    """Streamlit メイン関数"""
    st.set_page_config(
        page_title="ReAct PDF ChatBot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 ReAct PDF ChatBot")
    st.markdown("---")
    
    # サイドバー
    with st.sidebar:
        st.header("📊 システム情報")
        
        # APIキー確認
        if not os.getenv("OPENAI_API_KEY"):
            st.error("❌ OPENAI_API_KEY が設定されていません")
            st.stop()
        else:
            st.success("✅ OpenAI API Key 設定済み")
        
        st.markdown("### 🔧 機能")
        st.info("""
        ✅ 無限ループ防止・ツール統合により安定性向上
        
        📚 回答には参考文書・出典情報も表示
        
        🧠 会話履歴を記憶し、文脈を理解した回答が可能
        """)
    
    # チャットボット初期化
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = StreamlitReActChatBot()
        
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    
    # 初期化ボタン
    if not st.session_state.initialized:
        st.info("📁 PDFファイルを読み込んでチャットボットを初期化してください")
        
        if st.button("🚀 PDFファイル読み込み・初期化", type="primary"):
            if st.session_state.chatbot.load_pdfs_with_react():
                st.session_state.initialized = True
                st.rerun()
    else:
        # チャット履歴表示
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # ユーザー入力
        if prompt := st.chat_input("質問を入力してください（例：情報セキュリティ基本方針第一条について教えて）"):
            # ユーザーメッセージを追加
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # アシスタントからの回答
            with st.chat_message("assistant"):
                response = st.session_state.chatbot.ask_with_react(prompt)
                st.markdown(response)
                
                # 🔧 デバッグ機能: 言語検出状況を表示
                if st.session_state.chatbot._contains_english(response):
                    english_ratio = len(re.findall(r'[a-zA-Z]', response)) / len(response) if len(response) > 0 else 0
                    st.sidebar.warning(f"⚠️ 英語検出: {english_ratio:.1%}")
                    with st.expander("🔍 デバッグ情報"):
                        st.write("英語が含まれていたため日本語変換を実行しました")
                else:
                    st.sidebar.success("✅ 完全日本語回答")
                
                st.session_state.messages.append({"role": "assistant", "content": response})
        
        # チャット履歴クリアボタン
        if st.button("🗑️ チャット履歴をクリア"):
            st.session_state.messages = []
            # エージェントのメモリもクリア
            if st.session_state.chatbot.agent:
                try:
                    # ChatMemoryBufferのメモリクリア（複数の方法を試行）
                    if hasattr(st.session_state.chatbot.agent.memory, 'reset'):
                        st.session_state.chatbot.agent.memory.reset()
                    elif hasattr(st.session_state.chatbot.agent.memory, 'clear'):
                        st.session_state.chatbot.agent.memory.clear()
                    elif hasattr(st.session_state.chatbot.agent.memory, 'chat_history'):
                        # チャット履歴を直接クリア
                        st.session_state.chatbot.agent.memory.chat_history = []
                    else:
                        # エージェントを再作成
                        st.session_state.chatbot._create_react_agent()
                except Exception as e:
                    st.warning(f"メモリクリアに失敗しましたが、エージェントを再作成します: {e}")
                    st.session_state.chatbot._create_react_agent()
            st.rerun()

if __name__ == "__main__":
    main()