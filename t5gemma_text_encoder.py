import torch
import logging
import re

# ロガーの設定
logger = logging.getLogger("LLM-SDXL-Adapter")

class T5GEMMATextEncoder:
    """
    ComfyUI node that encodes text using a loaded Language Model.
    This final version automatically cleans the prompt (replaces commas with spaces)
    and outputs the attention_mask to maximize the adapter's performance.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        ノードの入力タイプを定義します。
        """
        return {
            "required": {
                "model": ("LLM_MODEL",),
                "tokenizer": ("LLM_TOKENIZER",),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "masterpiece, best quality, 1girl, anime style"
                }),
            }
        }
    
    RETURN_TYPES = ("LLM_HIDDEN_STATES", "LLM_INPUT_IDS", "LLM_ATTENTION_MASK", "STRING")
    RETURN_NAMES = ("hidden_states", "input_ids", "attention_mask", "info")
    FUNCTION = "encode_text"
    CATEGORY = "llm_sdxl"
    
    def encode_text(self, model, tokenizer, text):
        """
        テキストをエンコードし, hidden states, input_ids, そして重要なattention_maskを返します。
        プロンプト内のコンマは自動的にスペースに置換されます。
        """
        try:
            # モデルが配置されているデバイス（例: 'cuda:0'）を取得
            device = next(model.parameters()).device
            
            # --- ★★★ 最終改善点 ★★★ ---
            # プロンプト内のコンマをスペースに置換し、アダプターの性能を最大化します。
            # "1girl, masterpiece" -> "1girl masterpiece"
            # 複数のスペースが連続しないように正規表現で堅牢に処理します。
            processed_text = re.sub(r'\s*,\s*', ' ', text).strip()
            
            # トークナイザーはCPU上で実行し、生成されたテンソルのみをGPUに転送します。
            inputs = tokenizer(
                processed_text, # 整形後のテキストを使用
                return_tensors="pt",       # PyTorchテンソル形式で返す
                padding="max_length",      # シーケンスを最大長までパディングする
                max_length=512,            # 最大シーケンス長
                truncation=True,           # 最大長を超える場合は切り詰める
            )

            # 必要なテンソルをモデルと同じデバイスに移動
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            # --- デバッグ情報の出力 ---
            print("--- Tokenizer Debug ---")
            # トークナイズされたIDを再度デコードして、モデルがどのように入力を解釈したかを確認
            decoded_text = tokenizer.decode(input_ids[0])
            print(f"Decoded Input: {decoded_text}")
            
            # 勾配計算を無効にして、メモリ使用量と計算速度を最適化
            with torch.no_grad():
                # attention_maskをモデルに渡し、パディング部分を無視させることができます。
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
            # モデルの最終層の隠れ状態（hidden states）を取得し、データ型をfloatに変換
            hidden_states = outputs['last_hidden_state'].to(torch.float)
            
            # UIに表示するための情報文字列を作成
            info = f"Original: {text[:50]}...\nProcessed: {processed_text[:50]}..."
            
            logger.info(f"Encoded text with shape: {hidden_states.shape}")
            
            return (hidden_states, input_ids, attention_mask, info)
            
        except Exception as e:
            # エラーが発生した場合はログに出力し、例外を発生させる
            logger.error(f"Failed to encode text: {str(e)}")
            raise Exception(f"Text encoding failed: {str(e)}")

# ComfyUIにこのカスタムノードを登録するためのマッピング
NODE_CLASS_MAPPINGS = {
    "T5GEMMATextEncoder": T5GEMMATextEncoder
}

# ComfyUIのメニューに表示されるノード名を定義
NODE_DISPLAY_NAME_MAPPINGS = {
    "T5GEMMATextEncoder": "T5Gemma Text Encoder"
}
