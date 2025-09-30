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
    
    ### MODIFIED ###
    It now allows selecting which hidden layer to output.
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
                ### MODIFIED ###: Add a dropdown to select the output layer.
                "layer_index": (
                    [
                        "-1 (last layer)", 
                        "-2 (penultimate layer)", 
                        "-3",
                        "-4"
                    ], {
                    "default": "-1 (last layer)"
                }),
            }
        }
    
    RETURN_TYPES = ("LLM_HIDDEN_STATES", "LLM_INPUT_IDS", "LLM_ATTENTION_MASK", "STRING")
    RETURN_NAMES = ("hidden_states", "input_ids", "attention_mask", "info")
    FUNCTION = "encode_text"
    CATEGORY = "llm_sdxl"
    
    ### MODIFIED ###: The method now accepts 'layer_index' as an argument.
    def encode_text(self, model, tokenizer, text, layer_index):
        """
        テキストをエンコードし, 指定された層のhidden states, input_ids, そしてattention_maskを返します。
        プロンプト内のコンマは自動的にスペースに置換されます。
        """
        try:
            # モデルが配置されているデバイス（例: 'cuda:0'）を取得
            device = next(model.parameters()).device
            
            # --- ★★★ 最終改善点 ★★★ ---
            # プロンプト内のコンマをスペースに置換し、アダプターの性能を最大化します。
            processed_text = re.sub(r'\s*,\s*', ' ', text).strip()
            
            # トークナイザーはCPU上で実行し、生成されたテンソルのみをGPUに転送します。
            inputs = tokenizer(
                processed_text,
                return_tensors="pt",
                padding="max_length",
                max_length=512,
                truncation=True,
            )

            # 必要なテンソルをモデルと同じデバイスに移動
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            # --- デバッグ情報の出力 ---
            print("--- Tokenizer Debug ---")
            decoded_text = tokenizer.decode(input_ids[0])
            print(f"Decoded Input: {decoded_text}")
            
            # 勾配計算を無効にして、メモリ使用量と計算速度を最適化
            with torch.no_grad():
                ### MODIFIED ###: Set 'output_hidden_states=True' to get all layer outputs.
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    output_hidden_states=True
                )
                
            ### MODIFIED ###: Select the hidden state from the specified layer.
            # layer_indexの文字列（例: "-1 (last layer)"）から数値（-1）を抽出
            selected_layer = int(re.match(r"(-?\d+)", layer_index).group(1))

            # outputs.hidden_statesは全層の出力を含むタプル
            # 最後の要素が'last_hidden_state'と同じ
            hidden_states = outputs.hidden_states[selected_layer].to(torch.float)
            
            # UIに表示するための情報文字列を作成
            info = (f"Original: {text[:50]}...\n"
                    f"Processed: {processed_text[:50]}...\n"
                    f"Outputting from Layer: {selected_layer}")
            
            logger.info(f"Encoded text with shape: {hidden_states.shape} from layer {selected_layer}")
            
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
