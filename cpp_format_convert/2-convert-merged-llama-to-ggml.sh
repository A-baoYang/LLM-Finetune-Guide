COMBINED_DIR=/home/jovyan/stock-pred/temp_model
COMBINED_MODEL=multi-ee-llama-1024-3e-4-combined
LLAMACPP_DIR=/home/jovyan/gpt/open_gpt/llama.cpp

mv $COMBINED_DIR/$COMBINED_MODEL $LLAMACPP_DIR/zh-models/$COMBINED_MODEL
mv $LLAMACPP_DIR/zh-models/$COMBINED_MODEL/tokenizer.model $LLAMACPP_DIR/zh-models/
python convert.py $LLAMACPP_DIR/zh-models/$COMBINED_MODEL/