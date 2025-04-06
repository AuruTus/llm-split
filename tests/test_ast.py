import transformers
from llm_split.model_ast import DecoderAST
from llm_split.utils import get_read_write_vars


def test_pytest():
    t = DecoderAST(transformers.LlamaModel)
    decoder_for_block, decoder_blocks_idx = t.find_decoder_blocks()
    read, write = get_read_write_vars(decoder_for_block)
    print(read)
    print("===================")
    print(write)
