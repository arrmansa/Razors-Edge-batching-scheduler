import numpy as np
import time

def accurate_sleep(seconds):
    start_time = time.perf_counter_ns()
    time.sleep(max(seconds-0.1, 0))
    end_time = int(start_time + seconds * 1e9 - 1e6)
    while time.perf_counter_ns() < end_time:
        pass

def model_time(text_count: int, token_size: int) -> float:
    """
    Model performance based on matrix matrix multiplication complexity.
    1. At first the bottleneck is memory - (putting big model matrix and token matrix into memory takes time) - 0.02
    2. Later the bottleneck is compute - (quadratic on token size and linear on text count) - upto 0.2 at 1000 token 1 text, minimum of 0.01
    """
    MEMORY_OVERHEAD = 0.02
    COMPUTE_UNIT = 2e-7
    expected_model_time = MEMORY_OVERHEAD + max(0.01, text_count * (token_size ** 2) * COMPUTE_UNIT)
    return expected_model_time


class DummyTokenizer:

    @staticmethod
    def encode(texts: list[str]):
        input_ids = np.zeros((len(texts), max(map(len, texts))), dtype=np.int64)
        attention_mask = input_ids.copy()
        for i, text in enumerate(texts):
            input_ids[i, :len(text)] = list(map(ord, text))
            attention_mask[i, :len(text)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @staticmethod
    def decode(input_ids: np.ndarray) -> list[str]:
        """Decode tokenized array back to text strings."""
        if len(input_ids.shape) == 1:
            input_ids = [input_ids]
        texts = []
        for tokens in input_ids:
            text = "_" + "".join(map(chr, tokens))
            texts.append(text.strip(chr(0))[1:])
        return texts
    
    def __call__(self, texts: list[str]):
        return self.encode(texts)

def dummy_model_encode(*, input_ids, attention_mask):
    embedding = np.zeros((input_ids.shape[0], 64), dtype=np.int64)
    trunc_size = min(input_ids.shape[1], 64)
    embedding[:, :trunc_size] = input_ids[:, :trunc_size]
    accurate_sleep(model_time(*input_ids.shape))
    return embedding

if __name__ == "__main__":
    print("testing")
    print(DummyTokenizer.encode(["a", "bcd"]))
    print(dummy_model_encode(**DummyTokenizer.encode(["a", "bcd"])))
    print(DummyTokenizer.decode(DummyTokenizer.encode(["a", "bcd"])["input_ids"]))
