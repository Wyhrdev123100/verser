import uatp.hardware as hw


def test_detect():
    info = hw.detect()
    assert info.device in ("cpu", "cuda", "mps")
    assert info.cpu_count >= 1


def test_recommend():
    info = hw.detect()
    rec = hw.recommend_config(info, mode="safe")
    assert rec["micro_batch_size"] >= 1
    rec2 = hw.recommend_config(info, mode="performance")
    assert rec2["sequence_length"] >= 128