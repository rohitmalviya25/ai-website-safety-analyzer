
from features import FEATURE_NAMES, extract_features


def test_normal_url():
    url = "https://example.com/login"
    features = extract_features(url)

    assert len(features) == len(FEATURE_NAMES)
    assert features["https_flag"] == 1
    assert features["domain_name_length"] == len("example.com")
    assert features["path_length"] == len("/login")


def test_bare_hostname():
    features = extract_features("example.com")

    assert features["https_flag"] == 0
    assert features["domain_name_length"] == len("example.com")


def test_ip_address():
    features = extract_features("http://192.168.1.1/login")

    assert features["has_ip_address"] == 1


def test_invalid_inputs():
    for url in ["", "   ", "ftp://example.com", "https://"]:
        try:
            extract_features(url)
            assert False, f"Expected ValueError for {url!r}"
        except ValueError:
            pass


if __name__ == "__main__":
    test_normal_url()
    test_bare_hostname()
    test_ip_address()
    test_invalid_inputs()

    print("All feature extraction tests passed!")
