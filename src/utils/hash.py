import hashlib
import base64

#  ---


def hash_256(string: str):
    return hashlib.sha256(str(string).encode('utf-8')).hexdigest()


def hash_256_tob(string: str):
    return hashlib.sha256(str(string).encode('utf-8')).digest()


def encode_base64(bytes: bytes):
    return base64.urlsafe_b64encode(bytes).decode('utf-8')
